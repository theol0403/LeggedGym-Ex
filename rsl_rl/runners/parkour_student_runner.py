import os
import statistics
import time
from collections import deque

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict, get_load_path
from rsl_rl.algorithms import ParkourDistillation
from rsl_rl.modules import ActorCriticParkour, ActorCriticParkourStudent
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class ParkourStudentRunner(OnPolicyRunner):
    def _configure_torch_fast_path(self):
        if not str(self.device).startswith("cuda"):
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def _init_agent_and_algo(self):
        self._configure_torch_fast_path()
        self.teacher = self._load_frozen_teacher()
        actor_critic = ActorCriticParkourStudent(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_teacher_actor_obs,
            **self.policy_cfg,
        ).to(self.device)
        actor_critic.load_teacher_actor_weights(self.teacher.state_dict())
        alg_cfg = dict(self.alg_cfg)
        self._yaw_threshold = float(alg_cfg.pop("yaw_threshold", 0.6))
        self.alg = ParkourDistillation(actor_critic, device=self.device, **alg_cfg)

    def _init_storage(self):
        return None

    def _load_frozen_teacher(self):
        from legged_gym.utils.task_registry import task_registry

        teacher_task = self.cfg.get("teacher_task", "go2_parkour_teacher")
        teacher_env_cfg, teacher_train_cfg = task_registry.get_cfgs(teacher_task)
        teacher_log_root = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            teacher_train_cfg.runner.experiment_name,
        )
        teacher_ckpt = get_load_path(
            teacher_log_root,
            load_run=self.cfg.get("teacher_load_run", -1),
            checkpoint=self.cfg.get("teacher_ckpt", -1),
        )
        print(f"Loading frozen teacher from: {teacher_ckpt}")

        if teacher_env_cfg.env.num_observations != self.env.num_teacher_actor_obs:
            raise RuntimeError(
                "Teacher actor observation dimension mismatch: "
                f"teacher task has {teacher_env_cfg.env.num_observations}, "
                f"student task expects {self.env.num_teacher_actor_obs}."
            )
        if teacher_env_cfg.env.num_actions != self.env.num_actions:
            raise RuntimeError(
                "Teacher action dimension mismatch: "
                f"teacher task has {teacher_env_cfg.env.num_actions}, "
                f"student task expects {self.env.num_actions}."
            )

        teacher = ActorCriticParkour(
            num_actor_obs=teacher_env_cfg.env.num_observations,
            num_critic_obs=teacher_env_cfg.env.num_privileged_obs,
            num_actions=teacher_env_cfg.env.num_actions,
            **class_to_dict(teacher_train_cfg.policy),
        ).to(self.device)
        state = torch.load(teacher_ckpt, map_location=self.device)
        teacher.load_state_dict(state["model_state_dict"], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    def _teacher_actions(self, teacher_actor_obs):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_actor_obs)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._pre_learn(init_at_random_ep_len)
        obs, teacher_actor_obs, student_depth, _depth_updated = (
            t.to(self.device, non_blocking=True) for t in self.env.get_observations()
        )
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self.alg.actor_critic.detach_gru_hidden()
            self.alg.begin_rollout(self.num_steps_per_env)

            iter_start = time.time()
            for _ in range(self.num_steps_per_env):
                teacher_actions = self._teacher_actions(teacher_actor_obs)

                latent, _yaw_raw, yaw_scaled = self.alg.actor_critic.forward_depth(
                    student_depth.clone(), obs
                )
                oracle_heading = obs[:, 4:6]
                obs_actor = self.alg.actor_critic.apply_mts_yaw_to_obs(
                    obs, yaw_scaled, oracle_heading, self._yaw_threshold
                )
                student_actions = self.alg.actor_critic.actor_from_latent(obs_actor, latent)

                self.alg.store_step(
                    student_actions=student_actions,
                    teacher_actions=teacher_actions,
                    predicted_yaw=yaw_scaled,
                    oracle_yaw=oracle_heading,
                )

                (
                    obs,
                    teacher_actor_obs,
                    student_depth,
                    _depth_updated,
                    rewards,
                    dones,
                    infos,
                ) = self.env.step(student_actions.detach())

                obs = obs.to(self.device, non_blocking=True)
                teacher_actor_obs = teacher_actor_obs.to(self.device, non_blocking=True)
                student_depth = student_depth.to(self.device, non_blocking=True)
                rewards = rewards.to(self.device, non_blocking=True)
                dones = dones.to(self.device, non_blocking=True)

                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            metrics = self.alg.finish_rollout()
            iteration_time = time.time() - iter_start

            if self.log_dir is not None:
                self.log(
                    {
                        "it": it,
                        "num_learning_iterations": num_learning_iterations,
                        "iteration_time": iteration_time,
                        "ep_infos": ep_infos,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "mean_total_loss": metrics["loss"],
                        "mean_action_loss": metrics["action_loss"],
                        "mean_yaw_loss": metrics["yaw_loss"],
                    }
                )
                self._maybe_save_best_success_model(it, ep_infos)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        iteration_time = locs["iteration_time"]
        self.tot_time += iteration_time

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    value = ep_info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor([value], device=self.device)
                    if len(value.shape) == 0:
                        value = value.unsqueeze(0)
                    infotensor = torch.cat((infotensor, value.to(self.device)))
                mean_value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, mean_value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {mean_value:.4f}\n"""

        fps = int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-6))
        self.writer.add_scalar("Loss/total", locs["mean_total_loss"], locs["it"])
        self.writer.add_scalar("Loss/action", locs["mean_action_loss"], locs["it"])
        self.writer.add_scalar("Loss/yaw", locs["mean_yaw_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        heading = (
            f" \033[1m Learning iteration {locs['it']}/"
            f"{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        )
        log_string = (
            f"{'#' * width}\n"
            f"{heading.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (iter: {iteration_time:.3f}s)\n"
            f"{'Total loss:':>{pad}} {locs['mean_total_loss']:.4f}\n"
            f"{'Action loss:':>{pad}} {locs['mean_action_loss']:.4f}\n"
            f"{'Yaw loss:':>{pad}} {locs['mean_yaw_loss']:.4f}\n"
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} "
            f"{self.tot_time / max(locs['it'] + 1, 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
