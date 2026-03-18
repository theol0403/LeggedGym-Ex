import torch

from rsl_rl.algorithms import PPO_ParkourStudent
from rsl_rl.modules import ActorCriticParkourStudent
from rsl_rl.runners.ts_runner import TSRunner


class ParkourStudentRunner(TSRunner):
    """Runner for parkour student training with optional teacher weight initialization."""

    def _init_agent_and_algo(self):
        actor_critic = ActorCriticParkourStudent(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_privileged_obs,
            self.env.num_history_obs,
            self.env.num_latent_dims,
            self.env.num_critic_obs,
            **self.policy_cfg,
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        teacher_ckpt = self.policy_cfg.get("teacher_checkpoint")
        if teacher_ckpt:
            print(f"Loading teacher weights from: {teacher_ckpt}")
            state = torch.load(teacher_ckpt, map_location=self.device)
            self.alg.actor_critic.load_teacher_weights(state["model_state_dict"])
