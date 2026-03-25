import os
import copy
import torch
import numpy as np
import random
import argparse
import re

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if isinstance(obj, (str, int, float, bool, type(None), np.ndarray, torch.Tensor)):
        return obj
    if isinstance(obj, tuple):
        return tuple(class_to_dict(item) for item in obj)
    if isinstance(obj, dict):
        return {key: class_to_dict(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [class_to_dict(item) for item in obj]
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if callable(val):
            continue
        result[key] = class_to_dict(val)
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _has_checkpoints(path):
    return any(re.match(r"^model_\d+\.pt$", f) for f in os.listdir(path))

def _get_load_run_dir(root, load_run):
    runs = sorted(
        run for run in os.listdir(root)
        if os.path.isdir(os.path.join(root, run)) and _has_checkpoints(os.path.join(root, run))
    )
    if not runs:
        raise ValueError("No runs with checkpoints in: " + root)
    if load_run == -1:
        return os.path.join(root, runs[-1])
    run_dir = os.path.join(root, load_run)
    if not os.path.isdir(run_dir):
        raise ValueError("Run directory does not exist: " + run_dir)
    return run_dir

def _get_latest_checkpoint(load_run, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.pt$")
    candidates = []
    for file_name in os.listdir(load_run):
        match = pattern.match(file_name)
        if match:
            candidates.append((int(match.group(1)), file_name))
    if not candidates:
        raise ValueError(f"No {prefix} checkpoints found in: {load_run}")
    return max(candidates)[1]

def get_load_path(root, load_run=-1, checkpoint=-1):
    load_run = _get_load_run_dir(root, load_run)
    model = _get_latest_checkpoint(load_run, "model") if checkpoint == -1 else f"model_{checkpoint}.pt"
    load_path = os.path.join(load_run, model)
    if not os.path.isfile(load_path):
        raise ValueError("Checkpoint does not exist: " + load_path)
    return load_path

def get_load_path_ee(root, load_run=-1, checkpoint=-1):
    load_run = _get_load_run_dir(root, load_run)
    if checkpoint == -1:
        model = _get_latest_checkpoint(load_run, "model")
        estimator = _get_latest_checkpoint(load_run, "estimator")
    else:
        model = f"model_{checkpoint}.pt"
        estimator = f"estimator_{checkpoint}.pt"

    actor_load_path = os.path.join(load_run, model)
    estimator_load_path = os.path.join(load_run, estimator)
    if not os.path.isfile(actor_load_path):
        raise ValueError("Checkpoint does not exist: " + actor_load_path)
    if not os.path.isfile(estimator_load_path):
        raise ValueError("Checkpoint does not exist: " + estimator_load_path)
    return actor_load_path, estimator_load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    """Override some configuration parameters from command line arguments
       Called in task_registry.py/make_env()

    Args:
        env_cfg : environment configuration
        cfg_train : training configuration
        args : command line arguments

    Returns:
        env_cfg : updated environment configuration
        cfg_train : updated training configuration
    """
    # environment parameters
    if env_cfg is not None:
        # num envs
        if getattr(args, "num_envs", None) is not None:
            env_cfg.env.num_envs = args.num_envs
        if hasattr(env_cfg, "viewer") and hasattr(args, "follow_robot"):
            env_cfg.viewer.follow_robot = bool(args.follow_robot)
        if getattr(args, "debug", False):
            env_cfg.env.debug = args.debug
        if getattr(args, "motion_file", None) is not None:
            env_cfg.env.motion_file = args.motion_file
        parkour_cfg = getattr(env_cfg.terrain, "parkour", None)
        if parkour_cfg is not None and getattr(parkour_cfg, "enable", False):
            if getattr(args, "parkour_force_family", None) is not None:
                parkour_cfg.force_family = args.parkour_force_family
            if getattr(args, "parkour_force_row", None) is not None:
                parkour_cfg.force_row = args.parkour_force_row
            
    # training parameters
    if cfg_train is not None:
        resume_requested = getattr(args, "resume", False) or getattr(args, "load_run", None) is not None or (
            getattr(args, "ckpt", None) is not None and args.ckpt >= 0
        )
        # alg runner parameters
        if getattr(args, "max_iterations", None) is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if resume_requested:
            cfg_train.runner.resume = True
        if getattr(args, "sync_wandb", False):
            cfg_train.runner.sync_wandb = args.sync_wandb
        if getattr(args, "ckpt", None) is not None:
            cfg_train.runner.checkpoint = args.ckpt
        if getattr(args, "load_run", None) is not None:
            cfg_train.runner.load_run = args.load_run
        if getattr(args, "teacher_task", None) is not None:
            cfg_train.runner.teacher_task = args.teacher_task
        if getattr(args, "teacher_load_run", None) is not None:
            cfg_train.runner.teacher_load_run = args.teacher_load_run
        if getattr(args, "teacher_ckpt", None) is not None:
            cfg_train.runner.teacher_ckpt = args.teacher_ckpt

    return env_cfg, cfg_train

def get_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='go2', help="task name")
    parser.add_argument('--headless',       action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',            action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--num_envs',       type=int, default=None, help="number of parallel environments")
    parser.add_argument('--max_iterations', type=int, default=None, help="max number of training iterations")
    parser.add_argument('--resume',         action='store_true', default=False, help="resume training from specified checkpoint")
    parser.add_argument('--sync_wandb',     action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx',    action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug',          action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--depth_debug',    action='store_true', default=False, help="enable depth camera debug rendering")
    parser.add_argument('--rgb_debug',      action='store_true', default=False, help="enable RGB camera debug rendering when supported")
    parser.add_argument('--depth_model_type', type=str, default=None,
                        choices=[
                            'depth_anything_v2',
                            'depth_anything_v2_metric_indoor',
                            'depth_anything_v2_metric_outdoor',
                            'video_depth_anything_metric',
                            'metric3d_v2',
                            'unidepth_v2',
                        ],
                        help="inferred-depth model family; passing this enables the inferred-depth debug panel")
    parser.add_argument('--depth_model_size', type=str, default=None,
                        choices=['small', 'base', 'large', 'giant'],
                        help="inferred-depth model size; defaults to the config value when omitted")
    parser.add_argument('--depth_update_interval', type=int, default=1,
                        help="run inferred-depth updates every N rendered debug frames")
    parser.add_argument('--load_run',       type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt',           type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--command_mode',   type=str, default='auto', choices=['auto', 'keyboard', 'joystick'],
                        help="play command source: auto uses the task command sampler, keyboard opens a teleop window, joystick reads a gamepad")
    parser.add_argument('--command_scale',  type=float, default=1.0,
                        help="fraction of the task command range used by keyboard/joystick teleop")
    parser.add_argument('--use_joystick',   action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type',  type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot',   action='store_true', default=False, help="whether the viewer camera follows the robot when visualization is enabled")
    parser.add_argument('--log_play_stats', action='store_true', default=False, help="collect and plot play-time state/reward logs")
    parser.add_argument('--motion_file',    type=str, 
                        default=None, 
                        help="motion file to load")
    parser.add_argument('--parkour_force_family', type=str, default=None,
                        choices=['stairs', 'hurdle_block', 'gap'],
                        help="force the Genesis parkour task to sample a single obstacle family")
    parser.add_argument('--parkour_force_row', type=int, default=None,
                        help="force the Genesis parkour task to sample a single curriculum row")
    parser.add_argument('--teacher_task',    type=str, default=None,
                        help="teacher task used to resolve the frozen parkour teacher checkpoint")
    parser.add_argument('--teacher_load_run', type=str, default=None,
                        help="teacher run to load, default: latest run for the teacher task")
    parser.add_argument('--teacher_ckpt',    type=int, default=None,
                        help="teacher checkpoint to load, default: latest checkpoint for the teacher run")

    return parser.parse_args()

class PolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
    
    def forward(self, obs):
        return self.actor(obs)
    
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)
        
        # export onnx model if needed
        if export_onnx:
            filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_observations)
            torch.onnx.export(self, dummy_input, path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterTS(torch.nn.Module):
    """Policy exporter for teacher student policies

    Attention: This module is consistent with ActorCriticTS in rsl_rl/modules/actor_critic_ts.py
               When ActorCriticTS is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.encoder = copy.deepcopy(actor_critic.history_encoder)
        self.history_encoder_type = actor_critic.history_encoder_type
    
    def forward(self, obs, history):
        if self.history_encoder_type == "TCN":
            history = history.unsqueeze(1)
        latent = self.encoder(history)
        x = torch.cat([obs, latent], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)
        
        # export onnx model if needed
        if export_onnx:
            filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)


class PolicyExporterParkourStudent(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = copy.deepcopy(actor_critic)

    def forward(self, obs, student_depth):
        return self.actor_critic.act_inference(obs, student_depth)

    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)

        if export_onnx:
            filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "student_depth_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_depth = torch.randn(1, *env_cfg.env.student_depth_shape)
            torch.onnx.export(
                self,
                (dummy_obs, dummy_depth),
                path_onnx,
                verbose=True,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                opset_version=11,
            )


class PolicyExporterEE(torch.nn.Module):
    """Policy exporter for explicit estimator policies

    Attention: This module is consistent with ActorCriticEE in rsl_rl/modules/actor_critic_ee.py
               When ActorCriticEE is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator)
    
    def forward(self, obs_history):
        estimated_state = self.estimator(obs_history)
        x = torch.cat([obs_history, estimated_state], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        pt_path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(pt_path)
        
        # export onnx model if needed
        if export_onnx:
            filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            onnx_path = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_estimator_features)
            torch.onnx.export(self, dummy_input, onnx_path, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterWaQ(torch.nn.Module):
    """Policy exporter for DreamWaQ policies
    
    Attention: This module is consistent with ActorCriticDreamWaQ in rsl_rl/modules/actor_critic_dreamwaq.py
               When ActorCriticDreamWaQ is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
    
    def forward(self, obs, obs_history):
        vae_out = self.vae.inference(obs_history)
        x = torch.cat([obs, vae_out], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)
        
        # export onnx model if needed
        if export_onnx:
            filename = str(train_cfg.runner.load_run) + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
