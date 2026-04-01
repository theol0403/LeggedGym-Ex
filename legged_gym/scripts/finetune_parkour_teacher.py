"""Fine-tune the pre-trained CAI23sbP teacher in Genesis.

The pre-trained model was trained in PhysX/IsaacLab. This script adapts it
to Genesis physics while preserving parkour skills via short fine-tuning
with conservative learning rate.

Usage:
    SIMULATOR=genesis python -m legged_gym.scripts.finetune_parkour_teacher --headless \
        --num_envs 4096 --load_run external_cai23sbp --ckpt 49999
"""
import os
import torch
import pathlib

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, ensure_runtime_initialized
from legged_gym.utils.helpers import get_load_path, class_to_dict
from rsl_rl.runners import OnPolicyRunner


def finetune(args):
    ensure_runtime_initialized(args)
    args.task = "go2_parkour_teacher"

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    # Fine-tuning overrides — conservative to preserve parkour skills
    FINETUNE_LR = 1e-4
    FINETUNE_ITERATIONS = 1000
    FINETUNE_SAVE_INTERVAL = 50

    train_cfg.algorithm.learning_rate = FINETUNE_LR
    train_cfg.runner.max_iterations = FINETUNE_ITERATIONS
    train_cfg.runner.save_interval = FINETUNE_SAVE_INTERVAL
    train_cfg.runner.run_name = "finetune_genesis"

    # Create environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Create runner (with fresh optimizer at FINETUNE_LR)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, 'finetune_genesis')
    os.makedirs(log_dir, exist_ok=True)

    train_cfg_dict = class_to_dict(train_cfg)
    device = "cpu" if args.cpu else "cuda"
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=device)

    # Load pre-trained weights (WITHOUT optimizer state)
    load_run = getattr(args, 'load_run', None) or 'external_cai23sbp'
    ckpt = getattr(args, 'ckpt', None)
    if ckpt is None or ckpt < 0:
        ckpt = 49999

    ckpt_path = get_load_path(
        pathlib.Path(log_root),
        load_run=load_run,
        checkpoint=ckpt,
    )
    print(f"Loading pre-trained weights from: {ckpt_path}")
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded['model_state_dict'])
    # Do NOT load optimizer state — use fresh optimizer at FINETUNE_LR
    # Reset iteration counter to 0 for clean logging
    runner.current_learning_iteration = 0

    print(f"Fine-tuning with LR={FINETUNE_LR}, iterations={FINETUNE_ITERATIONS}")
    print(f"Action clip: {env_cfg.normalization.clip_actions}")
    print(f"Logging to: {log_dir}")

    try:
        runner.learn(num_learning_iterations=FINETUNE_ITERATIONS, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == '__main__':
    args = get_args()
    finetune(args)
