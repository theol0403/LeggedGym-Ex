import os
import inspect

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, ensure_runtime_initialized
import shutil


def _backup_task_sources(log_dir, env, env_cfg):
    backup_paths = []
    for source in (inspect.getfile(env.__class__), inspect.getfile(env_cfg.__class__)):
        if os.path.isfile(source) and source not in backup_paths:
            backup_paths.append(source)
    for source in backup_paths:
        shutil.copy2(source, log_dir)

def train(args):
    ensure_runtime_initialized(args)
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Copy task sources to log_dir for backup
    log_dir = ppo_runner.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    _backup_task_sources(log_dir, env, env_cfg)
    
    # Start training session
    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.num_envs = 1
    train(args)
