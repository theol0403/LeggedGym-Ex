import os
import inspect
from pathlib import Path

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, ensure_runtime_initialized
import shutil


def _iter_local_class_sources(cls):
    for base_cls in inspect.getmro(cls):
        if base_cls is object:
            continue
        module = inspect.getmodule(base_cls)
        if module is None:
            continue
        try:
            src_path = inspect.getsourcefile(module) or inspect.getfile(module)
        except (OSError, TypeError):
            continue
        if src_path is None:
            continue
        src_path = os.path.realpath(src_path)
        if src_path.startswith(os.path.realpath(LEGGED_GYM_ROOT_DIR) + os.sep):
            yield src_path


def _copy_task_sources(log_dir, env, env_cfg):
    source_paths = {
        os.path.realpath(__file__),
    }
    source_paths.update(_iter_local_class_sources(env.__class__))
    source_paths.update(_iter_local_class_sources(type(env_cfg)))
    snapshot_root = Path(log_dir) / "source"
    for src_path in sorted(source_paths):
        relative_path = os.path.relpath(src_path, LEGGED_GYM_ROOT_DIR)
        dst_path = snapshot_root / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def train(args):
    ensure_runtime_initialized(args)
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Copy env.py and env_config.py to log_dir for backup
    log_dir = ppo_runner.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    _copy_task_sources(log_dir, env, env_cfg)
    
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
