"""Evaluate the fine-tuned teacher on all terrain types and rows."""
import torch
import os
os.environ["SIMULATOR"] = "genesis"

from legged_gym.envs import task_registry
from legged_gym.utils.helpers import get_args
from rsl_rl.modules import ActorCriticRMA

args = get_args()
args.headless = True
args.num_envs = 64
args.task = "go2_parkour_teacher"

env_cfg, train_cfg = task_registry.get_cfgs(args.task)
env_cfg.env.num_envs = 64
env_cfg.noise.add_noise = False
env_cfg.domain_rand.randomize_friction = False
env_cfg.domain_rand.randomize_base_mass = False
env_cfg.domain_rand.push_robots = False
env_cfg.domain_rand.randomize_com_displacement = False
env_cfg.domain_rand.randomize_pd_gain = False

env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)

CKPT = int(os.environ.get("EVAL_CKPT", "450"))
ckpt_path = f"logs/go2_parkour_teacher/finetune_genesis/model_{CKPT}.pt"
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=env.device, weights_only=False)

policy_cfg = train_cfg.policy
model = ActorCriticRMA(
    num_actor_obs=env_cfg.env.num_observations,
    num_critic_obs=env_cfg.env.num_privileged_obs,
    num_actions=env_cfg.env.num_actions,
    num_prop=policy_cfg.num_prop,
    num_scan=policy_cfg.num_scan,
    num_priv_explicit=policy_cfg.num_priv_explicit,
    num_priv_latent=policy_cfg.num_priv_latent,
    num_hist=policy_cfg.num_hist,
    scan_encoder_dims=policy_cfg.scan_encoder_dims,
    priv_encoder_dims=policy_cfg.priv_encoder_dims,
    actor_hidden_dims=policy_cfg.actor_hidden_dims,
    critic_hidden_dims=policy_cfg.critic_hidden_dims,
    history_encoder_channel_size=policy_cfg.history_encoder_channel_size,
).to(env.device)

model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

N = 64
CLIP = env_cfg.normalization.clip_actions

def run_eval(label, steps=1000, terrain_family=None, terrain_row=None):
    if terrain_family is not None:
        env.cfg.terrain.parkour.force_family = terrain_family
    if terrain_row is not None:
        env.cfg.terrain.parkour.force_row = terrain_row
    env.reset()
    obs = env.obs_buf
    total_dones = 0
    max_progress = torch.zeros(N, device=env.device)
    with torch.no_grad():
        for step in range(steps):
            actions = model.act_inference(obs, hist_encoding=True)
            actions = torch.clamp(actions, -CLIP, CLIP)
            obs, _, rews, dones, infos = env.step(actions)
            max_progress = torch.max(max_progress, env.progress_ratio)
            total_dones += dones.sum().item()
    print(f"[{label:35s}] resets={total_dones:4d}, mean_progress={max_progress.mean():.4f}, "
          f"max_progress={max_progress.max():.4f}")

print(f"Evaluating ckpt={CKPT}, clip={CLIP}")
print("=" * 90)
for family in ["stairs", "hurdle_block", "gap"]:
    for row in range(4):
        run_eval(f"{family}_row{row}", terrain_family=family, terrain_row=row)
    print()
