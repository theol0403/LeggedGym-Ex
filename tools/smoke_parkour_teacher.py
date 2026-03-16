#!/usr/bin/env python3
from types import SimpleNamespace

import torch

from legged_gym.envs import task_registry
from legged_gym.utils.math_utils import quat_rotate_inverse, wrap_to_pi
from legged_gym.utils import class_to_dict, ensure_runtime_initialized


def main():
    args = SimpleNamespace(cpu=True, headless=True)
    ensure_runtime_initialized(args)

    env_cfg, train_cfg = task_registry.get_cfgs("go2_parkour_teacher")
    env_cfg.env.num_envs = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.parkour.include_flat_debug = True
    env_cfg.terrain.parkour.force_row = 0
    env_cfg.terrain.parkour.force_family = None
    train_cfg.runner.max_iterations = 1

    expected_obs = env_cfg.env.num_observations
    expected_priv = env_cfg.env.num_privileged_obs
    task_class = task_registry.get_task_class("go2_parkour_teacher")
    env = task_class(
        cfg=env_cfg,
        sim_params=class_to_dict(env_cfg.sim),
        sim_device="cpu",
        headless=True,
    )

    obs, privileged_obs = env.reset()
    if obs.shape != (env_cfg.env.num_envs, expected_obs):
        raise AssertionError(f"Actor observation shape mismatch: {tuple(obs.shape)} != {(env_cfg.env.num_envs, expected_obs)}")
    if privileged_obs.shape != (env_cfg.env.num_envs, expected_priv):
        raise AssertionError(
            f"Critic observation shape mismatch: {tuple(privileged_obs.shape)} != {(env_cfg.env.num_envs, expected_priv)}"
        )
    env_ids = torch.arange(env_cfg.env.num_envs, device=env.device, dtype=torch.long)
    goal_local = env.simulator.lane_waypoints[env_ids, env.active_waypoint_idx]
    tile_origin_xy = env.simulator.env_origins[:, :2] - env.tile_half_extent
    goal_world = goal_local.clone()
    goal_world[:, 0] += tile_origin_xy[:, 0]
    goal_world[:, 1] += tile_origin_xy[:, 1]
    goal_delta_world = goal_world - env.simulator.base_pos
    goal_delta_base = quat_rotate_inverse(env.simulator.base_quat, goal_delta_world)
    goal_heading_world = torch.atan2(goal_delta_world[:, 1], goal_delta_world[:, 0])
    heading_error = wrap_to_pi(goal_heading_world - env.simulator.base_euler[:, 2])
    expected_commands = torch.stack(
        [goal_delta_base[:, 0], goal_delta_base[:, 1], heading_error, env.goal_speed_targets],
        dim=-1,
    )
    if not torch.allclose(env.commands, expected_commands, atol=1e-5, rtol=1e-5):
        raise AssertionError("Reset command targets do not match the current spawn pose and active waypoint.")
    if not torch.allclose(obs[:, :4], expected_commands * env.commands_scale, atol=1e-5, rtol=1e-5):
        raise AssertionError("First post-reset actor observation does not expose the active goal command correctly.")

    zero_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions), dtype=torch.float, device=env.device)
    obs, privileged_obs, rewards, dones, infos = env.step(zero_actions)
    if obs.shape[1] != expected_obs or privileged_obs.shape[1] != expected_priv:
        raise AssertionError("Observation shapes changed after stepping the environment.")

    families = env.simulator.lane_family.cpu().tolist()
    rows = env.simulator.lane_difficulty_row.cpu().tolist()
    if families != [0, 1, 2, 3]:
        raise AssertionError(f"Expected deterministic terrain family ordering [0, 1, 2, 3], got {families}")
    if rows != [0, 0, 0, 0]:
        raise AssertionError(f"Expected deterministic difficulty rows [0, 0, 0, 0], got {rows}")
    if env.simulator.lane_waypoints.shape != (4, env_cfg.terrain.parkour.max_waypoints, 3):
        raise AssertionError(f"Unexpected waypoint tensor shape: {tuple(env.simulator.lane_waypoints.shape)}")
    if env.simulator.lane_edge_masks.shape[0] != env_cfg.env.num_envs:
        raise AssertionError("Missing per-environment lane edge masks.")
    waypoint_counts = env.simulator.lane_waypoint_counts.cpu().tolist()
    if waypoint_counts[:3] != [4, 5, 5]:
        raise AssertionError(f"Expected waypoint counts [4, 5, 5] for stairs/block/gap at row 0, got {waypoint_counts[:3]}")

    for _ in range(env_cfg.terrain.parkour.max_waypoints + 1):
        env._update_local_base_positions()
        env._update_current_section_state()
        env._update_command_targets()
        env.simulator._base_pos[:] = env._lane_local_to_world(env._get_active_goal_local())
        for _ in range(env_cfg.terrain.parkour.waypoint_dwell_steps):
            env._update_local_base_positions()
            env._update_current_section_state()
            env._update_command_targets()
            advance_ids = env._update_parkour_progress()
            if len(advance_ids) > 0:
                env._update_command_targets(advance_ids)
        if torch.all(env.course_success):
            break
    if not torch.all(env.course_success):
        raise AssertionError("Scripted goal teleports did not complete the full multi-waypoint course.")

    print(f"registry_ok task=go2_parkour_teacher actor_obs={expected_obs} critic_obs={expected_priv}")
    print(f"step_ok rewards_shape={tuple(rewards.shape)} dones_shape={tuple(dones.shape)}")
    print(
        "metadata_ok "
        f"families={families} rows={rows} "
        f"waypoint_counts={waypoint_counts} "
        f"waypoints_shape={tuple(env.simulator.lane_waypoints.shape)}"
    )
    print(
        "progress_ok "
        f"active_waypoint_idx={env.active_waypoint_idx.cpu().tolist()} "
        f"waypoints_reached={env.waypoints_reached.cpu().tolist()} "
        f"success={env.course_success.cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
