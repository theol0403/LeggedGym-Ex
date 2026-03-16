import torch

from legged_gym.envs.base.legged_robot_parkour import LeggedRobotParkour
from legged_gym.utils.math_utils import torch_rand_float


class Go2ParkourTeacher(LeggedRobotParkour):
    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_vel = torch.zeros(
            (len(env_ids), self.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + torch_rand_float(
            -0.2, 0.2, (len(env_ids), 4), self.device
        )
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + torch_rand_float(
            -0.4, 0.4, (len(env_ids), 4), self.device
        )
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + torch_rand_float(
            -0.4, 0.4, (len(env_ids), 4), self.device
        )
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
