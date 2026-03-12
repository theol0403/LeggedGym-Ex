from legged_gym.envs.base.legged_robot_ts import *

class LeggedRobotCTS(LeggedRobotTS):
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        num_envs = self.cfg.env.num_envs
        if num_envs < 2:
            raise ValueError("Concurrent teacher-student training requires at least 2 environments.")

        requested_num_teacher = getattr(self.cfg.env, "num_teacher", None)
        default_num_teacher = num_envs // 4 * 3
        if requested_num_teacher is None or requested_num_teacher >= num_envs:
            self.num_teacher = max(1, min(default_num_teacher, num_envs - 1))
        else:
            self.num_teacher = requested_num_teacher
        self.cfg.env.num_teacher = self.num_teacher
