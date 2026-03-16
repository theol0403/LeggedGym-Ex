import os
from dataclasses import dataclass

import numpy as np
import pygame


def resolve_command_mode(args) -> str:
    if args.command_mode == "auto" and args.use_joystick:
        return "joystick"
    return args.command_mode


def supports_manual_velocity_commands(env_cfg) -> bool:
    ranges = getattr(getattr(env_cfg, "commands", None), "ranges", None)
    if ranges is None:
        return False
    return all(
        hasattr(ranges, range_name)
        for range_name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
    )


@dataclass(frozen=True)
class VelocityCommandLimits:
    x: float
    y: float
    yaw: float


class PlayCommandController:
    def __init__(self, args, env_cfg):
        self.mode = resolve_command_mode(args)
        self.args = args
        self.env_cfg = env_cfg
        self.requires_external_command_source = self.mode != "auto"
        self.command = np.zeros(3, dtype=np.float32)
        self._window = None
        self._font = None
        self._joystick = None
        self._joystick_axis = None

        if not self.requires_external_command_source:
            self._limits = None
            return

        if not supports_manual_velocity_commands(env_cfg):
            raise NotImplementedError(
                f"Manual command mode '{self.mode}' is only implemented for velocity-command tasks."
            )
        if self.mode == "keyboard" and args.headless:
            raise ValueError("Keyboard command mode requires a visible desktop session.")
        if not (0.0 < args.command_scale <= 1.0):
            raise ValueError("--command_scale must be in the interval (0, 1].")

        self._limits = self._get_command_limits(env_cfg, args.command_scale)
        if self.mode == "keyboard":
            self._init_keyboard_window()
            print(
                "Keyboard teleop active: W/S forward-back, A/D left-right, Q/E yaw, "
                "Shift for fine control, Space to zero, Esc to quit."
            )
        elif self.mode == "joystick":
            self._init_joystick(args.joystick_type)
            print(
                "Joystick teleop active: left stick drives linear velocity, right stick X drives yaw."
            )
        else:
            raise ValueError(f"Unsupported command mode: {self.mode}")

    def _get_command_limits(self, env_cfg, command_scale: float) -> VelocityCommandLimits:
        ranges = env_cfg.commands.ranges
        planar_limit = command_scale * max(
            abs(ranges.lin_vel_x[0]),
            abs(ranges.lin_vel_x[1]),
            abs(ranges.lin_vel_y[0]),
            abs(ranges.lin_vel_y[1]),
        )
        return VelocityCommandLimits(
            x=planar_limit,
            y=planar_limit,
            yaw=command_scale * max(abs(ranges.ang_vel_yaw[0]), abs(ranges.ang_vel_yaw[1])),
        )

    def _init_keyboard_window(self):
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        pygame.init()
        pygame.display.set_caption("Legged Gym Keyboard Teleop")
        self._window = pygame.display.set_mode((560, 180))
        self._font = pygame.font.SysFont("monospace", 20)

    def _init_joystick(self, joystick_type: str):
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick connected.")
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        if joystick_type == "xbox":
            self._joystick_axis = {"lx": 0, "ly": 1, "rx": 3}
        elif joystick_type == "switch":
            self._joystick_axis = {"lx": 0, "ly": 1, "rx": 2}
        else:
            raise ValueError(f"Unsupported joystick type: {joystick_type}")
        print(f"Joystick name: {self._joystick.get_name()}")

    def _apply_command(self, env):
        command = np.asarray(self.command, dtype=np.float32)
        env.commands[:, :3] = env.commands.new_tensor(command).unsqueeze(0)

    def initialize_env_commands(self, env):
        if self.requires_external_command_source:
            self._apply_command(env)

    def update(self, env) -> bool:
        if self.mode == "auto":
            return True
        if self.mode == "keyboard":
            return self._update_keyboard(env)
        if self.mode == "joystick":
            return self._update_joystick(env)
        raise ValueError(f"Unsupported command mode: {self.mode}")

    def _update_keyboard(self, env) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self.command[:] = 0.0
        else:
            fine_scale = 0.35 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0
            self.command[0] = fine_scale * self._limits.x * (
                float(keys[pygame.K_w] or keys[pygame.K_UP]) -
                float(keys[pygame.K_s] or keys[pygame.K_DOWN])
            )
            self.command[1] = fine_scale * self._limits.y * (
                float(keys[pygame.K_a] or keys[pygame.K_LEFT]) -
                float(keys[pygame.K_d] or keys[pygame.K_RIGHT])
            )
            self.command[2] = fine_scale * self._limits.yaw * (
                float(keys[pygame.K_q]) -
                float(keys[pygame.K_e])
            )

        self._apply_command(env)
        self._draw_keyboard_overlay()
        return True

    def _update_joystick(self, env) -> bool:
        pygame.event.pump()
        lx = self._joystick.get_axis(self._joystick_axis["lx"])
        ly = self._joystick.get_axis(self._joystick_axis["ly"])
        rx = self._joystick.get_axis(self._joystick_axis["rx"])
        self.command[0] = self._apply_deadzone(-ly) * self._limits.x
        self.command[1] = self._apply_deadzone(-lx) * self._limits.y
        self.command[2] = self._apply_deadzone(-rx) * self._limits.yaw
        self._apply_command(env)
        return True

    def _draw_keyboard_overlay(self):
        self._window.fill((20, 20, 24))
        lines = (
            "W/S or Up/Down: forward/back",
            "A/D or Left/Right: strafe left/right",
            "Q/E: yaw left/right",
            "Shift: fine control    Space: zero    Esc: quit",
            (
                f"cmd = [{self.command[0]: .2f}, {self.command[1]: .2f}, "
                f"{self.command[2]: .2f}]"
            ),
        )
        for idx, line in enumerate(lines):
            surface = self._font.render(line, True, (230, 230, 230))
            self._window.blit(surface, (18, 18 + 30 * idx))
        pygame.display.flip()

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float = 0.12) -> float:
        if abs(value) < deadzone:
            return 0.0
        return float(value)

    def close(self):
        if self.mode in {"keyboard", "joystick"}:
            pygame.quit()
