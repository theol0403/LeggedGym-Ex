from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import terrain_utils


PARKOUR_FAMILY_IDS = {
    "stairs": 0,
    "hurdle_block": 1,
    "gap": 2,
}

PARKOUR_SECTION_IDS = {"jump": 1, "stairs": 2}


@dataclass(frozen=True)
class ParkourLaneMetadata:
    family: int
    difficulty_row: int
    spawn_pose: np.ndarray
    safe_spawn_region: np.ndarray
    lane_bounds: np.ndarray
    waypoints: np.ndarray
    waypoint_count: int
    terminal_goal: np.ndarray
    section_bounds: np.ndarray
    section_tags: np.ndarray
    jump_expected_mask: np.ndarray
    edge_mask: np.ndarray


class ParkourLaneBuilder:
    def __init__(self, terrain_cfg):
        self.cfg = terrain_cfg
        self.env_length = float(terrain_cfg.terrain_length)
        self.env_width = float(terrain_cfg.terrain_width)
        self.length_per_env_pixels = int(self.env_length / terrain_cfg.horizontal_scale)
        self.width_per_env_pixels = int(self.env_width / terrain_cfg.horizontal_scale)
        self.max_waypoints = int(terrain_cfg.parkour.max_waypoints)
        self.max_obstacles = int(terrain_cfg.parkour.max_obstacles)
        self.max_sections = int(terrain_cfg.parkour.max_sections)
        if self.max_obstacles < 1:
            raise ValueError("ParkourLaneBuilder requires terrain.parkour.max_obstacles >= 1.")
        if self.max_sections < self.max_obstacles:
            raise ValueError(
                "Parkour terrain config is inconsistent: max_sections must be >= max_obstacles "
                f"(got {self.max_sections} < {self.max_obstacles})."
            )
        if self.max_waypoints < self.max_obstacles + 1:
            raise ValueError(
                "Parkour terrain config is inconsistent: max_waypoints must be >= max_obstacles + 1 "
                f"(got {self.max_waypoints} < {self.max_obstacles + 1})."
            )
        self._base_slot_centers = np.array([4.0, 8.0, 12.0], dtype=np.float32)
        self._slot_offset_patterns = np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.35, 0.15, 0.40],
                [0.25, -0.25, 0.10],
                [0.40, 0.25, -0.30],
            ],
            dtype=np.float32,
        )

    def build_lane(self, family: str, difficulty_row: int, variant_id: int = 0):
        if difficulty_row < 0:
            raise ValueError(f"Parkour difficulty_row must be non-negative, got {difficulty_row}.")
        terrain = terrain_utils.SubTerrain(
            "parkour",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        edge_mask = np.zeros_like(terrain.height_field_raw, dtype=np.uint8)

        lane_center_y = 0.5 * self.env_width
        lane_half_width = 1.1
        spawn_x = 0.75
        goal_x = self.env_length - 0.6
        y_min = lane_center_y - lane_half_width
        y_max = lane_center_y + lane_half_width

        section_bounds = np.zeros((self.max_sections, 2), dtype=np.float32)
        section_tags = np.zeros((self.max_sections,), dtype=np.int32)
        section_jump_expected = np.zeros((self.max_sections,), dtype=np.bool_)
        waypoints = np.zeros((self.max_waypoints, 3), dtype=np.float32)

        waypoint_count = 0
        section_count = 0
        row_idx = min(difficulty_row, 3)
        obstacle_count_by_row = [1, 2, 3, 3]
        obstacle_count = min(self.max_obstacles, obstacle_count_by_row[row_idx])
        self._validate_lane_capacity(obstacle_count)
        slot_centers = self._slot_centers_for_variant(variant_id)

        if family == "stairs":
            step_heights = [0.06, 0.08, 0.10, 0.12]
            step_counts = [2, 3, 4, 4]
            post_offsets = [0.90, 0.80, 0.68, 0.58]
            step_height = step_heights[row_idx]
            step_count = step_counts[row_idx]
            for slot_idx in range(obstacle_count):
                slot_center_x = float(slot_centers[slot_idx])
                end_x, _ = self._build_stairs_feature(
                    terrain=terrain,
                    center_x=slot_center_x,
                    y_min=y_min,
                    y_max=y_max,
                    step_height=step_height,
                    step_count=step_count,
                )
                section_start_x = self._stairs_start_x(slot_center_x, step_count)
                section_bounds[section_count] = np.array(
                    [section_start_x - 0.1, end_x + 0.3], dtype=np.float32
                )
                section_tags[section_count] = PARKOUR_SECTION_IDS["stairs"]
                section_jump_expected[section_count] = False
                waypoint_x = min(end_x + post_offsets[row_idx], goal_x)
                waypoints[waypoint_count] = np.array(
                    [waypoint_x, lane_center_y, self._sample_local_height(terrain, waypoint_x, lane_center_y)],
                    dtype=np.float32,
                )
                waypoint_count += 1
                section_count += 1
        elif family == "hurdle_block":
            block_heights = [0.10, 0.14, 0.20, 0.26]
            block_lengths = [0.20, 0.22, 0.25, 0.28]
            y_halves = [0.55, 0.65, 0.75, 0.85]
            post_offsets = [0.95, 0.80, 0.68, 0.55]
            block_height = block_heights[row_idx]
            block_length = block_lengths[row_idx]
            y_half = y_halves[row_idx]
            for slot_idx in range(obstacle_count):
                slot_center_x = float(slot_centers[slot_idx])
                x0 = slot_center_x - 0.5 * block_length
                x1 = x0 + block_length
                block_y_min = lane_center_y - y_half
                block_y_max = lane_center_y + y_half
                self._fill_rect_height(terrain, x0, x1, block_y_min, block_y_max, block_height)
                self._mark_rect_perimeter(edge_mask, x0, x1, block_y_min, block_y_max)
                section_bounds[section_count] = np.array([x0 - 0.2, x1 + 0.35], dtype=np.float32)
                section_tags[section_count] = PARKOUR_SECTION_IDS["jump"]
                section_jump_expected[section_count] = True
                waypoint_x = min(x1 + post_offsets[row_idx], goal_x)
                waypoints[waypoint_count] = np.array(
                    [waypoint_x, lane_center_y, self._sample_local_height(terrain, waypoint_x, lane_center_y)],
                    dtype=np.float32,
                )
                waypoint_count += 1
                section_count += 1
        elif family == "gap":
            gap_widths = [0.16, 0.24, 0.36, 0.48]
            gap_y_halves = [0.85, 0.95, 1.05, 1.15]
            post_offsets = [1.00, 0.85, 0.74, 0.62]
            gap_width = gap_widths[row_idx]
            gap_y_half = gap_y_halves[row_idx]
            for slot_idx in range(obstacle_count):
                slot_center_x = float(slot_centers[slot_idx])
                x0 = slot_center_x - 0.5 * gap_width
                x1 = x0 + gap_width
                gap_y_min = lane_center_y - gap_y_half
                gap_y_max = lane_center_y + gap_y_half
                self._fill_rect_height(terrain, x0, x1, gap_y_min, gap_y_max, -5.0)
                self._mark_gap_edges(edge_mask, x0, x1, gap_y_min, gap_y_max)
                section_bounds[section_count] = np.array([x0 - 0.2, x1 + 0.45], dtype=np.float32)
                section_tags[section_count] = PARKOUR_SECTION_IDS["jump"]
                section_jump_expected[section_count] = True
                waypoint_x = min(x1 + post_offsets[row_idx], goal_x)
                waypoints[waypoint_count] = np.array(
                    [waypoint_x, lane_center_y, self._sample_local_height(terrain, waypoint_x, lane_center_y)],
                    dtype=np.float32,
                )
                waypoint_count += 1
                section_count += 1
        else:
            raise ValueError(f"Unsupported parkour family '{family}'")

        terminal_goal = np.array(
            [goal_x, lane_center_y, self._sample_local_height(terrain, goal_x, lane_center_y)],
            dtype=np.float32,
        )
        terminal_index = min(waypoint_count, self.max_waypoints - 1)
        waypoints[terminal_index] = terminal_goal
        waypoint_count = terminal_index + 1

        metadata = ParkourLaneMetadata(
            family=PARKOUR_FAMILY_IDS[family],
            difficulty_row=difficulty_row,
            spawn_pose=np.array(
                [spawn_x, lane_center_y, self._sample_local_height(terrain, spawn_x, lane_center_y), 0.0],
                dtype=np.float32,
            ),
            safe_spawn_region=np.array([0.6, 1.2, lane_center_y - 0.3, lane_center_y + 0.3], dtype=np.float32),
            lane_bounds=np.array([0.4, self.env_length - 0.4, y_min, y_max], dtype=np.float32),
            waypoints=waypoints,
            waypoint_count=waypoint_count,
            terminal_goal=terminal_goal,
            section_bounds=section_bounds,
            section_tags=section_tags,
            jump_expected_mask=section_jump_expected,
            edge_mask=edge_mask,
        )
        return terrain, metadata

    def _validate_lane_capacity(self, obstacle_count: int):
        required_sections = obstacle_count
        required_waypoints = obstacle_count + 1
        if required_sections > self.max_sections:
            raise ValueError(
                "Parkour lane exceeds configured section capacity: "
                f"required_sections={required_sections}, max_sections={self.max_sections}."
            )
        if required_waypoints > self.max_waypoints:
            raise ValueError(
                "Parkour lane exceeds configured waypoint capacity: "
                f"required_waypoints={required_waypoints}, max_waypoints={self.max_waypoints}."
            )

    def _build_stairs_feature(self, terrain, center_x, y_min, y_max, step_height, step_count):
        row_fraction = max(step_count - 2, 0) / 2.0
        step_tread = self._lerp(0.28, 0.38, row_fraction)
        flat_top = self._lerp(0.18, 0.34, row_fraction)

        start_x = self._stairs_start_x(center_x, step_count)
        current_height = 0.0
        cursor_x = start_x
        for _ in range(step_count):
            current_height += step_height
            self._fill_rect_height(terrain, cursor_x, cursor_x + step_tread, y_min, y_max, current_height)
            cursor_x += step_tread

        self._fill_rect_height(terrain, cursor_x, cursor_x + flat_top, y_min, y_max, current_height)
        cursor_x += flat_top

        for step_idx in range(step_count):
            descending_height = current_height - step_height * (step_idx + 1)
            self._fill_rect_height(
                terrain,
                cursor_x,
                cursor_x + step_tread,
                y_min,
                y_max,
                max(descending_height, 0.0),
            )
            cursor_x += step_tread

        return cursor_x, current_height

    def _slot_centers_for_variant(self, variant_id: int):
        offset_row = self._slot_offset_patterns[variant_id % len(self._slot_offset_patterns)]
        slot_centers = self._base_slot_centers + offset_row
        return np.clip(slot_centers, 3.3, self.env_length - 3.3)

    def _stairs_start_x(self, center_x: float, step_count: int):
        row_fraction = max(step_count - 2, 0) / 2.0
        step_tread = self._lerp(0.28, 0.38, row_fraction)
        flat_top = self._lerp(0.18, 0.34, row_fraction)
        total_length = (2 * step_count * step_tread) + flat_top
        return center_x - 0.5 * total_length

    def _meters_to_cell(self, value: float, upper_bound: int) -> int:
        return int(np.clip(np.round(value / self.cfg.horizontal_scale), 0, upper_bound - 1))

    def _fill_rect_height(self, terrain, x_min, x_max, y_min, y_max, height_m):
        px0 = self._meters_to_cell(x_min, terrain.width)
        px1 = max(px0 + 1, self._meters_to_cell(x_max, terrain.width))
        py0 = self._meters_to_cell(y_min, terrain.length)
        py1 = max(py0 + 1, self._meters_to_cell(y_max, terrain.length))
        height_cells = int(np.round(height_m / self.cfg.vertical_scale))
        terrain.height_field_raw[px0:px1, py0:py1] = height_cells

    def _mark_rect_perimeter(self, edge_mask, x_min, x_max, y_min, y_max):
        px0 = self._meters_to_cell(x_min, edge_mask.shape[0])
        px1 = max(px0 + 1, self._meters_to_cell(x_max, edge_mask.shape[0]))
        py0 = self._meters_to_cell(y_min, edge_mask.shape[1])
        py1 = max(py0 + 1, self._meters_to_cell(y_max, edge_mask.shape[1]))
        edge_mask[px0:px1, py0] = 1
        edge_mask[px0:px1, py1 - 1] = 1
        edge_mask[px0, py0:py1] = 1
        edge_mask[px1 - 1, py0:py1] = 1

    def _mark_gap_edges(self, edge_mask, x_min, x_max, y_min, y_max):
        px0 = self._meters_to_cell(x_min, edge_mask.shape[0])
        px1 = max(px0 + 1, self._meters_to_cell(x_max, edge_mask.shape[0]))
        py0 = self._meters_to_cell(y_min, edge_mask.shape[1])
        py1 = max(py0 + 1, self._meters_to_cell(y_max, edge_mask.shape[1]))
        edge_mask[px0:px0 + 1, py0:py1] = 1
        edge_mask[max(px1 - 1, px0):px1, py0:py1] = 1

    def _sample_local_height(self, terrain, x, y):
        px = self._meters_to_cell(x, terrain.width)
        py = self._meters_to_cell(y, terrain.length)
        return float(terrain.height_field_raw[px, py] * self.cfg.vertical_scale)

    @staticmethod
    def _lerp(start: float, end: float, alpha: float) -> float:
        return float(start + (end - start) * alpha)
