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
        self._slot_margin = 3.0
        self._base_slot_centers = np.linspace(
            self._slot_margin,
            self.env_length - self._slot_margin,
            num=self.max_obstacles,
            dtype=np.float32,
        )
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
        lane_half_width = 1.2
        spawn_x = 0.55
        goal_x = self.env_length - 0.45
        y_min = lane_center_y - lane_half_width
        y_max = lane_center_y + lane_half_width

        section_bounds = np.zeros((self.max_sections, 2), dtype=np.float32)
        section_tags = np.zeros((self.max_sections,), dtype=np.int32)
        section_jump_expected = np.zeros((self.max_sections,), dtype=np.bool_)
        waypoints = np.zeros((self.max_waypoints, 3), dtype=np.float32)

        waypoint_count = 0
        section_count = 0
        row_idx = min(difficulty_row, 3)
        obstacle_count_by_row = [2, 3, 3, 3]
        obstacle_count = min(self.max_obstacles, obstacle_count_by_row[row_idx])
        self._validate_lane_capacity(obstacle_count)
        slot_centers = self._slot_centers_for_variant(variant_id)

        for slot_idx in range(obstacle_count):
            slot_center_x = float(slot_centers[slot_idx])
            wp_x, sec_bounds, sec_tag, sec_jump = self._build_single_obstacle(
                terrain, edge_mask, family, row_idx, slot_center_x,
                lane_center_y, y_min, y_max, goal_x,
            )
            section_bounds[section_count] = sec_bounds
            section_tags[section_count] = sec_tag
            section_jump_expected[section_count] = sec_jump
            waypoints[waypoint_count] = np.array(
                [wp_x, lane_center_y, self._sample_local_height(terrain, wp_x, lane_center_y)],
                dtype=np.float32,
            )
            waypoint_count += 1
            section_count += 1

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
            safe_spawn_region=np.array([0.45, 0.95, lane_center_y - 0.3, lane_center_y + 0.3], dtype=np.float32),
            lane_bounds=np.array([0.3, self.env_length - 0.3, y_min, y_max], dtype=np.float32),
            waypoints=waypoints,
            waypoint_count=waypoint_count,
            terminal_goal=terminal_goal,
            section_bounds=section_bounds,
            section_tags=section_tags,
            jump_expected_mask=section_jump_expected,
            edge_mask=edge_mask,
        )
        return terrain, metadata

    def build_gauntlet_lane(self, families: list, obstacles_per_family: int, difficulty_row: int):
        """Build an extended lane with multiple families sequenced."""
        if difficulty_row < 0:
            raise ValueError(f"Parkour difficulty_row must be non-negative, got {difficulty_row}.")
        terrain = terrain_utils.SubTerrain(
            "parkour_gauntlet",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        edge_mask = np.zeros_like(terrain.height_field_raw, dtype=np.uint8)

        lane_center_y = 0.5 * self.env_width
        lane_half_width = 1.2
        spawn_x = 0.55
        goal_x = self.env_length - 0.45
        y_min = lane_center_y - lane_half_width
        y_max = lane_center_y + lane_half_width

        total_obstacles = obstacles_per_family * len(families)
        self._validate_lane_capacity(total_obstacles)

        section_bounds = np.zeros((self.max_sections, 2), dtype=np.float32)
        section_tags = np.zeros((self.max_sections,), dtype=np.int32)
        section_jump_expected = np.zeros((self.max_sections,), dtype=np.bool_)
        waypoints = np.zeros((self.max_waypoints, 3), dtype=np.float32)

        slot_centers = np.linspace(
            self._slot_margin,
            self.env_length - self._slot_margin,
            num=total_obstacles,
            dtype=np.float32,
        )

        row_idx = min(difficulty_row, 3)
        waypoint_count = 0
        section_count = 0

        # Build shuffled obstacle sequence: N of each family, randomly interleaved
        obstacle_sequence = [fam for fam in families for _ in range(obstacles_per_family)]
        np.random.shuffle(obstacle_sequence)

        for slot_idx, family in enumerate(obstacle_sequence):
            slot_center_x = float(slot_centers[slot_idx])
            wp_x, sec_bounds, sec_tag, sec_jump = self._build_single_obstacle(
                terrain, edge_mask, family, row_idx, slot_center_x,
                lane_center_y, y_min, y_max, goal_x,
            )
            section_bounds[section_count] = sec_bounds
            section_tags[section_count] = sec_tag
            section_jump_expected[section_count] = sec_jump
            waypoints[waypoint_count] = np.array(
                [wp_x, lane_center_y, self._sample_local_height(terrain, wp_x, lane_center_y)],
                dtype=np.float32,
            )
            waypoint_count += 1
            section_count += 1

        terminal_goal = np.array(
            [goal_x, lane_center_y, self._sample_local_height(terrain, goal_x, lane_center_y)],
            dtype=np.float32,
        )
        terminal_index = min(waypoint_count, self.max_waypoints - 1)
        waypoints[terminal_index] = terminal_goal
        waypoint_count = terminal_index + 1

        family_id = PARKOUR_FAMILY_IDS[families[0]] if len(families) == 1 else -1
        metadata = ParkourLaneMetadata(
            family=family_id,
            difficulty_row=difficulty_row,
            spawn_pose=np.array(
                [spawn_x, lane_center_y, self._sample_local_height(terrain, spawn_x, lane_center_y), 0.0],
                dtype=np.float32,
            ),
            safe_spawn_region=np.array([0.45, 0.95, lane_center_y - 0.3, lane_center_y + 0.3], dtype=np.float32),
            lane_bounds=np.array([0.3, self.env_length - 0.3, y_min, y_max], dtype=np.float32),
            waypoints=waypoints,
            waypoint_count=waypoint_count,
            terminal_goal=terminal_goal,
            section_bounds=section_bounds,
            section_tags=section_tags,
            jump_expected_mask=section_jump_expected,
            edge_mask=edge_mask,
        )
        return terrain, metadata

    # -- Obstacle parameter tables (indexed by row_idx 0-3) --
    _STAIRS_STEP_HEIGHTS = [0.06, 0.08, 0.10, 0.12]
    _STAIRS_STEP_COUNTS = [2, 3, 4, 4]
    _STAIRS_POST_OFFSETS = [0.90, 0.80, 0.68, 0.58]
    _HURDLE_HEIGHTS = [0.10, 0.14, 0.20, 0.26]
    _HURDLE_LENGTHS = [0.20, 0.22, 0.25, 0.28]
    _HURDLE_Y_HALVES = [0.55, 0.65, 0.75, 0.85]
    _HURDLE_POST_OFFSETS = [0.95, 0.80, 0.68, 0.55]
    _GAP_WIDTHS = [0.24, 0.32, 0.42, 0.50]
    _GAP_Y_HALVES = [0.85, 0.95, 1.05, 1.15]
    _GAP_POST_OFFSETS = [1.00, 0.85, 0.74, 0.62]

    def _build_single_obstacle(self, terrain, edge_mask, family, row_idx, slot_center_x,
                                lane_center_y, y_min, y_max, goal_x):
        """Place a single obstacle and return (waypoint_x, section_bounds, section_tag, jump_expected)."""
        if family == "stairs":
            step_count = self._STAIRS_STEP_COUNTS[row_idx]
            end_x, _ = self._build_stairs_feature(
                terrain=terrain, center_x=slot_center_x,
                y_min=y_min, y_max=y_max,
                step_height=self._STAIRS_STEP_HEIGHTS[row_idx],
                step_count=step_count,
            )
            section_start_x = self._stairs_start_x(slot_center_x, step_count)
            sec_bounds = np.array([section_start_x - 0.1, end_x + 0.3], dtype=np.float32)
            wp_x = min(end_x + self._STAIRS_POST_OFFSETS[row_idx], goal_x)
            return wp_x, sec_bounds, PARKOUR_SECTION_IDS["stairs"], False
        elif family == "hurdle_block":
            block_length = self._HURDLE_LENGTHS[row_idx]
            y_half = self._HURDLE_Y_HALVES[row_idx]
            x0 = slot_center_x - 0.5 * block_length
            x1 = x0 + block_length
            block_y_min = lane_center_y - y_half
            block_y_max = lane_center_y + y_half
            self._fill_rect_height(terrain, x0, x1, block_y_min, block_y_max, self._HURDLE_HEIGHTS[row_idx])
            self._mark_rect_perimeter(edge_mask, x0, x1, block_y_min, block_y_max)
            sec_bounds = np.array([x0 - 0.2, x1 + 0.35], dtype=np.float32)
            wp_x = min(x1 + self._HURDLE_POST_OFFSETS[row_idx], goal_x)
            return wp_x, sec_bounds, PARKOUR_SECTION_IDS["jump"], True
        elif family == "gap":
            gap_width = self._GAP_WIDTHS[row_idx]
            gap_y_half = self._GAP_Y_HALVES[row_idx]
            x0 = slot_center_x - 0.5 * gap_width
            x1 = x0 + gap_width
            gap_y_min = lane_center_y - gap_y_half
            gap_y_max = lane_center_y + gap_y_half
            self._fill_rect_height(terrain, x0, x1, gap_y_min, gap_y_max, -5.0)
            self._mark_gap_edges(edge_mask, x0, x1, gap_y_min, gap_y_max)
            sec_bounds = np.array([x0 - 0.2, x1 + 0.45], dtype=np.float32)
            wp_x = min(x1 + self._GAP_POST_OFFSETS[row_idx], goal_x)
            return wp_x, sec_bounds, PARKOUR_SECTION_IDS["jump"], True
        else:
            raise ValueError(f"Unsupported parkour family '{family}'")

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
        return np.clip(slot_centers, self._slot_margin - 0.2, self.env_length - self._slot_margin + 0.2)

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
