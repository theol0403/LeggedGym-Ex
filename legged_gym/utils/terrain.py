# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import trimesh

from . import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    PARKOUR_FAMILY_IDS = {
        "stairs": 0,
        "hurdle_block": 1,
        "gap": 2,
        "flat": 3,
    }

    PARKOUR_SECTION_IDS = {"jump": 1, "stairs": 2}

    def __init__(self, cfg: LeggedRobotCfg.terrain) -> None:

        self.cfg = cfg
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.parkour_cfg = getattr(cfg, "parkour", None)
        self.parkour_enabled = bool(
            self.parkour_cfg is not None and getattr(self.parkour_cfg, "enable", False)
        )
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.platform_size = cfg.platform_size
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # row - length, X
        # col - width,  Y
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
    
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.terrain_meshes = []
        self._init_metadata_arrays()
        if cfg.curriculum and cfg.selected:
            raise ValueError("Curriculum and selected terrain cannot be both True.")
        if self.parkour_enabled:
            print("Generating parkour curriculum terrain...")
            self.parkour_curriculum()
        elif cfg.curriculum:
            print("Generating curriculum terrain...")
            self.curiculum()
        elif cfg.selected:
            print("Generating selected terrain...")
            self.selected_terrain()
        else:
            print("Generating randomized terrain...")
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self._add_terrain_border()
            self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)
            
            # self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
            #                                                                                 self.cfg.horizontal_scale,
            #                                                                                 self.cfg.vertical_scale,
            #                                                                                 self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):     # Y
            for i in range(self.cfg.num_rows): # X
                difficulty = i / self.cfg.num_rows      # add difficulty along X axis, row
                choice = j / self.cfg.num_cols + 0.001 # change terrain type along Y axis, col

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def parkour_curriculum(self):
        for j, family in enumerate(self.parkour_family_names):
            for i in range(self.cfg.num_rows):
                terrain, metadata = self.make_parkour_lane(family, i)
                self.add_terrain_to_map(terrain, i, j, metadata=metadata)

    def selected_terrain(self):
        terrain_kwargs = dict(self.cfg.terrain_kwargs)
        terrain_type = terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.length_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)
                
            eval(terrain_type)(terrain, **terrain_kwargs, terrain_type=self.type)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.15 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 0.3 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2: # slope
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, 
                                                 slope=slope, 
                                                 platform_size=self.platform_size,
                                                 terrain_type=self.type)
        elif choice < self.proportions[1]: # random uniform
            terrain_utils.random_uniform_terrain(terrain, 
                                                 min_height=-0.05, 
                                                 max_height=0.05, 
                                                 step=0.005, 
                                                 downsampled_scale=0.2, 
                                                 terrain_type=self.type)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]: # stairs
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, 
                                                 step_width=0.4, 
                                                 step_height=step_height, 
                                                 platform_size=self.platform_size,
                                                 terrain_type=self.type)
        elif choice < self.proportions[4]: # discrete obstacles
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, 
                                                     discrete_obstacles_height, 
                                                     rectangle_min_size, 
                                                     rectangle_max_size, 
                                                     num_rectangles, 
                                                     platform_size=self.platform_size,
                                                     terrain_type=self.type)
        elif choice < self.proportions[5]: # stepping stones
            terrain_utils.stepping_stones_terrain(terrain, 
                                                  stone_size=stepping_stones_size, 
                                                  stone_distance=stone_distance, 
                                                  max_height=0., 
                                                  platform_size=self.platform_size,
                                                  terrain_type=self.type)
        elif choice < self.proportions[6]: # gap
            terrain_utils.gap_terrain(terrain, 
                                      gap_size=gap_size, 
                                      platform_size=self.platform_size,
                                      terrain_type=self.type)
        else: # pit
            terrain_utils.pit_terrain(terrain, 
                                      depth=pit_depth, 
                                      platform_size=self.platform_size,
                                      terrain_type=self.type)

        return terrain

    def make_parkour_lane(self, family: str, difficulty_row: int):
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
        cursor_x = 1.75

        if family == "stairs":
            difficulty_values = self._parkour_difficulty_series(difficulty_row, min(self.max_obstacles, 3))
            for local_difficulty in difficulty_values:
                end_x, _ = self._build_stairs_feature(
                    terrain,
                    start_x=cursor_x,
                    y_min=y_min,
                    y_max=y_max,
                    difficulty=local_difficulty,
                )
                section_bounds[section_count] = np.array([cursor_x - 0.1, end_x + 0.3], dtype=np.float32)
                section_tags[section_count] = self.PARKOUR_SECTION_IDS["stairs"]
                section_jump_expected[section_count] = False
                waypoint_x = min(end_x + 0.4, goal_x)
                waypoints[waypoint_count] = np.array(
                    [waypoint_x, lane_center_y, self._sample_local_height(terrain, waypoint_x, lane_center_y)],
                    dtype=np.float32,
                )
                waypoint_count += 1
                section_count += 1
                cursor_x = end_x + self._lerp(0.65, 0.45, local_difficulty)
        elif family == "hurdle_block":
            difficulty_values = self._parkour_difficulty_series(difficulty_row, min(self.max_obstacles, 4))
            for local_difficulty in difficulty_values:
                block_height = self._lerp(0.10, 0.34, local_difficulty)
                block_length = self._lerp(0.34, 0.80, local_difficulty)
                y_half = self._lerp(0.62, 0.90, local_difficulty)
                x0 = cursor_x
                x1 = x0 + block_length
                block_y_min = lane_center_y - y_half
                block_y_max = lane_center_y + y_half
                self._fill_rect_height(terrain, x0, x1, block_y_min, block_y_max, block_height)
                self._mark_rect_perimeter(edge_mask, x0, x1, block_y_min, block_y_max)
                section_bounds[section_count] = np.array([x0 - 0.2, x1 + 0.35], dtype=np.float32)
                section_tags[section_count] = self.PARKOUR_SECTION_IDS["jump"]
                section_jump_expected[section_count] = True
                waypoint_x = min(x1 + self._lerp(0.7, 0.55, local_difficulty), goal_x)
                waypoints[waypoint_count] = np.array(
                    [waypoint_x, lane_center_y, self._sample_local_height(terrain, waypoint_x, lane_center_y)],
                    dtype=np.float32,
                )
                waypoint_count += 1
                section_count += 1
                cursor_x = x1 + self._lerp(0.6, 0.4, local_difficulty)
        elif family == "gap":
            difficulty_values = self._parkour_difficulty_series(difficulty_row, min(self.max_obstacles, 4))
            for local_difficulty in difficulty_values:
                gap_width = self._lerp(0.22, 0.80, local_difficulty)
                gap_y_half = self._lerp(1.0, 1.15, local_difficulty)
                x0 = cursor_x
                x1 = x0 + gap_width
                gap_y_min = lane_center_y - gap_y_half
                gap_y_max = lane_center_y + gap_y_half
                self._fill_rect_height(terrain, x0, x1, gap_y_min, gap_y_max, -5.0)
                self._mark_gap_edges(edge_mask, x0, x1, gap_y_min, gap_y_max)
                section_bounds[section_count] = np.array([x0 - 0.2, x1 + 0.45], dtype=np.float32)
                section_tags[section_count] = self.PARKOUR_SECTION_IDS["jump"]
                section_jump_expected[section_count] = True
                waypoint_x = min(x1 + self._lerp(0.8, 0.6, local_difficulty), goal_x)
                waypoints[waypoint_count] = np.array(
                    [waypoint_x, lane_center_y, self._sample_local_height(terrain, waypoint_x, lane_center_y)],
                    dtype=np.float32,
                )
                waypoint_count += 1
                section_count += 1
                cursor_x = x1 + self._lerp(0.65, 0.45, local_difficulty)
        elif family != "flat":
            raise ValueError(f"Unsupported parkour family '{family}'")

        terminal_goal = np.array(
            [goal_x, lane_center_y, self._sample_local_height(terrain, goal_x, lane_center_y)],
            dtype=np.float32,
        )
        terminal_index = min(waypoint_count, self.max_waypoints - 1)
        waypoints[terminal_index] = terminal_goal
        waypoint_count = terminal_index + 1
        spawn_pose = np.array(
            [spawn_x, lane_center_y, self._sample_local_height(terrain, spawn_x, lane_center_y), 0.0],
            dtype=np.float32,
        )
        metadata = {
            "family": self.PARKOUR_FAMILY_IDS[family],
            "difficulty_row": difficulty_row,
            "spawn_pose": spawn_pose,
            "safe_spawn_region": np.array([0.6, 1.2, lane_center_y - 0.3, lane_center_y + 0.3], dtype=np.float32),
            "waypoints": waypoints,
            "waypoint_count": waypoint_count,
            "terminal_goal": terminal_goal,
            "section_bounds": section_bounds,
            "section_tags": section_tags,
            "jump_expected_mask": section_jump_expected,
            "edge_mask": edge_mask,
        }
        return terrain, metadata

    def add_terrain_to_map(self, terrain, row, col, metadata=None):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # use the origin height as the max height of a 2mx2m square
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        if metadata is not None:
            env_origin_z = float(metadata["spawn_pose"][2])
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        if metadata is not None:
            self.lane_family[i, j] = metadata["family"]
            self.lane_difficulty_row[i, j] = metadata["difficulty_row"]
            self.lane_spawn_pose[i, j] = metadata["spawn_pose"]
            self.lane_safe_spawn_region[i, j] = metadata["safe_spawn_region"]
            self.lane_waypoints[i, j] = metadata["waypoints"]
            self.lane_waypoint_counts[i, j] = metadata["waypoint_count"]
            self.lane_terminal_goal[i, j] = metadata["terminal_goal"]
            self.lane_section_bounds[i, j] = metadata["section_bounds"]
            self.lane_section_tags[i, j] = metadata["section_tags"]
            self.lane_jump_expected_mask[i, j] = metadata["jump_expected_mask"]
            self.lane_edge_masks[i, j] = metadata["edge_mask"]
        
        if self.type == "trimesh":
            # apply translation to the trimesh, align with the env origin
            translation = np.array([
                start_x * terrain.horizontal_scale,
                start_y * terrain.horizontal_scale,
                0
            ])
            terrain.terrain_mesh.apply_translation(translation)
            self.terrain_meshes.append(terrain.terrain_mesh)
    
    #---------- Protected Methods ----------#
    
    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.terrain_length + 2 * self.cfg.border_size,
            self.cfg.num_cols * self.cfg.terrain_width + 2 * self.cfg.border_size,
        )
        inner_size = (
            self.cfg.num_rows * self.cfg.terrain_length - self.cfg.horizontal_scale, # a small offset to align the subterrain with border
            self.cfg.num_cols * self.cfg.terrain_width - self.cfg.horizontal_scale
        )
        border_center = (
            self.cfg.num_rows * self.cfg.terrain_length / 2 + self.cfg.border_size,
            self.cfg.num_cols * self.cfg.terrain_width / 2 + self.cfg.border_size,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = terrain_utils.make_border(border_size, 
                                                  inner_size, 
                                                  height=abs(self.cfg.border_height), 
                                                  position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)

    def _init_metadata_arrays(self):
        self.metadata_enabled = self.parkour_enabled
        if not self.metadata_enabled:
            self.parkour_family_names = []
            self.family_col_indices = {}
            return

        self.parkour_family_names = list(self.parkour_cfg.families)
        if getattr(self.parkour_cfg, "include_flat_debug", False) and "flat" not in self.parkour_family_names:
            self.parkour_family_names.append("flat")
        if self.cfg.num_cols != len(self.parkour_family_names):
            raise ValueError(
                "Parkour terrain expects cfg.num_cols to match cfg.parkour.families "
                f"(plus optional flat debug). Got num_cols={self.cfg.num_cols}, "
                f"families={self.parkour_family_names}."
            )
        self.family_col_indices = {
            family_name: col_idx for col_idx, family_name in enumerate(self.parkour_family_names)
        }
        self.max_waypoints = int(self.parkour_cfg.max_waypoints)
        self.max_obstacles = int(self.parkour_cfg.max_obstacles)
        self.max_sections = int(self.parkour_cfg.max_sections)
        self.lane_family = np.full((self.cfg.num_rows, self.cfg.num_cols), -1, dtype=np.int32)
        self.lane_difficulty_row = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        self.lane_spawn_pose = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 4), dtype=np.float32)
        self.lane_safe_spawn_region = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 4), dtype=np.float32)
        self.lane_waypoints = np.zeros((self.cfg.num_rows, self.cfg.num_cols, self.max_waypoints, 3), dtype=np.float32)
        self.lane_waypoint_counts = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        self.lane_terminal_goal = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype=np.float32)
        self.lane_section_bounds = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.max_sections, 2), dtype=np.float32
        )
        self.lane_section_tags = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.max_sections), dtype=np.int32
        )
        self.lane_jump_expected_mask = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.max_sections), dtype=np.bool_
        )
        self.lane_edge_masks = np.zeros(
            (self.cfg.num_rows, self.cfg.num_cols, self.length_per_env_pixels, self.width_per_env_pixels),
            dtype=np.uint8,
        )

    def _lerp(self, start: float, end: float, alpha: float) -> float:
        return float(start + (end - start) * alpha)

    def _parkour_difficulty_series(self, difficulty_row: int, count: int):
        if count <= 0:
            return np.zeros((0,), dtype=np.float32)
        max_row = max(self.cfg.num_rows - 1, 1)
        row_fraction = difficulty_row / max_row
        start = self._lerp(0.10, 0.45, row_fraction)
        end = self._lerp(0.60, 1.00, row_fraction)
        return np.linspace(start, end, count, dtype=np.float32)

    def _build_stairs_feature(self, terrain, start_x, y_min, y_max, difficulty):
        step_height = self._lerp(0.05, 0.13, float(difficulty))
        step_count = int(np.round(self._lerp(2.0, 4.0, float(difficulty))))
        step_tread = self._lerp(0.28, 0.38, float(difficulty))
        flat_top = self._lerp(0.18, 0.34, float(difficulty))

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

    def _meters_to_cell(self, x: float, upper_bound: int) -> int:
        return int(np.clip(np.round(x / self.cfg.horizontal_scale), 0, upper_bound - 1))

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
