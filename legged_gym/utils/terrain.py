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
from .parkour_terrain import PARKOUR_FAMILY_IDS, PARKOUR_SECTION_IDS, ParkourLaneBuilder

class Terrain:
    PARKOUR_FAMILY_IDS = PARKOUR_FAMILY_IDS
    PARKOUR_SECTION_IDS = PARKOUR_SECTION_IDS

    def __init__(self, cfg) -> None:

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
        self._parkour_lane_builder = ParkourLaneBuilder(cfg) if self.parkour_enabled else None
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
                terrain, metadata = self._parkour_lane_builder.build_lane(family, i)
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
            env_origin_z = float(metadata.spawn_pose[2])
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        if metadata is not None:
            self.lane_family[i, j] = metadata.family
            self.lane_difficulty_row[i, j] = metadata.difficulty_row
            self.lane_spawn_pose[i, j] = metadata.spawn_pose
            self.lane_safe_spawn_region[i, j] = metadata.safe_spawn_region
            self.lane_bounds[i, j] = metadata.lane_bounds
            self.lane_waypoints[i, j] = metadata.waypoints
            self.lane_waypoint_counts[i, j] = metadata.waypoint_count
            self.lane_terminal_goal[i, j] = metadata.terminal_goal
            self.lane_section_bounds[i, j] = metadata.section_bounds
            self.lane_section_tags[i, j] = metadata.section_tags
            self.lane_jump_expected_mask[i, j] = metadata.jump_expected_mask
            self.lane_edge_masks[i, j] = metadata.edge_mask
        
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
        self.lane_bounds = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 4), dtype=np.float32)
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
