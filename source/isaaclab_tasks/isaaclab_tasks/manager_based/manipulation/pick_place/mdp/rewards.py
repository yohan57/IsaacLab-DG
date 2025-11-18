# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # Use relative height to environment origin
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(object_height > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_link_name: str = "link7",
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for the tanh kernel.
        object_cfg: Configuration for the object entity.
        robot_cfg: Configuration for the robot entity.
        eef_link_name: Name of the end-effector link.
    
    Returns:
        Reward tensor based on distance between end-effector and object.
    """
    # extract the used quantities (to type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get end-effector position relative to environment origin
    body_pos_w = robot.data.body_pos_w
    eef_idx = robot.data.body_names.index(eef_link_name)
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins
    
    # Get object position relative to environment origin
    object_pos = object.data.root_pos_w - env.scene.env_origins
    
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_pos - eef_pos, dim=1)
    
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_placement(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = -0.15,
    max_y: float = 0.15,
    max_height: float = 1.10,
) -> torch.Tensor:
    """Reward the agent for placing the object near the target region.
    
    This function provides dense rewards based on distance to target region center,
    with bonus rewards when the object is in the target region and at appropriate height.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for the tanh kernel.
        minimal_height: Minimum height for the object to be considered lifted.
        object_cfg: Configuration for the object entity.
        min_x: Minimum x position for target region.
        max_x: Maximum x position for target region.
        min_y: Minimum y position for target region.
        max_y: Maximum y position for target region.
        max_height: Maximum height for successful placement.
    
    Returns:
        Reward tensor based on object placement near target region.
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Extract object position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Compute distance to center of target region
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_distance = torch.sqrt((object_x - center_x) ** 2 + (object_y - center_y) ** 2)
    
    # Base reward: always provide distance-based reward (dense reward)
    # This ensures the agent gets feedback even when far from target
    base_reward = 1 - torch.tanh(center_distance / std)
    
    # Bonus reward when object is in target x-y region
    in_x_range = (object_x >= min_x) & (object_x <= max_x)
    in_y_range = (object_y >= min_y) & (object_y <= max_y)
    in_xy_region = in_x_range & in_y_range
    
    # Bonus reward when object is lifted and below max height
    is_lifted = object_height > minimal_height
    below_max_height = object_height < max_height
    height_ok = is_lifted & below_max_height
    
    # Apply bonuses: multiply base reward by bonus factors when conditions are met
    # This gives extra reward when object is in the right place, but still provides
    # learning signal when object is approaching the target
    placement_reward = base_reward * (1.0 + 0.5 * in_xy_region.float() * height_ok.float())
    
    return placement_reward

