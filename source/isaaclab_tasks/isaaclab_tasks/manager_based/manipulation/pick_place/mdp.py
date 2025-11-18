# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script defines the custom observation, reward, and termination functions for the Piper pick-and-place task.
"""

import torch
from typing import Tuple

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


##
# Observation Functions
##

def object_ee_position(
    env: ManagerBasedRLEnv, eef_link_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Computes the position of the object relative to the end-effector."""
    object: RigidObject = env.scene[object_cfg.name]
    robot = env.scene["robot"]
    ee_pos_w = robot.data.body_pos_w[..., robot.body_names.index(eef_link_name), :]
    object_pos_w = object.data.root_pos_w
    return object_pos_w - ee_pos_w


##
# Reward Functions
##

def object_ee_distance(
    env: ManagerBasedRLEnv, std: float, eef_link_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Calculates a reward based on the distance between the end-effector and the object."""
    object: RigidObject = env.scene[object_cfg.name]
    robot = env.scene["robot"]
    ee_pos_w = robot.data.body_pos_w[..., robot.body_names.index(eef_link_name), :]
    object_pos_w = object.data.root_pos_w
    distance = torch.norm(ee_pos_w - object_pos_w, p=2, dim=-1)
    return torch.exp(-distance / std)


def grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    eef_link_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_action_name: str = "gripper_action",
) -> torch.Tensor:
    """Rewards the agent for closing the gripper when the end-effector is near the object."""
    object: RigidObject = env.scene[object_cfg.name]
    robot = env.scene["robot"]
    ee_pos_w = robot.data.body_pos_w[..., robot.body_names.index(eef_link_name), :]
    object_pos_w = object.data.root_pos_w
    dist_ee_obj = torch.norm(ee_pos_w - object_pos_w, p=2, dim=-1)
    
    gripper_action = env.action_manager.get_action(gripper_action_name)
    is_closing = (gripper_action < 0.0).squeeze(-1)
    
    reward = torch.where(is_closing, torch.exp(-dist_ee_obj / std), torch.zeros_like(dist_ee_obj))
    return reward


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Rewards the agent for lifting the object above a certain height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[..., 2] > minimal_height, 1.0, 0.0)


def object_goal_placement(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    max_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Computes a reward based on how close the object is to the goal region."""
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w

    # Target region center
    goal_x = (min_x + max_x) / 2.0
    goal_y = (min_y + max_y) / 2.0
    
    # Calculate distance to goal only if the object is lifted
    is_lifted = object_pos[..., 2] > minimal_height
    dist_to_goal = torch.norm(object_pos[..., :2] - torch.tensor([goal_x, goal_y], device=env.device), dim=-1)
    
    # Reward is high when close to the goal
    reward = torch.exp(-dist_to_goal / std)
    
    return torch.where(is_lifted, reward, 0.0)


##
# Termination Functions
##

def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str,
    right_wrist_max_x: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Termination condition for the pick and place task.
    Terminates when the object is placed, and the arm is retracted.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot = env.scene["robot"]

    # Object placed in goal (using hardcoded goal region for simplicity)
    object_pos = object.data.root_pos_w
    in_goal_x = (object_pos[..., 0] > 0.40) & (object_pos[..., 0] < 0.85)
    in_goal_y = (object_pos[..., 1] > 0.35) & (object_pos[..., 1] < 0.60)
    is_placed = in_goal_x & in_goal_y

    # Arm retracted
    task_link_pos = robot.data.body_pos_w[..., robot.body_names.index(task_link_name), :]
    is_retracted = task_link_pos[..., 0] < right_wrist_max_x

    return is_placed & is_retracted
