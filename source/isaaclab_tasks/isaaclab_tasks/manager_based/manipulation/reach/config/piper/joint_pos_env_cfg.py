# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg


##
# Environment configuration
##


@configclass
class PiperReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Piper robot configuration
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UsdFileCfg(
                usd_path="/home/neubility-sim/piper_ros/src/piper_description/urdf/piper_description/piper_description.usd",
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.9996),
                joint_pos={
                    "joint1": 0.0,
                    "joint2": 0.0,
                    "joint3": 0.0,
                    "joint4": 0.0,
                    "joint5": 0.0,
                    "joint6": 0.0,
                    "joint7": 0.0,
                    "joint8": 0.0,
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"],
                    # Effort limits from URDF (N⋅m for revolute, N for prismatic)
                    effort_limit_sim={
                        "joint1": 100.0,
                        "joint2": 100.0,
                        "joint3": 100.0,
                        "joint4": 100.0,
                        "joint5": 100.0,
                        "joint6": 100.0,
                        "joint7": 10.0,  # Gripper has lower effort limit
                        "joint8": 10.0,
                    },
                    # Velocity limits from URDF (rad/s for revolute, m/s for prismatic)
                    velocity_limit_sim={
                        "joint1": 5.0,
                        "joint2": 5.0,
                        "joint3": 5.0,
                        "joint4": 5.0,
                        "joint5": 5.0,
                        "joint6": 3.0,
                        "joint7": 1.0,  # Gripper has lower velocity limit
                        "joint8": 1.0,
                    },
                    # PD gains - typical values for arm control
                    # Stiffness: higher for better position tracking
                    stiffness={
                        "joint1": 100.0,
                        "joint2": 100.0,
                        "joint3": 100.0,
                        "joint4": 100.0,
                        "joint5": 100.0,
                        "joint6": 100.0,
                        "joint7": 50.0,  # Lower stiffness for gripper
                        "joint8": 50.0,
                    },
                    # Damping: critical damping ratio ~0.7-1.0
                    damping={
                        "joint1": 10.0,
                        "joint2": 10.0,
                        "joint3": 10.0,
                        "joint4": 10.0,
                        "joint5": 10.0,
                        "joint6": 10.0,
                        "joint7": 5.0,  # Lower damping for gripper
                        "joint8": 5.0,
                    },
                ),
            },
        )

        # Override rewards to use link7 as end-effector
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link7"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link7"]
        
        # Adjust reward weights to prioritize position over orientation
        # This helps the agent focus on reaching the target position first
        self.rewards.end_effector_position_tracking.weight = -0.5  # Increased from -0.2
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.2  # Increased from 0.1
        self.rewards.end_effector_orientation_tracking.weight = -0.05  # Decreased from -0.1

        # Override actions for Piper robot
        # Use joint-specific scales based on URDF limits to ensure all joints can move effectively
        # Scale is relative to joint limits: scale * action_range should cover reasonable portion of joint range
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],  # Exclude gripper joints
            scale={
                "joint1": 1.0,  # Range: [-2.618, 2.168] -> scale 1.0 covers ±1.0 rad
                "joint2": 1.0,  # Range: [0, 3.14] -> scale 1.0 covers ±1.0 rad from default
                "joint3": 1.0,  # Range: [-2.967, 0] -> scale 1.0 covers ±1.0 rad
                "joint4": 0.8,  # Range: [-1.745, 1.745] -> scale 0.8 covers ±0.8 rad
                "joint5": 0.6,  # Range: [-1.22, 1.22] -> scale 0.6 covers ±0.6 rad
                "joint6": 1.0,  # Range: [-2.0944, 2.0944] -> scale 1.0 covers ±1.0 rad
            },
            use_default_offset=True,
        )

        # Override command generator body
        # Piper's end-effector (link7) orientation
        self.commands.ee_pose.body_name = "link7"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)  # Adjust based on Piper's end-effector orientation


@configclass
class PiperReachEnvCfg_PLAY(PiperReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

