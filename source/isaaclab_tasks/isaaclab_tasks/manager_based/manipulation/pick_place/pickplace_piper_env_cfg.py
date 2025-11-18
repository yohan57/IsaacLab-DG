# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

import carb
from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, 0.3, 1.0203], rot=[1, 0, 0, 0]),
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.04),  # 4cm cube - appropriate size for Piper robot
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),  # Blue color
        ),
    )

    # Piper robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path="/home/neubility-sim/piper_ros/src/piper_description/urdf/piper_description/piper_description.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,  # 팔이 고정된 로봇으로 가정합니다.
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

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Action for the arm joints (1-6)
    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        scale=1.0,  # Using a uniform scale for simplicity, can be tuned later
        use_default_offset=True,
    )

    # Action for the gripper joints (7-8)
    gripper_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        scale=1.0,  # Action value from -1.0 (closed) to 1.0 (open)
        offset=(0.0, 0.0),
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        # Basic observations
        actions = ObsTerm(func=base_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        eef_pos = ObsTerm(func=base_mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["link7"])})

        # Added observations for better learning
        # 1. Object position relative to the end-effector
        object_relative_eef_pos = ObsTerm(
            func=mdp.object_ee_position, params={"eef_link_name": "link7"}
        )
        # 2. Gripper joint positions (to know if it's open or closed)
        gripper_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint7", "joint8"])},
        )
        # 3. Object full pose
        object_pose = ObsTerm(func=base_mdp.root_pose_w, params={"asset_cfg": SceneEntityCfg("object")})


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # Concatenate all observations into a single vector

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    # Success condition: object placed in target region
    # Note: right_wrist_max_x is relaxed for Piper robot (0.5 instead of 0.26)
    # This allows the robot to complete the task without requiring full retraction
    success = DoneTerm(
        func=mdp.task_done_pick_place,
        params={
            "task_link_name": "link7",
            "right_wrist_max_x": 0.5,  # Relaxed from default 0.26 for Piper robot workspace
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Reach the object with end-effector
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "eef_link_name": "link7"},
        weight=5.0,
    )

    # 2. Grasping reward to encourage closing the gripper near the object
    grasping_object = RewTerm(
        func=mdp.grasp_reward,
        params={"eef_link_name": "link7", "std": 0.05},
        weight=2.5,
    )

    # 3. Lift the object
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=10.0,
    )

    # 4. Place the object near the target
    object_goal_placement = RewTerm(
        func=mdp.object_goal_placement,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "min_x": 0.40,
            "max_x": 0.85,
            "min_y": 0.35,
            "max_y": 0.60,
            "max_height": 1.10,
        },
        weight=5.0,
    )

    # Regularization penalties
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.1, 0.1],
                "y": [-0.1, 0.1],
                "z": [0.0, 0.0], # Keep z constant
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlacePiperEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Piper pick-place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    
    # Enable debug print to see USD default values vs configured values
    # Set to True to see what values are actually used from USD file
    scene.robot.actuator_value_resolution_debug_print = True
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation
