import itertools
import os

import gym
import numpy as np
import pybullet as p
import trimesh
from igibson.external.pybullet_tools.utils import get_all_links, get_center_extent
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils import utils
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

import ssg


class Fetch(TwoWheelRobot, ActiveCameraRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    """

    def __init__(
        self,
        rigid_trunk=False,
        default_trunk_offset=0.365,
        **kwargs,
    ):
        """
        :param reset_joint_pos: None or str or Array[float], if specified, should be the joint positions that the robot
            should be set to during a reset. If str, should be one of {tuck, untuck}, corresponds to default
            configurations for un/tucked modes. If None (default), self.default_joint_pos (untuck mode) will be used
            instead.
        :param rigid_trunk: bool, if True, will prevent the trunk from moving during execution.
        :param default_trunk_offset: float, sets the default height of the robot's trunk
        :param **kwargs: see ManipulationRobot, TwoWheelRobot, ActiveCameraRobot
        """
        # Store args
        self.rigid_trunk = rigid_trunk
        self.default_trunk_offset = default_trunk_offset

        # Run super init
        super().__init__(**kwargs)

    def apply_action(self, action):
        """
        Converts inputted actions into low-level control signals and deploys them on the robot

        :param action: Array[float], n-DOF length array of actions to convert and deploy on the robot
        """

        orn = self.get_orientation()
        rotation = R.from_quat(orn)
        oriented_vector = rotation.apply([2.0, 0, 0])
        oriented_vector[2] = 0

        oriented_vector_rev = rotation.apply([-2.0, 0, 0])
        oriented_vector_rev[2] = 0

        # If we're using discrete action space, we grab the specific action and use that to convert to control
        if action == 0:
            self.set_velocities(
                [[oriented_vector, [0.0] * 3]] * len(self.get_body_ids())
            )
        elif action == 1:
            self.set_velocities(
                [[oriented_vector_rev, [0.0] * 3]] * len(self.get_body_ids())
            )
        elif action == 2:
            rot = R.from_euler("z", -30, degrees=True)
            orientation = rotation * rot
            self.set_orientation(orientation.as_quat())
            self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
        elif action == 3:
            rot = R.from_euler("z", 30, degrees=True)
            orientation = rotation * rot
            self.set_orientation(orientation.as_quat())
            self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
        elif action == 4:
            self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))

        # Update state
        self.update_state()

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Fetch"

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        return gym.spaces.Discrete(5)

    def load(self, simulator):
        # Run super method
        ids = super().load(simulator)

        assert len(ids) == 1, "Fetch robot is expected to have only one body ID."

        return ids

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        dic["trunk_qpos"] = self.joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = self.joint_velocities[self.trunk_control_idx]
        dic["camera_qpos"] = np.array(
            [self.joint_positions[self.camera_control_idx[0]]]
        )
        dic["camera_qvel"] = np.array(
            [self.joint_velocities[self.camera_control_idx[0]]]
        )

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "camera"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        return cfg

    @property
    def default_joint_pos(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.02,  # trunk
                0.0,
                0.0,  # head
                1.1707963267948966,
                1.4707963267948965,
                -0.4,
                1.6707963267948966,
                0.0,
                1.5707963267948966,
            ]
        )

    def reset(self):
        """
        Reset function for each specific robot. Can be overwritten by subclass

        By default, sets all joint states (pos, vel) to 0, and resets all controllers.
        """
        for joint, joint_pos in zip(self._joints.values(), self.default_joint_pos):
            joint.reset_state(joint_pos, 0.0)

        for controller in self._controllers.values():
            controller.reset()

        self._joints["head_tilt_joint"].set_pos(0)
        self._joints["torso_lift_joint"].set_pos(0)
        self._joints["head_pan_joint"].set_pos(0)

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def gripper_link_to_grasp_point(self):
        return {self.default_arm: np.array([0.1, 0, 0])}

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def trunk_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([2])

    @property
    def camera_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([4, 3])

    @property
    def disabled_collision_pairs(self):
        return [
            ["torso_lift_link", "torso_fixed_link"],
            ["caster_wheel_link", "estop_link"],
            ["caster_wheel_link", "laser_link"],
            ["caster_wheel_link", "torso_fixed_link"],
            ["caster_wheel_link", "l_wheel_link"],
            ["caster_wheel_link", "r_wheel_link"],
        ]

    @property
    def model_file(self):
        return os.path.join(ssg.ROOT_PATH, "assets/fetch/fetch_armless.urdf")

    def dump_config(self):
        """Dump robot config"""
        dump = super(Fetch, self).dump_config()
        dump.update(
            {
                "rigid_trunk": self.rigid_trunk,
                "default_trunk_offset": self.default_trunk_offset,
            }
        )
        return dump

    def get_base_aligned_bounding_box(
        self,
        body_id=None,
        link_id=None,
        visual=False,
        xy_aligned=False,
        fallback_to_aabb=False,
    ):
        """Get a bounding box for this object that's axis-aligned in the object's base frame."""
        body_id = self.get_body_ids()[0]

        # Get the base position transform.
        pos, orn = p.getBasePositionAndOrientation(body_id)
        base_com_to_world = utils.quat_pos_to_mat(pos, orn)

        # Compute the world-to-base frame transform.
        world_to_base_com = trimesh.transformations.inverse_matrix(base_com_to_world)

        # Grab the corners of all the different links' bounding boxes. We will later fit a bounding box to
        # this set of points to get our final, base-frame bounding box.
        points = []

        links = [link_id] if link_id is not None else get_all_links(body_id)
        for link in links:
            # If no BB annotation is available, get the AABB for this link.
            aabb_center, aabb_extent = get_center_extent(body_id, link=link)
            aabb_vertices_in_world = aabb_center + np.array(
                list(itertools.product((1, -1), repeat=3))
            ) * (aabb_extent / 2)
            aabb_vertices_in_base_com = trimesh.transformations.transform_points(
                aabb_vertices_in_world, world_to_base_com
            )
            points.extend(aabb_vertices_in_base_com)

        if xy_aligned:
            # If the user requested an XY-plane aligned bbox, convert everything to that frame.
            # The desired frame is same as the base_com frame with its X/Y rotations removed.
            translate = trimesh.transformations.translation_from_matrix(
                base_com_to_world
            )

            # To find the rotation that this transform does around the Z axis, we rotate the [1, 0, 0] vector by it
            # and then take the arctangent of its projection onto the XY plane.
            rotated_X_axis = base_com_to_world[:3, 0]
            rotation_around_Z_axis = np.arctan2(rotated_X_axis[1], rotated_X_axis[0])
            xy_aligned_base_com_to_world = trimesh.transformations.compose_matrix(
                translate=translate, angles=[0, 0, rotation_around_Z_axis]
            )

            # We want to move our points to this frame as well.
            world_to_xy_aligned_base_com = trimesh.transformations.inverse_matrix(
                xy_aligned_base_com_to_world
            )
            base_com_to_xy_aligned_base_com = np.dot(
                world_to_xy_aligned_base_com, base_com_to_world
            )
            points = trimesh.transformations.transform_points(
                points, base_com_to_xy_aligned_base_com
            )

            # Finally update our desired frame.
            desired_frame_to_world = xy_aligned_base_com_to_world
        else:
            # Default desired frame is base CoM frame.
            desired_frame_to_world = base_com_to_world

        # TODO: Implement logic to allow tight bounding boxes that don't necessarily have to match the base frame.
        # All points are now in the desired frame: either the base CoM or the xy-plane-aligned base CoM.
        # Now fit a bounding box to all the points by taking the minimum/maximum in the desired frame.
        aabb_min_in_desired_frame = np.amin(points, axis=0)
        aabb_max_in_desired_frame = np.amax(points, axis=0)
        bbox_center_in_desired_frame = (
            aabb_min_in_desired_frame + aabb_max_in_desired_frame
        ) / 2
        bbox_extent_in_desired_frame = (
            aabb_max_in_desired_frame - aabb_min_in_desired_frame
        )

        # Transform the center to the world frame.
        bbox_center_in_world = trimesh.transformations.transform_points(
            [bbox_center_in_desired_frame], desired_frame_to_world
        )[0]
        bbox_orn_in_world = Rotation.from_matrix(
            desired_frame_to_world[:3, :3]
        ).as_quat()

        return (
            bbox_center_in_world,
            bbox_orn_in_world,
            bbox_extent_in_desired_frame,
            bbox_center_in_desired_frame,
        )
