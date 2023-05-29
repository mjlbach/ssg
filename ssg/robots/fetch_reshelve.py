import os

import gym
import igibson
import numpy as np
from igibson.object_states.robot_related_states import ObjectsInFOVOfRobot
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from scipy.spatial.transform import Rotation as R

import ssg


class FetchReshelve(TwoWheelRobot, ActiveCameraRobot):
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
        self.inventory_constraint = None
        self.inventory = None

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
            # self.set_velocities([[[0.0] * 3, [0.0, 0.0, -10.0]]] * len(self.get_body_ids()))
        elif action == 3:
            rot = R.from_euler("z", 30, degrees=True)
            orientation = rotation * rot
            self.set_orientation(orientation.as_quat())
            self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
            # self.set_velocities([[[0.0] * 3, [0.0, 0.0, 10.0]]] * len(self.get_body_ids()))
        elif action == 4:
            self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
        elif action == 5:
            self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
            obj_ids = self.states[ObjectsInFOVOfRobot].get_value()
            robot_pos = self.get_position()
            if self.inventory == None:
                for obj_id in obj_ids:
                    obj = self.simulator.scene.objects_by_id[obj_id]
                    if obj.category == "notebook":
                        x, y, z = obj.get_position()
                        if (
                            np.linalg.norm(
                                np.array((robot_pos[0], robot_pos[1]))
                                - np.array((x, y))
                            )
                            > 1.0
                        ):
                            continue
                        if self.dump_config().get("use_realistic_constraints", False):
                            obj.set_position([x, y, 0.2])
                            self.inventory_constraint = p.createConstraint(
                                obj.get_body_ids()[0],
                                -1,
                                self.get_body_ids()[0],
                                -1,
                                p.JOINT_FIXED,
                                [0, 0, 0],
                                [-0.125, 0, -0.2],
                                [0, 0, 0],
                            )
                        self.inventory = obj
                        break
            else:
                obj_list = []
                for obj_id in obj_ids:
                    obj = self.simulator.scene.objects_by_id[obj_id]
                    if obj.category == "shelf":
                        obj_list.append(obj)
                obj_list = sorted(
                    obj_list,
                    key=lambda x: np.linalg.norm(
                        x.get_position() - self.get_position()
                    ),
                )
                if len(obj_list) > 0:
                    obj = obj_list[0]
                    self.inventory.states[igibson.object_states.Inside].set_value(
                        obj, True, use_ray_casting_method=True
                    )
                    self.inventory = None
        elif action == 6:
            self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
            obj_ids = self.states[ObjectsInFOVOfRobot].get_value()
            obj_list = []
            for obj_id in obj_ids:
                obj = self.simulator.scene.objects_by_id[obj_id]
                if obj.category == "bottom_cabinet":
                    obj_list.append(obj)
            if len(obj_list) > 0:
                obj_list = sorted(
                    obj_list,
                    key=lambda x: np.linalg.norm(
                        x.get_position() - self.get_position()
                    ),
                )
                obj = obj_list[0]
                obj.force_wakeup()
                if obj.states[igibson.object_states.Open].get_value():
                    obj.states[igibson.object_states.Open].set_value(False, True)
                else:
                    obj.states[igibson.object_states.Open].set_value(True, True)

        # elif action == 5:
        #     self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
        #     joint = self._joints['head_tilt_joint']
        #     desired_joint_state = 0.0
        #     joint.reset_state(desired_joint_state, 0)
        #     joint.set_pos(desired_joint_state)
        # elif action == 6:
        #     self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
        #     joint = self._joints['head_tilt_joint']
        #     desired_joint_state = 0.69
        #     joint.reset_state(desired_joint_state, 0)
        #     joint.set_pos(desired_joint_state)
        # elif action == 7:
        #     self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
        #     joint = self._joints['torso_lift_joint']
        #     joint.reset_state(joint.lower_limit, 0)
        #     joint.set_pos(joint.lower_limit)
        # elif action == 8:
        #     self.set_velocities([[[0.0] * 3, [0.0] * 3]] * len(self.get_body_ids()))
        #     joint = self._joints['torso_lift_joint']
        #     joint.reset_state(joint.upper_limit, 0)
        #     joint.set_pos(joint.upper_limit)

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
        dump = super(FetchReshelve, self).dump_config()
        dump.update(
            {
                "rigid_trunk": self.rigid_trunk,
                "default_trunk_offset": self.default_trunk_offset,
            }
        )
        return dump
