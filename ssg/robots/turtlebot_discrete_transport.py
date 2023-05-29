import os

import gym
import igibson
import numpy as np
import pybullet as p
from igibson.object_states.robot_related_states import ObjectsInFOVOfRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from scipy.spatial.transform import Rotation as R


class Turtlebot(TwoWheelRobot):
    """
    Turtlebot robot
    Reference: http://wiki.ros.org/Robots/TurtleBot
    Uses joint velocity control
    """

    def __init__(
        self,
        name=None,
        control_freq=None,
        action_type="continuous",
        action_normalize=True,
        proprio_obs="default",
        reset_joint_pos=None,
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=False,
        **kwargs
    ):
        super().__init__(
            name,
            control_freq,
            action_type,
            action_normalize,
            proprio_obs,
            reset_joint_pos,
            controller_config,
            base_name,
            scale,
            self_collision,
            **kwargs
        )
        self.inventory_constraint = None

    def _create_discrete_action_space(self):
        # Set action list based on controller (joint or DD) used

        # We set straight velocity to be 50% of max velocity for the wheels
        max_wheel_joint_vels = self.control_limits["velocity"][1][self.base_control_idx]
        assert (
            len(max_wheel_joint_vels) == 2
        ), "TwoWheelRobot must only have two base (wheel) joints!"
        assert (
            max_wheel_joint_vels[0] == max_wheel_joint_vels[1]
        ), "Both wheels must have the same max speed!"
        wheel_straight_vel = 0.5 * max_wheel_joint_vels[0] * 2
        wheel_rotate_vel = 0.5 * 3
        if self.controller_config["base"]["name"] == "JointController":
            action_list = [
                [wheel_straight_vel, wheel_straight_vel],
                [-wheel_straight_vel, -wheel_straight_vel],
                [wheel_rotate_vel, -wheel_rotate_vel],
                [-wheel_rotate_vel, wheel_rotate_vel],
                [0, 0],
            ]
        else:
            # DifferentialDriveController
            lin_vel = wheel_straight_vel * self.wheel_radius
            ang_vel = (
                wheel_rotate_vel * self.wheel_radius * 2.0 / self.wheel_axle_length
            )
            action_list = [
                [lin_vel, 0],
                [-lin_vel, 0],
                [0, ang_vel],
                [0, -ang_vel],
                [0, 0],
            ]

        self.action_list = action_list

        # Return this action space
        return gym.spaces.Discrete(5)

    def apply_action(self, action):
        """
        Converts inputted actions into low-level control signals and deploys them on the robot

        :param action: Array[float], n-DOF length array of actions to convert and deploy on the robot
        """
        # Update state
        self.update_state()

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
                            > 0.5
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
                for obj_id in obj_ids:
                    obj = self.simulator.scene.objects_by_id[obj_id]
                    if obj.category == "bowl":
                        x, y, z = obj.get_position()
                        if (
                            np.linalg.norm(
                                np.array((robot_pos[0], robot_pos[1]))
                                - np.array((x, y))
                            )
                            > 0.5
                        ):
                            continue
                        self.inventory.set_position([x, y, 0.2])
                        if self.inventory_constraint:
                            p.removeConstraint(self.inventory_constraint)
                        self.inventory = None
                        for _ in range(5):
                            self.simulator.step()
                        break

    def reset(self):
        """
        Reset function for each specific robot. Can be overwritten by subclass

        By default, sets all joint states (pos, vel) to 0, and resets all controllers.
        """
        for joint, joint_pos in zip(self._joints.values(), self.reset_joint_pos):
            joint.reset_state(joint_pos, 0.0)

        for controller in self._controllers.values():
            controller.reset()

        self.inventory = None
        if self.inventory_constraint:
            p.removeConstraint(self.inventory_constraint)
        self.inventory_constraint = None

    def _deploy_control(self, control, control_type):
        """
        Deploys control signals @control with corresponding @control_type on this robot

        :param control: Array[float], raw control signals to send to the robot's joints
        :param control_type: Array[ControlType], control types for each joint
        """
        # Run sanity check
        joints = self._joints.values()
        assert len(control) == len(control_type) == len(joints), (
            "Control signals, control types, and number of joints should all be the same!"
            "Got {}, {}, and {} respectively.".format(
                len(control), len(control_type), len(joints)
            )
        )

        # Loop through all control / types, and deploy the signal
        for joint, ctrl, ctrl_type in zip(joints, control, control_type):
            if ctrl_type == ControlType.TORQUE:
                joint.set_torque(ctrl)
            elif ctrl_type == ControlType.VELOCITY:
                joint.set_vel(ctrl)
            elif ctrl_type == ControlType.POSITION:
                joint.set_pos(ctrl)
            else:
                raise ValueError("Invalid control type specified: {}".format(ctrl_type))

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Turtlebot"

    @property
    def wheel_radius(self):
        return 0.038

    @property
    def wheel_axle_length(self):
        return 0.23

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([1, 0])

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/turtlebot/turtlebot.urdf")
