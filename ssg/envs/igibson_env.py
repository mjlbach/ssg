import logging
from collections import OrderedDict

import gym
import gym.spaces
import numpy as np
import pybullet as p
from igibson.envs.igibson_env import iGibsonEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.mesh_util import ortho
from ray.rllib.utils.spaces.repeated import Repeated
from transforms3d.euler import quat2euler

from ssg import REGISTERED_TASKS
from ssg.obs.graph import Graph
from ssg.obs.object_set import ObjectSet
from ssg.obs.slam import SimpleSlam, VisitedSlam
from gym.spaces import Box

def force_sleep(obj, body_id=None):
    if body_id is None:
        body_id = obj.get_body_ids()[0]

    activationState = p.ACTIVATION_STATE_SLEEP + p.ACTIVATION_STATE_DISABLE_WAKEUP
    p.changeDynamics(body_id, -1, activationState=activationState)

class iGibsonEnv(iGibsonEnv):
    """
    iGibson Environment (OpenAI Gym interface).
    """

    metadata = {"render.modes": ["rgb_array"]}

    def load_task_setup(self):
        """
        Load task setup.
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep**2)

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get("collision_ignore_body_b_ids", [])
        )
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get("collision_ignore_link_a_ids", [])
        )

        assert (
            drop_distance < self.initial_pos_z_offset
        ), "initial_pos_z_offset is too small for collision checking"

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            "texture_randomization_freq", None
        )
        self.object_randomization_freq = self.config.get(
            "object_randomization_freq", None
        )

        task_class = self.config["task"]
        self.task = REGISTERED_TASKS[task_class](self)

    def load_observation_space(self):
        """
        Load observation space.
        """
        self.output = self.config["output"]
        self.raw_outputs = self.config.get("raw_outputs", [])
        self.image_width = self.config.get("image_width", 128)
        self.image_height = self.config.get("image_height", 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = set()
        scan_modalities = []

        if self.config["task"] == "search":
            vision_modalities.update(["ins_seg"])

        if "rgb" in self.output:
            observation_space["rgb"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0,
                high=1,
            )
            vision_modalities.update(["rgb"])
        if "depth" in self.output:
            observation_space["depth"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0,
                high=1.0,
            )
            vision_modalities.update(["depth"])

        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0,
                high=MAX_CLASS_COUNT,
            )
            vision_modalities.update(["seg"])

        if "ins_seg" in self.output:
            observation_space["ins_seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0,
                high=MAX_INSTANCE_COUNT,
            )
            vision_modalities.update(["ins_seg"])

        if "sem_map" in self.output:
            observation_space["sem_map"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=-np.inf, high=np.inf
            )
            vision_modalities.update(["seg"])

        if "ego_sem_map" in self.output:
            resolution = self.config.get("metric_map_resolution", 128)
            voxel_size = self.config.get("metric_map_voxel_size", 0.23)
            observation_space["ego_sem_map"] = self.build_obs_space(
                shape=(resolution, resolution, 1), low=-np.inf, high=np.inf
            )

            vision_modalities.update(["seg"])
            vision_modalities.update(["pc"])
            vision_modalities.update(["ins_seg"])

            if self.config.get("dynamic_remap", True):
                object_categories = list(self.simulator.scene.objects_by_category)
            else:
                object_categories = None

            # Need to figure this out
            if "env_ids" in self.config and len(self.config["env_ids"]) > 1:
                assert object_categories == None

            self.slam = SimpleSlam(
                # normalization_map=self.config["semantic_normalization_map"]
                # normalization_map=remapped_ids,
                object_categories=object_categories,
                resolution=resolution,
                voxel_size=voxel_size,
                egocentric=self.config.get("use_egocentric_projection", False),
            )

        if "agent_path" in self.output:
            resolution = self.config.get("metric_map_resolution", 128)
            voxel_size = self.config.get("metric_map_voxel_size", 0.23)
            observation_space["agent_path"] = self.build_obs_space(
                shape=(resolution, resolution, 1), low=-np.inf, high=np.inf
            )
            self.agent_slam = VisitedSlam(
                resolution=resolution,
                voxel_size=voxel_size,
                egocentric=self.config.get("use_egocentric_projection", False),
            )

        if "multichannel_map" in self.output:
            resolution = self.config.get("metric_map_resolution", 128)
            voxel_size = self.config.get("metric_map_voxel_size", 0.23)
            observation_space["multichannel_map"] = self.build_obs_space(
                shape=(resolution, resolution, 2), low=-np.inf, high=np.inf
            )

            vision_modalities.update(["seg"])
            vision_modalities.update(["pc"])
            vision_modalities.update(["ins_seg"])

            if self.config.get("dynamic_remap", True):
                object_categories = list(self.simulator.scene.objects_by_category)
            else:
                object_categories = None

            # Need to figure this out
            if "env_ids" in self.config and len(self.config["env_ids"]) > 1:
                assert object_categories == None

            self.slam = SimpleSlam(
                # normalization_map=self.config["semantic_normalization_map"]
                # normalization_map=remapped_ids,
                object_categories=object_categories,
                resolution=resolution,
                voxel_size=voxel_size,
                egocentric=self.config.get("use_egocentric_projection", False),
            )
            self.agent_slam = VisitedSlam(
                resolution=resolution,
                voxel_size=voxel_size,
                egocentric=self.config.get("use_egocentric_projection", False),
            )

        if "agent_pose" in self.output:
            observation_space["agent_pose"] = self.build_obs_space(
                shape=(6,), low=-np.inf, high=np.inf
            )

        if "task_obs" in self.output:
            observation_space["task_obs"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )

        if "scene_graph" in self.output:
            vision_modalities.update(["ins_seg"])
            self.scene_graph = Graph(
                self,
                features=self.config["node_features"],
                edge_groups=self.config["edge_features"],
                full_observability=self.config.get("full_observability", False),
            )
            observation_dict = {
                    "nodes": Repeated(
                        Box(
                            low=-np.inf, high=np.inf, shape=(self.scene_graph.node_dim,), dtype=np.float32
                        ),
                        max_len=150,
                    ),
            }

            for edge_type in self.scene_graph.edge_groups:
                observation_dict[edge_type] = Repeated(
                    Box(low=0, high=1000, shape=(2,), dtype=np.int64),
                    max_len=300,
                )

            observation_space["scene_graph"] = gym.spaces.Dict(observation_dict)

        if "object_set" in self.output:
            vision_modalities.update(["ins_seg"])
            self.object_set = ObjectSet(
                self, full_observability=self.config["full_observability"]
            )
            observation_space["object_set"] = Repeated(
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.object_set.obj_dim,),
                    dtype=np.float32,
                ),
                max_len=self.config["max_objects"],
            )

        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            self.occupancy_grid_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.grid_resolution, self.grid_resolution, 1)
            )
            observation_space["occupancy_grid"] = self.occupancy_grid_space
            scan_modalities.append("occupancy_grid")

        if "proprioception" in self.output:
            observation_space["proprioception"] = self.build_obs_space(
                shape=(self.robots[0].proprioception_dim,), low=-np.inf, high=np.inf
            )

        if "task_graph_obs" in self.output:
            observation_space["task_graph_obs"] = self.build_obs_space(
                shape=(self.task.task_raw_category_obs_dim,), low=0.0, high=1.0
            )

        vision_modalities.update(self.raw_outputs)

        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, list(vision_modalities))

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        # For ray, observations need to be ordered by key.
        # See complex_input_net.py
        self.observation_space = gym.spaces.Dict(dict(observation_space))
        self.sensors = sensors

    def get_state(self):
        """
        Get the current observation.

        :return: observation as a dictionary
        """
        state = OrderedDict()
        raw_state = OrderedDict()
        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)

        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                if modality in self.output:
                    state[modality] = vision_obs[modality]
                else:
                    raw_state[modality] = vision_obs[modality]
            raw_state.update(state)

        if "sem_map" in self.output:
            ## rendering topdown
            assert self.simulator.renderer
            shadow_hidden_instances = [
                i
                for i in self.simulator.renderer.instances
                if not i.shadow_caster and not i.hidden
            ]
            for instance in shadow_hidden_instances:
                instance.hidden = True
            self.simulator.renderer.update_hidden_highlight_state(
                shadow_hidden_instances
            )
            original_P = np.copy(self.simulator.renderer.P)
            self.simulator.renderer.P = ortho(-5, 5, -5, 5, -10, 20.0)
            self.simulator.renderer.set_camera([0, 0, 3], [0, 0, 0], [0, 1, 0])
            frame = self.simulator.renderer.render(modes=("seg"))

            self.simulator.renderer.P = original_P
            for instance in shadow_hidden_instances:
                instance.hidden = False
            self.simulator.renderer.update_hidden_highlight_state(
                shadow_hidden_instances
            )

            state["sem_map"] = frame[0][:, :, 0:1] / MAX_CLASS_COUNT

        if "ego_sem_map" in self.output:
            state["ego_sem_map"] = self.slam.update(
                raw_state, self.simulator.renderer.V
            )

        if "agent_path" in self.output:
            position, orientation = self.robots[0].get_position_orientation()
            state["agent_path"] = self.agent_slam.update(
                position, orientation, extrinsic=self.simulator.renderer.V
            )

        if "agent_pose" in self.output:
            position, orientation = self.robots[0].get_position_orientation()
            orientation = quat2euler(orientation)
            state["agent_pose"] = np.hstack((position, orientation))

        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]

        if "scene_graph" in self.output:
            ins_seg = raw_state["ins_seg"][..., 0]
            self.scene_graph.update(ins_seg)
            state["scene_graph"] = self.scene_graph.to_ray()

        if "object_set" in self.output:
            ins_seg = raw_state["ins_seg"][..., 0]
            self.object_set.update(ins_seg)
            state["object_set"] = self.object_set.to_numpy()

        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)

        if "proprioception" in self.output:
            state["proprioception"] = np.array(self.robots[0].get_proprioception())

        if "multichannel_map" in self.output:
            ego_sem_map = self.slam.update(raw_state, self.simulator.renderer.V)
            raw_state["ego_sem_map"] = ego_sem_map
            position, orientation = self.robots[0].get_position_orientation()
            agent_path = self.agent_slam.update(
                position, orientation, extrinsic=self.simulator.renderer.V
            )
            state["multichannel_map"] = np.concatenate(
                [ego_sem_map, agent_path], axis=-1
            )

        if "task_graph_obs" in self.output:
            task_raw_category_obs = self.task.get_task_raw_category_obs(self)
            task_graph_obs = [self.scene_graph.category_mapping[cat]
                              for cat in task_raw_category_obs]
            state["task_graph_obs"] = np.array(task_graph_obs, dtype=np.float32)

        self.last_state = raw_state
        state = OrderedDict(sorted(list(state.items())))
        return state

    def render(self, mode="rgb"):
        if "rgb" in self.last_state:
            return (self.last_state["rgb"] * 255).astype(np.uint8)
        else:
            rgb = self.simulator.renderer.render_robot_cameras(modes=("rgb"))[0]
            return (rgb[..., :3] * 255).astype(np.uint8)

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        body_id = obj.get_body_ids()[0]

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(2.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                if (np.abs(obj.get_velocities()[0][1]) < 1e-2).all():
                    land_success = True
                    break

        if not land_success:
            logging.warning("Object failed to land.")

        if is_robot:
            obj.reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0
        if "ego_sem_map" in self.output or "multichannel_map" in self.output:
            self.slam.reset()
        if "agent_path" in self.output or "multichannel_map" in self.output:
            self.agent_slam.reset()
        if "scene_graph" in self.output:
            self.scene_graph.reset()

    def reset(self):
        """
        Reset episode.
        """
        self.randomize_domain()
        # Move robot away from the scene.
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset(self)
        self.simulator.sync(force_sync=True)
        self.reset_variables()
        preseed_mode = self.config.get("preseed_scenegraph", None)

        if "scene_graph" in self.output:
            bids = []
            if preseed_mode == "all":
                for id, obj in self.simulator.scene.objects_by_id.items():  # type: ignore
                    xy = obj.get_position()[:2]
                    room_instance = self.simulator.scene.get_room_instance_by_point(xy) #type: ignore
                    if room_instance is not None:
                        bids.append(id)
            if preseed_mode == "all_no_goal":
                for id, obj in self.simulator.scene.objects_by_id.items():  # type: ignore
                    xy = obj.get_position()[:2]
                    room_instance = self.simulator.scene.get_room_instance_by_point(xy) #type: ignore
                    if room_instance is not None and obj != self.task.target_obj:
                        bids.append(id)
            self.scene_graph.update_bids(bids)

        if "multichannel_map" in self.output:
            if preseed_mode == "all_no_goal":
                self.slam.load()
                self.step(3)
                self.current_step = 1
            

        state = self.get_state()

        return state

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        robot = self.robots[0]
        if hasattr(robot, "inventory") and robot.inventory != None:
            # print('eyes', self.links['kinect_camera'].get_position())
            # (Pdb++) pp self.links['camera_rgb_grame'].get_position()
            # (Pdb++) pp self.links['camera_rgb_frame'].get_position()
            # (Pdb++) pp self.links['scan_link'].get_position()
            # (Pdb++) pp self.links['kinect_camera'].get_position()
            pos, orn = robot.links["eyes"].get_position_orientation()
            pos, orn = p.multiplyTransforms(pos, orn, [0.2, 0, 0.2], [0, 0, 0, 1])
            robot.inventory.force_wakeup()
            robot.inventory.set_position_orientation(pos, orn)
            self.simulator.sync()

        state = self.get_state()
        info = {}
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            info["last_observation"] = state
            state = self.reset()

        return state, reward, done, info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=[
            "headless",
            "headless_tensor",
            "gui_interactive",
            "gui_non_interactive",
        ],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = iGibsonEnv(
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 10.0,
        physics_timestep=1.0 / 40.0,
    )

    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print("reward", reward)
            if done:
                break
        print("Episode finished after {} timesteps.".format(env.current_step))
    env.close()
