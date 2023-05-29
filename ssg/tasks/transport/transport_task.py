import copy

import numpy as np
import pybullet as p
from igibson.object_states import OnTop
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.timeout import Timeout

from ssg.envs.floor_sampler import sample_on_floor
from ssg.envs.termination_conditions import TransportTermination
from ssg.tasks.transport_reward import (
    PickReward,
    PlaceReward,
    PotentialReward,
    VisitationReward,
)
from ssg.tasks.utils import import_object


class TransportTask(BaseTask):
    def __init__(self, env):
        super(TransportTask, self).__init__(env)

        self.scene = env.scene
        self.simulator = env.simulator

        self.reward_functions = []
        if "place" in self.config["reward_functions"]:
            self.reward_functions.append(PlaceReward(self.config))
        if "pick" in self.config["reward_functions"]:
            self.reward_functions.append(PickReward(self.config))
        if "visitation" in self.config["reward_functions"]:
            self.reward_functions.append(VisitationReward(self.config))
        if "potential" in self.config["reward_functions"]:
            self.reward_functions.append(PotentialReward(self.config))

        self.termination_conditions = [
            Timeout(self.config),
            TransportTermination(self.config),
        ]
        self.choose_task()
        self.task_obs_dim = 3

        # Debugging visualization
        # cyl_length = 0.2
        # self.visible_path = self.config.get("visible_path", True)
        #
        # if env.scene.build_graph:
        #     self.num_waypoints_vis = 250
        #     self.waypoints_vis = [
        #         VisualMarker(
        #             visual_shape=p.GEOM_CYLINDER,
        #             rgba_color=[0, 1, 0, 0.3],
        #             radius=0.1,
        #             length=cyl_length,
        #             initial_offset=[0, 0, cyl_length / 2.0],
        #         )
        #         for _ in range(self.num_waypoints_vis)
        #     ]
        #     for waypoint in self.waypoints_vis:
        #         env.simulator.import_object(waypoint)
        #         # The path to the target may be visible
        #         for instance in waypoint.renderer_instances:
        #             instance.hidden = not self.visible_path
        #
        #

    def choose_task(self):
        self.source = import_object(
            self.simulator,
            igibson_category="breakfast_table",
            model="1b4e6f9dd22a8c628ef9d976af675b86",
        )
        self.target = import_object(
            self.simulator,
            igibson_category="shelf",
            model="b7697d284c38fcade76bc197b3a3ffc0",
        )
        self.items = []
        for _ in range(self.config["num_objects"]):
            item = import_object(
                self.simulator, igibson_category="notebook", model="12_1"
            )
            self.items.append(item)

    def get_reward(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate reward functions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        reward = 0.0
        categorical_reward_info = {}
        for reward_function in self.reward_functions:
            category_reward = reward_function.get_reward(self, env)
            reward += category_reward
            categorical_reward_info[reward_function.name] = category_reward

        categorical_reward_info["total"] = reward
        info["reward_breakdown"] = categorical_reward_info

        return reward, info

    def get_task_obs(self, env):
        return env.robots[0].get_position()

    def reset_agent(self, env):
        env.robots[0].reset()
        if self.config.get("randomize_agent_reset", False):
            room = np.random.choice(
                np.array(list(self.scene.room_ins_name_to_ins_id.keys()))
            )
            sample_on_floor(env.robots[0], env.simulator.scene, room)
        else:
            env.land(env.robots[0], [1.0, 2.0, 0.0], [0, 0, 0])

    def reset_scene(self, env):
        # Hack because of the order of scene/robot reset :(
        if env.robots[0].inventory_constraint:
            p.removeConstraint(env.robots[0].inventory_constraint)
            env.robots[0].inventory_constraint = None

        # This is absolutely critical, reset doors
        env.scene.reset_scene_objects()

        if self.config.get("randomize_obj_reset", True):
            room_choices = copy.deepcopy(self.scene.room_ins_name_to_ins_id)

            source_room = np.random.choice(np.array(list(room_choices)))
            sample_on_floor(self.source, self.scene, room=source_room)
            env.simulator.step()

            del room_choices[source_room]
            target_room = np.random.choice(np.array(list(room_choices)))
            sample_on_floor(self.target, self.scene, room=target_room)

            for item in self.items:
                # item.set_position((x, y, i))
                item.states[OnTop].set_value(self.source, True, True)
                env.simulator.step()

            if not self.items[0].states[OnTop].get_value(self.source):
                print("warning: sampling failed")

        else:
            Exception("Unsupported")

    # def get_shortest_path(self, env, from_initial_pos=False, entire_path=False):
    #     """
    #     Get the shortest path and geodesic distance from the robot or the initial position to the target position
    #
    #     :param env: environment instance
    #     :param from_initial_pos: whether source is initial position rather than current position
    #     :param entire_path: whether to return the entire shortest path
    #     :return: shortest path and geodesic distance to the target position
    #     """
    #     source = env.robots[0].get_position()[:2]
    #     target = self.source.get_position()[:2]
    #     return env.scene.get_shortest_path(self.floor_num, source, target, entire_path=entire_path)
    #
    # def step_visualization(self, env):
    #     self.floor_num = 0
    #     cyl_length = 0.2
    #     if env.scene.build_graph:
    #         shortest_path, _ = self.get_shortest_path(env, entire_path=True)
    #         floor_height = env.scene.get_floor_height(self.floor_num)
    #         num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
    #         for i in range(num_nodes):
    #             self.waypoints_vis[i].set_position(
    #                 pos=np.array([shortest_path[i][0], shortest_path[i][1], floor_height])
    #             )
    #         for i in range(num_nodes, self.num_waypoints_vis):
    #             self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 100.0]))
    #
