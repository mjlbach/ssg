import copy

import numpy as np
import pybullet as p
from igibson.object_states import OnTop
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.timeout import Timeout

from ssg.envs.floor_sampler import sample_on_floor
from ssg.envs.termination_conditions import PickTermination
from ssg.tasks.pick_reward import (
    AdjacencyReward,
    PickReward,
    PotentialReward,
    ViewReward,
    VisitationReward,
)
from ssg.tasks.utils import import_object


class PickTask(BaseTask):
    def __init__(self, env):
        super(PickTask, self).__init__(env)
        self.scene = env.scene
        self.simulator = env.simulator

        self.reward_functions = []
        if "pick" in self.config["reward_functions"]:
            self.reward_functions.append(PickReward(self.config))
        if "visitation" in self.config["reward_functions"]:
            self.reward_functions.append(VisitationReward(self.config))
        if "potential" in self.config["reward_functions"]:
            self.reward_functions.append(PotentialReward(self.config))
        if "adjacency" in self.config["reward_functions"]:
            self.reward_functions.append(AdjacencyReward(self.config))
        if "view" in self.config["reward_functions"]:
            self.reward_functions.append(ViewReward(self.config))

        self.termination_conditions = [
            Timeout(self.config),
            PickTermination(self.config),
        ]
        self.choose_task()
        self.task_obs_dim = 3

    def choose_task(self):
        self.source = import_object(
            self.simulator,
            igibson_category="breakfast_table",
            model="1b4e6f9dd22a8c628ef9d976af675b86",
        )
        self.items = []
        for _ in range(1):
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
        env.robots[0].inventory = None
        if env.robots[0].inventory_constraint:
            p.removeConstraint(env.robots[0].inventory_constraint)
            env.robots[0].inventory_constraint = None

        # This is absolutely critical, reset doors
        env.scene.reset_scene_objects()

        if self.config.get("randomize_obj_reset", True):
            room_choices = copy.deepcopy(self.scene.room_ins_name_to_ins_id)

            source_room = np.random.choice(np.array(list(room_choices)))
            sample_on_floor(self.source, self.scene, room=source_room)
            for _ in range(10):
                env.simulator.step()

            for item in self.items:
                for _ in range(5):
                    success = item.states[OnTop].set_value(self.source, True, True)
                    if success:
                        break

            if not self.items[0].states[OnTop].get_value(self.source):
                print("warning: sampling failed")

        else:
            Exception("Unsupported")
