import numpy as np
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.timeout import Timeout

from ssg.utils.floor_sampler import sample_on_floor
from ssg.utils.object_utils import import_object

from .search_reward import (
    ObservationReward,
    PixelReward,
    PotentialReward,
    SearchReward,
    VisitationReward,
)
from .search_termination import SearchTermination


class SearchTask(BaseTask):
    def __init__(self, env):
        super(SearchTask, self).__init__(env)

        self.scene = env.scene
        self.simulator = env.simulator

        self.reward_functions = []
        if "search" in self.config["reward_functions"]:
            self.reward_functions.append(SearchReward(self.config))
        if "pixel" in self.config["reward_functions"]:
            self.reward_functions.append(PixelReward(self.config))
        if "visitation" in self.config["reward_functions"]:
            self.reward_functions.append(VisitationReward(self.config))
        if "potential" in self.config["reward_functions"]:
            self.reward_functions.append(PotentialReward(self.config))
        if "observation" in self.config["reward_functions"]:
            self.reward_functions.append(ObservationReward(self.config))

        self.termination_conditions = [
            Timeout(self.config),
            SearchTermination(self.config),
        ]
        self.choose_task()
        self.task_obs_dim = 3

    def choose_task(self):
        # obj_pro = self.import_object(wordnet_category = 'microwave.n.02' , model='7128')
        obj_pro = import_object(
            self.simulator,
            igibson_category="breakfast_table",
            model="1b4e6f9dd22a8c628ef9d976af675b86",
        )
        self.target_obj = obj_pro
        room = np.random.choice(np.array(list(self.scene.room_ins_name_to_ins_id)))
        sample_on_floor(obj_pro, self.scene, room=room)

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
        # This is absolutely critical, reset doors
        env.scene.reset_scene_objects()

        if self.config.get("randomize_obj_reset", True):
            room = np.random.choice(
                np.array(list(self.scene.room_ins_name_to_ins_id.keys()))
            )
            sample_on_floor(self.target_obj, self.scene, room=room)
        else:
            self.target_obj.set_position(
                np.array([0.79999999, -3.19999984, 0.48399325])
            )
