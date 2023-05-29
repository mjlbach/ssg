import igibson
import numpy as np
import pybullet as p
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.timeout import Timeout

from ssg.envs.floor_sampler import sample_on_floor
from ssg.envs.termination_conditions import ReshelveTermination
from ssg.tasks.reshelve_reward import PickReward, PlaceReward, VisitationReward
from ssg.tasks.utils import import_object


class ReshelveTask(BaseTask):
    def __init__(self, env):
        super(ReshelveTask, self).__init__(env)

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
        if "observation" in self.config["reward_functions"]:
            self.reward_functions.append(ObservationReward(self.config))

        self.termination_conditions = [
            Timeout(self.config),
            ReshelveTermination(self.config),
        ]
        self.choose_task()
        self.task_obs_dim = 3

    def choose_task(self):
        self.sources = []
        obj = import_object(
            self.simulator, igibson_category="bottom_cabinet", model="48859"
        )
        # 0.9, -3.4
        self.sources.append(obj)
        self.sources[0].set_position_orientation((1.7, 0, 0.59), (0.7071, 0.7071, 0, 0))

        # 1, -1
        obj = import_object(
            self.simulator, igibson_category="bottom_cabinet", model="48859"
        )
        self.sources.append(obj)
        self.sources[1].set_position_orientation((0, -3.8, 0.59), (0, 0, 1, 0))

        # 0, -3.5
        self.target = import_object(
            self.simulator,
            igibson_category="shelf",
            model="1170df5b9512c1d92f6bce2b7e6c12b7",
        )
        self.target.set_position_orientation((-1.9, 0, 0.59), (-0.7071, 0.7071, 0, 0))

        self.books = []
        for _ in range(4):
            obj = import_object(
                self.simulator, igibson_category="notebook", model="12_1"
            )
            self.books.append(obj)

        self.distractors = []
        for _ in range(2):
            lemon = import_object(
                self.simulator, igibson_category="lemon", model="07_1"
            )
            self.distractors.append(lemon)

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
        if env.robots[0].inventory_constraint:
            p.removeConstraint(env.robots[0].inventory_constraint)
            env.robots[0].inventory_constraint = None

        # This is absolutely critical, reset doors
        env.scene.reset_scene_objects()

        if self.config.get("randomize_obj_reset", True):
            # self.sources[0].set_position_orientation((1.7, 0, 0.59), (0.7071, 0.7071, 0, 0))
            self.sources[0].states[igibson.object_states.Open].set_value(False, True)

            # 1, -1
            # self.sources[1].set_position_orientation((0, -3.8, 0.59), (0, 0, 1, 0))
            self.sources[1].states[igibson.object_states.Open].set_value(False, True)

            # 0, -3.5
            # self.target.set_position_orientation((-1.9, 0, 0.59), (-0.7071, 0.7071, 0, 0))

            for obj in self.books:
                source = np.random.choice(self.sources)
                obj.states[igibson.object_states.Inside].set_value(source, True, True)

            for obj in self.distractors:
                source = np.random.choice(self.sources)
                obj.states[igibson.object_states.Inside].set_value(source, True, True)
        else:
            Exception("Unsupported")
