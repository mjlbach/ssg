import os
from collections import defaultdict

import numpy as np
import pybullet as p
from igibson.object_states import Inside, OnTop, Under
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.assets_utils import get_ig_category_path
from scipy.spatial.transform import Rotation as R

from ssg.utils.object_utils import import_object

from .choice_reward import ChoiceReward, FailurePenalty, PotentialReward, TimePenalty
from .choice_termination import ChoiceTermination, FailureTermination

tree = lambda: defaultdict(tree)


def convert_to_onehot(array):
    arr = np.eye(len(array))
    return {key: arr[idx] for idx, key in enumerate(array)}


class BinaryChoiceTask(BaseTask):
    def __init__(self, env):
        super().__init__(env)

        self.scene = env.scene
        self.simulator = env.simulator

        self.reward_functions = []
        if "choice" in self.config["reward_functions"]:
            self.reward_functions.append(ChoiceReward(self.config))
        if "failure" in self.config["reward_functions"]:
            self.reward_functions.append(FailurePenalty(self.config))
        if "potential" in self.config["reward_functions"]:
            self.reward_functions.append(PotentialReward(self.config))
        if "time_penalty" in self.config["reward_functions"]:
            self.reward_functions.append(TimePenalty(self.config))

        self.termination_conditions = []
        self.termination_conditions.append(Timeout(self.config))
        if "termination_functions" in self.config:
            if "choice" in self.config["termination_functions"]:
                self.termination_conditions.append(ChoiceTermination(self.config))
            if "failure" in self.config["termination_functions"]:
                self.termination_conditions.append(FailureTermination(self.config))

        self.stash_idx = 0
        self.mass_cache = {}
        self.constraints = {}
        self.initialize()

        self.task_obs_dim = 1

        self.associated_categories = ["breakfast_table", "shelf"]
        self.associated_to_onehot = convert_to_onehot(self.associated_categories)

    def initialize(self):
        # obj_pro = self.import_object(wordnet_category = 'microwave.n.02' , model='7128')
        self.support_surfaces = tree()
        self.all_objects = []

        # Import all shelves
        shelves = {
            # 'f000edc1cfdeda11bee0494534c13f8c', # Mostly solid
            # 'e9850d3c5243cc60f62af8489541547b', # Very wide
            "de3b28f255111570bc6a557844fbbce9": {"scale": (2, 2, 1.5)},
            # 'b7697d284c38fcade76bc197b3a3ffc0',
            "b079feff448e925546c4f23965b7dd40": {"scale": (2, 1.25, 1.5)},
            "71b87045c8fbce807c7680c7449f4f61": {"scale": None},  #
            # "6d5e521ebaba489bd9a347e62c5432e": {"scale": (2, 3, 2.5)}, # Too small ratio of cabinet space to height
            # '6ae80779dd34c194c664c3d4e2d59341', # Fails inside checking
            # '50fea70354d4d987d42b9650f19dd425', # Pointy top
            "3bff6c7c4ab1e47e2a9a1691b6f98331": {"scale": (1.5, 2, 2)},
            "38be55deba61bcfd808d50b6f71a45b": {"scale": (2, 1.5, 2)},
            "1170df5b9512c1d92f6bce2b7e6c12b7": {"scale": None},  #
        }
        category = "shelf"
        for model, info in shelves.items():
            if info["scale"] is None:
                scale = None
            else:
                scale = np.array(info["scale"])

            obj = import_object(
                self.simulator, igibson_category=category, model=model, scale=scale
            )
            self.all_objects.append(obj)
            self.support_surfaces[category][model] = obj

        # Import all tables
        category = "breakfast_table"
        tables = {
            "1b4e6f9dd22a8c628ef9d976af675b86": {"scale": (1, 1, 1.5)},
            "5f3f97d6854426cfb41eedea248a6d25": {"scale": (1, 1, 2)},
            "33e4866b6db3f49e6fe3612af521500": {"scale": (1, 1, 2)},
            # "72c8fb162c90a716dc6d75c6559b82a2": {} #Not under sampleable
            "242b7dde571b99bd3002761e7a3ba3bd": {"scale": (1, 1, 1)},
            # "26073": {"scale": None} # Not work
            "db665d85f1d9b1ea5c6a44a505804654": {"scale": (1, 1, 2)},
        }
        # tables = {}
        for model, info in tables.items():
            if info["scale"] is None:
                scale = None
            else:
                scale = np.array(info["scale"])
            obj = import_object(
                self.simulator,
                igibson_category=category,
                model=model,
                scale=scale,
            )
            self.all_objects.append(obj)
            self.support_surfaces[category][model] = obj
        for obj in self.all_objects:
            self.stash(obj)

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
            categorical_reward_info[
                reward_function.__class__.__name__
            ] = category_reward

        categorical_reward_info["total"] = reward
        info["reward_breakdown"] = categorical_reward_info

        return reward, info

    def reset_agent(self, env):
        env.robots[0].reset()
        env.land(env.robots[0], [0, 0, 0.0], [0, 0, 4.7])

    def stash(self, obj):
        stash_idx = self.stash_idx % 50
        obj.set_position((50 + stash_idx, 50, 0))
        self.stash_idx += 1
        bid = obj.get_body_ids()[obj.main_body]
        if bid not in self.mass_cache:
            self.mass_cache[bid] = p.getDynamicsInfo(bid, -1)[0]
        p.changeDynamics(obj.get_body_ids()[obj.main_body], -1, mass=0)

    def retrieve(self, obj):
        bid = obj.get_body_ids()[obj.main_body]
        p.changeDynamics(bid, -1, mass=self.mass_cache[bid])

    def fix(self, obj):
        pos, orn = obj.get_base_link_position_orientation()
        obj_base_id = obj.get_body_ids()[obj.main_body]
        constraint = p.createConstraint(
            parentBodyUniqueId=obj_base_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=pos,
            childFrameOrientation=orn,
        )

        self.constraints[obj_base_id] = constraint

    def release(self, obj):
        obj_base_id = obj.get_body_ids()[obj.main_body]
        if obj_base_id in self.constraints:
            p.removeConstraint(self.constraints[obj_base_id])
            del self.constraints[obj_base_id]

    def reset_scene(self, env):
        # This is absolutely critical, reset doors
        env.scene.reset_scene_objects()

        # Stash all objects off scene
        for obj in self.all_objects:
            # self.release(obj)
            self.stash(obj)

        # Choose an object type
        support_types = np.random.choice(list(self.support_surfaces.keys()), 2, replace=False)

        choice_objs = []
        # Retrieve and place support objs
        for support_type, offset in zip(support_types, (-1.25, 1.25)):
            support_model = np.random.choice(
                list(self.support_surfaces[support_type].keys())
            )
            obj = self.support_surfaces[support_type][support_model]
            self.retrieve(obj)
            obj.set_position_orientation(
                (offset, -1.5, 1.0),
                R.from_euler("xyz", angles=(0, 0, 180), degrees=True).as_quat(),
            )
            choice_objs.append(obj)

        choice_idx = np.random.choice([0, 1], 2, replace=False)
        self.target_obj = choice_objs[choice_idx[0]]
        self.distractor_obj = choice_objs[choice_idx[1]]

        if self.config.get("debug_task", True):
            print()
            print(
                f"Find: {support_types[choice_idx[0]]}"
            )
            print()

        self.task_obs = np.array([choice_idx[0]], dtype=np.float32)

    def get_task_obs(self, env):
        return self.task_obs.copy()
