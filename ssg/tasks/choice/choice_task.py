import os
from collections import defaultdict
from random import sample

import ssg
import json
import numpy as np
import pybullet as p
from igibson.object_states import Inside, OnTop, Under
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.assets_utils import get_ig_category_path
from scipy.spatial.transform import Rotation as R
from ssg.tasks.choice.choice_constants import (
    EXTRA_OBJECTS,
    NUM_EXTRA_OBJECTS_EACH,
    NUM_EXTRA_OBJECTS_TOTAL,
    OBJECTS_INFO,
)
from ssg.tasks.relational_search.relational_search_constants import (
    UNIFIED_CATEGORICAL_ENCODING,
    RELATIONAL_STATES,
)
from ssg.utils.object_utils import ObjectManager, import_object
from ssg.utils.other_utils import convert_to_onehot, retry

from .choice_reward import ChoiceReward, FailurePenalty, PotentialReward, TimePenalty
from .choice_termination import ChoiceTermination, FailureTermination

tree = lambda: defaultdict(tree)


class ChoiceTask(BaseTask):
    def __init__(self, env):
        super(ChoiceTask, self).__init__(env)

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

        self.mass_cache = {}
        self.object_manager = ObjectManager()
        self.initialize()

        self.choice_categories = ["apple", "gym_shoe", "bowl"]
        self.relation_categories = [OnTop, Inside, Under]
        self.associated_categories = ["breakfast_table", "shelf"]
        self.relation_to_onehot = convert_to_onehot(self.relation_categories)

        if self.config["use_onehot_categorical"]:
            self.choice_to_onehot = UNIFIED_CATEGORICAL_ENCODING
            self.associated_to_onehot = UNIFIED_CATEGORICAL_ENCODING
            self.task_obs_dim = len(UNIFIED_CATEGORICAL_ENCODING) * 2 + len(
                self.relation_categories
            )
        else:
            self.choice_to_onehot = convert_to_onehot(self.choice_categories)
            self.associated_to_onehot = convert_to_onehot(self.associated_categories)
            self.task_obs_dim = 8

    def initialize(self):
        # obj_pro = self.import_object(wordnet_category = 'microwave.n.02' , model='7128')
        self.allowed_states = {
            "breakfast_table": [OnTop, Under],
            "shelf": [OnTop, Inside],
        }
        self.support_surfaces = tree()
        self.choice_objects = tree()
        self.extra_support_surfaces = tree()
        self.extra_objects = []
        self.all_objects = []

        # Load cache dictionary.
        self.cache_dic = None
        if self.config.get("use_cache", True):
            with open(
                os.path.join(ssg.ROOT_PATH, "assets/choice_cache_dump.json"), "r"
            ) as f:
                self.cache_dic = json.load(f)
        else:
            print("use_cache = False. reset_scene may be slow.")

        # Two of choice objects and associated objects for symmetry
        for idx in range(2):
            # Import all shelves
            category = "shelf"
            for model, info in OBJECTS_INFO[category].items():
                if info["scale"] is None:
                    scale = None
                else:
                    scale = np.array(info["scale"])

                obj = import_object(
                    self.simulator, igibson_category=category, model=model, scale=scale
                )
                self.all_objects.append(obj)
                self.support_surfaces[category][model][idx] = obj

            # Import all tables
            category = "breakfast_table"
            for model, info in OBJECTS_INFO[category].items():
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
                self.support_surfaces[category][model][idx] = obj

            # Import all apples
            category = "apple"
            for model, info in OBJECTS_INFO[category].items():
                obj = import_object(
                    self.simulator,
                    igibson_category=category,
                    model=model,
                    scale=info["scale"],
                )
                bid = obj.get_body_ids()[obj.main_body]
                p.changeDynamics(bid, -1, rollingFriction=100)
                self.all_objects.append(obj)
                self.choice_objects[category][model][idx] = obj

            # Import all bowls
            category = "bowl"
            for model, info in OBJECTS_INFO[category].items():
                obj = import_object(
                    self.simulator,
                    igibson_category=category,
                    model=model,
                    scale=info["scale"],
                )
                self.all_objects.append(obj)
                self.choice_objects[category][model][idx] = obj

            # Import the rest
            for category in ["gym_shoe"]:
                for model in os.listdir(get_ig_category_path(category)):
                    obj = import_object(
                        self.simulator, igibson_category=category, model=model
                    )
                    self.all_objects.append(obj)
                    self.choice_objects[category][model][idx] = obj

        # Import extra objects
        if self.config.get("add_extra_objects", False):
            for category in EXTRA_OBJECTS:
                available_models = (
                    EXTRA_OBJECTS[category]
                    if EXTRA_OBJECTS[category]
                    else os.listdir(get_ig_category_path(category))
                )
                for model in available_models:
                    for _ in range(NUM_EXTRA_OBJECTS_EACH):
                        obj = import_object(
                            self.simulator, igibson_category=category, model=model
                        )
                        self.all_objects.append(obj)
                        self.extra_objects.append(obj)

        # Stash all objects off scene
        for obj in self.all_objects:
            self.object_manager.stash(obj)

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

    @retry(times=10)
    def reset_scene(self, env):
        # This is absolutely critical, reset doors
        env.scene.reset_scene_objects()

        # Stash all objects off scene
        for obj in self.all_objects:
            if obj.name in self.object_manager.stashed_objects:
                continue
            self.object_manager.release(obj)
            self.object_manager.stash(obj)

        if self.config.get("use_cache", True):
            num_samples = len(self.cache_dic)
            rand_idx = np.random.randint(num_samples)
            pre_sampling_dict = self.cache_dic[str(rand_idx)]

            # Load variables.
            support_type = pre_sampling_dict["support_type"]
            support_model = pre_sampling_dict["support_model"]
            candidate_type = pre_sampling_dict["candidate_type"]
            candidate_model = pre_sampling_dict["candidate_model"]
            target_obj_name = pre_sampling_dict["target_obj_name"]
            distractor_obj_name = pre_sampling_dict["distractor_obj_name"]
            target_support_name = pre_sampling_dict["target_support_name"]
            distractor_support_name = pre_sampling_dict["distractor_support_name"]

            self.target_obj = env.scene.objects_by_name[target_obj_name]
            self.distractor_obj = env.scene.objects_by_name[distractor_obj_name]
            target_support_obj = env.scene.objects_by_name[target_support_name]
            distractor_support_obj = env.scene.objects_by_name[distractor_support_name]
            target_state = RELATIONAL_STATES[pre_sampling_dict["target_state"]]
            distractor_state = RELATIONAL_STATES[pre_sampling_dict["distractor_state"]]

            # Place support objects
            self.object_manager.unstash_and_place(
                target_support_obj, pre_sampling_dict[target_support_name]
            )
            self.object_manager.unstash_and_place(
                distractor_support_obj, pre_sampling_dict[distractor_support_name]
            )

            # Place choice objects
            self.object_manager.unstash_and_place(
                self.target_obj, pre_sampling_dict[target_obj_name]
            )
            self.object_manager.unstash_and_place(
                self.distractor_obj, pre_sampling_dict[distractor_obj_name]
            )

            # Place extra objects
            for extra_obj in self.extra_objects:
                if extra_obj.name not in pre_sampling_dict:
                    continue
                self.object_manager.unstash_and_place(
                    extra_obj, pre_sampling_dict[extra_obj.name]
                )

        else:
            # Choose support objects and candidate objects
            support_type = np.random.choice(list(self.support_surfaces.keys()))
            support_model = np.random.choice(
                list(self.support_surfaces[support_type].keys())
            )
            support_objs = list(
                self.support_surfaces[support_type][support_model].values()
            )

            candidate_type = np.random.choice(list(self.choice_objects.keys()))
            candidate_model = np.random.choice(
                list(self.choice_objects[candidate_type].keys())
            )
            candidate_objs = list(
                self.choice_objects[candidate_type][candidate_model].values()
            )

            # Place support objects
            for obj, offset in zip(support_objs, (-1.25, 1.25)):
                self.object_manager.unstash(obj)
                obj.set_position_orientation(
                    (offset, -1.5, 1.0),
                    R.from_euler("xyz", angles=(0, 0, 180), degrees=True).as_quat(),
                )
                # Allow objects to settle so we can sample on them
                for _ in range(5):
                    self.simulator.step()

            # Choose supported states for each support obj
            states = np.random.choice(
                self.allowed_states[support_type], 2, replace=False
            )

            # Sample choice objects
            for candidate_obj, support_obj, state in zip(
                candidate_objs, support_objs, states
            ):
                self.object_manager.unstash(candidate_obj)
                # Hack for sampling under
                if state == Under:
                    pos = support_obj.get_position()
                    pos[2] = candidate_obj.bounding_box[2] / 2
                    candidate_obj.set_position(pos)
                    for _ in range(5):
                        self.simulator.step()
                    success = candidate_obj.states[state].get_value(support_obj)
                else:
                    success = candidate_obj.states[state].set_value(
                        support_obj, True, use_ray_casting_method=True
                    )
                    self.simulator.step()
                    success &= candidate_obj.states[state].get_value(support_obj)
                    if not success:
                        print("Failed sampling!")
                        print(
                            candidate_type,
                            candidate_model,
                            state,
                            support_type,
                            support_model,
                        )

            obj_idxs = np.random.choice(
                np.arange(len(candidate_objs)), 2, replace=False
            )
            target_obj_idx = obj_idxs[0]
            distractor_obj_idx = obj_idxs[1]
            self.target_obj = candidate_objs[target_obj_idx]
            self.distractor_obj = candidate_objs[distractor_obj_idx]
            target_state = states[target_obj_idx]
            distractor_state = states[distractor_obj_idx]
            target_support_obj = support_objs[target_obj_idx]
            distractor_support_obj = support_objs[distractor_obj_idx]

            # Sample extra objects
            if self.config.get("add_extra_objects", False):
                available_extra_objects = sample(
                    self.extra_objects, len(self.extra_objects)
                )
                idx = 0
                for extra_obj in available_extra_objects:
                    if idx >= NUM_EXTRA_OBJECTS_TOTAL:
                        break
                    self.object_manager.unstash(extra_obj)
                    # Pick which support object we will put on
                    rand_idx = np.random.randint(2)
                    support_obj = support_objs[rand_idx]
                    # Choose supported state for this support obj
                    state = np.random.choice(self.allowed_states[support_type])
                    # Hack for sampling under
                    if state == Under:
                        pos = support_obj.get_position()
                        pos[2] = extra_obj.bounding_box[2] / 2
                        extra_obj.set_position(pos)
                        success = extra_obj.states[state].get_value(support_obj)
                    else:
                        success = extra_obj.states[state].set_value(
                            support_obj, True, use_ray_casting_method=True
                        )
                    if success:
                        idx += 1
                        if self.config.get("debug_task", False):
                            print(
                                f"Extra: {extra_obj.category} {state.__name__} {support_type}"
                            )
                    else:
                        self.object_manager.stash(extra_obj)
                        if self.config.get("debug_task", False):
                            print("Failed sampling extra object!")
                            print(
                                extra_obj.category,
                                state,
                                support_type,
                                support_model,
                            )

        # Take another few steps and verify that the relational states are still satisfied
        for _ in range(5):
            self.simulator.step()

        # Fix all objects
        for obj in self.all_objects:
            if obj.name in self.object_manager.stashed_objects:
                continue
            self.object_manager.fix(obj)

        assert self.target_obj.states[target_state].get_value(target_support_obj)
        assert self.distractor_obj.states[distractor_state].get_value(
            distractor_support_obj
        )

        if self.config.get("debug_task", False):
            print()
            print(
                f"Find: {self.target_obj.category} {target_state.__name__} {support_type}"
            )
            print(
                f"Distraction: {self.distractor_obj.category} {distractor_state.__name__} {support_type}"
            )
            print()

        self.task_obs = np.concatenate(
            (
                self.choice_to_onehot[candidate_type],
                self.relation_to_onehot[target_state],
                self.associated_to_onehot[support_type],
            ),
            dtype=np.float32,
        )

    def get_task_obs(self, env):
        return self.task_obs.copy()
