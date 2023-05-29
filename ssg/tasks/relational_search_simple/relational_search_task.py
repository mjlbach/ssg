import json
import os
from collections import defaultdict
from random import sample

import numpy as np
import pybullet as p
import ssg
from igibson.object_states import Under
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.timeout import Timeout
from ssg.tasks.relational_search_simple.relational_search_constants import (
    OBJ_SAMPLING_STATES,
    OBJ_STATES,
    RELATIONAL_STATES,
    SEARCH_CATEGORY_STATES,
    SEARCH_OBJECTS,
    UNIFIED_CATEGORICAL_ENCODING,
)
from ssg.utils.floor_sampler import sample_on_floor
from ssg.utils.object_utils import ObjectManager, import_object
from ssg.utils.other_utils import convert_to_onehot, retry
from ssg.utils.room_assignment import RoomTracker
from ssg.utils.room_constants import EXTERIOR_DOORS, ROOMS, TRAVERSABLE_MAP

from .relational_search_reward import (
    ObservationReward,
    PotentialReward,
    SearchReward,
    VisitationReward,
)
from .relational_search_termination import FallTermination, SearchTermination

tree = lambda: defaultdict(tree)


class RelationalSimpleSearchTask(BaseTask):
    def __init__(self, env):
        super().__init__(env)
        self.room_tracker = RoomTracker(env.scene)

        self.scene = env.scene
        self.simulator = env.simulator

        self.reward_functions = []
        if "search" in self.config["reward_functions"]:
            self.reward_functions.append(SearchReward(self.config))
        if "visitation" in self.config["reward_functions"]:
            self.reward_functions.append(VisitationReward(self.config))
        if "potential" in self.config["reward_functions"]:
            self.reward_functions.append(PotentialReward(self.config))
        if "observation" in self.config["reward_functions"]:
            self.reward_functions.append(ObservationReward(self.config))

        self.termination_conditions = [
            Timeout(self.config),
            SearchTermination(self.config),
            FallTermination(self.config),
        ]

        search_objects = SEARCH_OBJECTS

        # For building goal descriptor
        self.search_categories = list(search_objects)
        self.relational_states = list(RELATIONAL_STATES.values())
        self.obj_states = list(OBJ_STATES.values())
        self.associated_categories = list(OBJ_SAMPLING_STATES)
        self.rooms = ROOMS

        # self.obj_state_to_onehot = convert_to_onehot(self.obj_states)
        self.relational_states_to_onehot = convert_to_onehot(self.relational_states)

        if self.config["use_onehot_categorical"]:
            self.search_obj_to_onehot = UNIFIED_CATEGORICAL_ENCODING
            self.associated_obj_to_onehot = UNIFIED_CATEGORICAL_ENCODING
            self.room_to_onehot = UNIFIED_CATEGORICAL_ENCODING
            self.task_obs_dim = (
                len(UNIFIED_CATEGORICAL_ENCODING)
                + len(self.relational_states)
                # + len(self.obj_states)
                + len(UNIFIED_CATEGORICAL_ENCODING)
                + len(UNIFIED_CATEGORICAL_ENCODING)
            )
        else:
            self.search_obj_to_onehot = convert_to_onehot(self.search_categories)
            self.associated_obj_to_onehot = convert_to_onehot(
                self.associated_categories
            )
            self.room_to_onehot = convert_to_onehot(self.rooms)
            self.task_obs_dim = (
                len(self.search_categories)
                + len(self.relational_states)
                # + len(self.obj_states)
                + len(self.associated_categories)
                + len(self.rooms)
            )

        self.task_raw_category_obs_dim = 3

        # Goal descriptor
        # Apple onTop Table in Kitchen

        # Restrict to states that are possible for each associated object type
        self.obj_sampling_states = OBJ_SAMPLING_STATES

        self.object_manager = ObjectManager()
        self.sampling_dict = tree()

        # Important to do this before befor fixing furniture
        self.remove_exterior_doors()

        # Important to do this before importing other objects as these constraints are not tracked!
        self.fix_furniture()

        # Load cache dictionary.
        self.cache_dic = None
        if self.config.get("use_cache", True):
            with open(os.path.join(ssg.ROOT_PATH, "assets/cache_dump.json") , "r") as f:
                cache_dic = json.load(f)
                scene_id = self.simulator.scene.scene_id
                assert (
                    scene_id in cache_dic
                ), f"Cache dictionary does not contain information about {scene_id}"
                self.cache_dic = cache_dic[scene_id]
        else:
            print("use_cache = False. reset_scene may be slow.")

        # Import all search objects
        self.search_objs = tree()
        for category, models in search_objects.items():
            for model in models:
                obj = import_object(
                    self.simulator,
                    igibson_category=category,
                    model=model,
                )
                self.search_objs[category][model] = obj
                self.target_obj = obj  # this is important
        for category in self.search_objs:
            for model in self.search_objs[category]:
                self.object_manager.stash(self.search_objs[category][model])

        # Import all distraction objects
        if self.config.get("add_distraction", False):
            self.distract_objs = tree()
            for category, models in search_objects.items():
                for model in models:
                    obj = import_object(
                        self.simulator,
                        igibson_category=category,
                        model=model,
                    )
                    self.distract_objs[category][model] = obj
                    self.distract_obj = obj  # this is important
            for category in self.distract_objs:
                for model in self.distract_objs[category]:
                    self.object_manager.stash(self.distract_objs[category][model])

    def fix_furniture(self):
        for obj in self.simulator.scene.objects_by_id.values():  # type: ignore
            if obj.category not in ["floors", "ceilings", "agent"]:
                pos, orn = obj.get_position_orientation()
                obj_base_id = obj.get_body_ids()[obj.main_body]
                p.createConstraint(
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

    def remove_exterior_doors(self):
        objs_to_remove = []
        for obj in self.simulator.scene.objects_by_category["door"]:
            if obj.name not in EXTERIOR_DOORS[self.simulator.scene.scene_id]:
                objs_to_remove.append(obj)
        for obj in objs_to_remove:
            self.object_manager.stash(obj)

    def set_random_object_state(self, obj):
        """
        Randomly set an object's state (e.g. Normal, Cooked, Soaked, Stained).

        :param obj: object to set set
        :return obj_state: the state to set
        """
        obj_state = None
        # Early return if we do not randomize object states.
        if not self.config.get("randomize_object_states", False):
            return obj_state
        available_object_states = SEARCH_CATEGORY_STATES[obj.category]
        for _ in range(10):
            rand_idx = np.random.randint(len(available_object_states) + 1)
            if rand_idx < len(available_object_states):
                obj_state = available_object_states[rand_idx]
                try:  # sometimes set_value fails
                    assert obj.states[obj_state].set_value(True)
                    break
                except:
                    print(f"Setting {obj.name} to state {obj_state.__name__} failed.")
            else:
                break
        return obj_state

    def set_relational_state(self, obj, associated_obj, relational_state):
        # Hack for sampling under
        if relational_state == Under:
            pos = associated_obj.get_position()
            pos[2] = obj.bounding_box[2] / 2
            obj.set_position(pos)
            success = obj.states[relational_state].get_value(associated_obj)
        else:
            success = obj.states[relational_state].set_value(associated_obj, True, True)
        return success

    def get_reward(self, env, collision_links=[], action=None, info={}):  # type: ignore
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
            # Only keep traversable rooms.
            available_rooms = {
                k for k, v in TRAVERSABLE_MAP[self.scene.scene_id].items() if v
            }
            agent_room_type = self.room_tracker.get_room_instance_by_point(
                self.target_obj.get_position()
            )
            # Do not spawn agent in the room with target object.
            room = np.random.choice(np.array(list(available_rooms - {agent_room_type})))
            sample_on_floor(env.robots[0], env.simulator.scene, room)
        else:
            env.land(env.robots[0], [0.0, 0.0, 0.0], [0, 0, 0])

    @retry(times=50)
    def reset_scene(self, env):
        # Remove the constraint from the last episode
        self.object_manager.release(self.target_obj)
        self.object_manager.stash(self.target_obj)
        if self.config.get("add_distraction", False):
            self.object_manager.release(self.distract_obj)
            self.object_manager.stash(self.distract_obj)

        # This is absolutely critical, reset doors DO NOT DO THIS IF REMOVING INTERIOR DOORS
        # env.scene.reset_scene_objects()
        # self.remove_exterior_doors()

        # The variables below form the goal descriptor for the task.
        target_obj = None
        associated_obj = None
        relational_state = None
        object_state = None
        room_type = None

        # The variables below are used for distractor object but do not go into the goal descriptor.
        distract_obj = None
        distract_associated_obj = None
        distract_relational_state = None

        if self.config.get("use_cache", True):
            # Load variables from cache.
            search_category = np.random.choice(list(SEARCH_OBJECTS))
            num_samples = len(self.cache_dic[search_category])
            rand_idx = np.random.randint(num_samples)
            (
                search_model,
                relational_state_str,
                associated_name,
                target_pos,
                target_orn,
                room_type,
            ) = self.cache_dic[search_category][rand_idx]
            target_obj = self.search_objs[search_category][search_model]
            self.target_obj = target_obj
            relational_state = RELATIONAL_STATES[relational_state_str]
            associated_obj = env.scene.objects_by_name[associated_name]
            # Place target object.
            self.object_manager.unstash(target_obj)
            target_obj.set_position_orientation(target_pos, target_orn)
            self.simulator.step()
            self.object_manager.fix(target_obj)
            # Set object state.
            object_state = self.set_random_object_state(target_obj)
            object_state_str = "Normal" if not object_state else object_state.__name__
            print(
                f"Find: {target_obj.category} ({object_state_str}) {relational_state.__name__} {associated_obj.category} in {room_type}"
            )

            # Add distraction.
            if self.config.get("add_distraction", False):
                rand_idx_arr = sample(range(num_samples), num_samples)
                for rand_idx in rand_idx_arr:
                    (
                        distract_search_model,
                        distract_relational_state_str,
                        distract_associated_name,
                        distract_target_pos,
                        distract_target_orn,
                        distract_room_type,
                    ) = self.cache_dic[search_category][rand_idx]
                    distract_obj = self.distract_objs[search_category][
                        distract_search_model
                    ]
                    # Distraction should not have associated object category, relational state and room type all the same.
                    if (
                        distract_relational_state_str == relational_state_str
                        and distract_obj.category == target_obj.category
                        and distract_room_type == room_type
                    ):
                        continue
                    self.distract_obj = distract_obj
                    distract_relational_state = RELATIONAL_STATES[
                        distract_relational_state_str
                    ]
                    distract_associated_obj = env.scene.objects_by_name[
                        distract_associated_name
                    ]
                    # Do not associate distraction with the same object.
                    if distract_associated_obj == associated_obj:
                        continue
                    # Place target object.
                    self.object_manager.unstash(distract_obj)
                    distract_obj.set_position_orientation(
                        distract_target_pos, distract_target_orn
                    )
                    self.simulator.step()
                    self.object_manager.fix(distract_obj)
                    # Set object state.
                    distract_object_state = self.set_random_object_state(distract_obj)
                    distract_object_state_str = (
                        "Normal"
                        if not distract_object_state
                        else distract_object_state.__name__
                    )
                    print(
                        f"Distraction: {distract_obj.category} ({distract_object_state_str}) {distract_relational_state.__name__} {distract_associated_obj.category} in {distract_room_type}"
                    )
                    break

        else:
            search_category = np.random.choice(self.search_categories)
            search_model = np.random.choice(list(self.search_objs[search_category]))

            target_obj = self.search_objs[search_category][search_model]
            self.target_obj = target_obj
            self.object_manager.unstash(target_obj)

            if self.config.get("add_distraction", False):
                distract_obj = self.distract_objs[search_category][search_model]
                self.distract_obj = distract_obj
                self.object_manager.unstash(distract_obj)

            # Iterate over all support objects
            success = False
            target_model_path = target_obj.model_path.split("/")
            target_cat = target_model_path[-2]
            target_model = target_model_path[-1]
            available_associated_categories = sample(
                self.associated_categories, len(self.associated_categories)
            )
            for ac in available_associated_categories:
                if success:
                    break
                cat_objs = env.scene.objects_by_category[ac]
                available_associated_objs = sample(cat_objs, len(cat_objs))
                available_relational_states = sample(
                    self.obj_sampling_states[ac], len(self.obj_sampling_states[ac])
                )
                for ao in available_associated_objs:
                    if success:
                        break
                    associated_obj = ao
                    associated_model_path = ao.model_path.split("/")
                    asso_cat = associated_model_path[-2]
                    asso_model = associated_model_path[-1]
                    for rs in available_relational_states:
                        if success:
                            break
                        relational_state = rs
                        # Don't try to sample a combination that previously failed
                        if (
                            self.sampling_dict[target_cat][target_model][
                                relational_state.__name__
                            ][asso_cat][asso_model]
                            == False
                        ):
                            continue
                        # Place target object.
                        success = self.set_relational_state(
                            target_obj, associated_obj, relational_state
                        )
                        if success:
                            self.sampling_dict[target_cat][target_model][
                                relational_state.__name__
                            ][asso_cat][asso_model] = True
                            self.object_manager.fix(target_obj)
                            room_type = self.room_tracker.get_room_sem_by_point(
                                target_obj.get_position()
                            )
                            object_state = self.set_random_object_state(target_obj)
                            object_state_str = (
                                "Normal" if not object_state else object_state.__name__
                            )
                            print(
                                f"Find: {target_cat} ({object_state_str}) {relational_state.__name__} {asso_cat} in {room_type}"
                            )
                        else:
                            self.sampling_dict[target_cat][target_model][
                                relational_state.__name__
                            ][asso_cat][asso_model] = False
                            print(
                                f"Failed to sample: {target_cat} {target_model} {relational_state.__name__} {asso_cat} {asso_model}"
                            )

            if self.config.get("add_distraction", False):
                distract_model_path = distract_obj.model_path.split("/")
                distract_cat = distract_model_path[-2]
                distract_model = distract_model_path[-1]
                distract_available_associated_categories = sample(
                    self.associated_categories, len(self.associated_categories)
                )
                distract_success = False
                for dac in distract_available_associated_categories:
                    if distract_success:
                        break
                    distract_cat_objs = env.scene.objects_by_category[dac]
                    available_distract_associated_objs = sample(
                        distract_cat_objs, len(distract_cat_objs)
                    )
                    for dao in available_distract_associated_objs:
                        if distract_success:
                            break
                        # Do not associate distraction with the same object.
                        if dao == ao:
                            continue
                        distract_associated_obj = dao
                        distract_associated_model_path = dao.model_path.split("/")
                        distract_asso_cat = distract_associated_model_path[-2]
                        distract_asso_model = distract_associated_model_path[-1]
                        available_distract_relational_states = sample(
                            self.obj_sampling_states[dac],
                            len(self.obj_sampling_states[dac]),
                        )
                        for drs in available_distract_relational_states:
                            if distract_success:
                                break
                            distract_relational_state = drs
                            # Place distract object.
                            distract_success = self.set_relational_state(
                                distract_obj,
                                distract_associated_obj,
                                distract_relational_state,
                            )
                            if distract_success:
                                self.sampling_dict[distract_cat][distract_model][
                                    distract_relational_state.__name__
                                ][distract_asso_cat][distract_asso_model] = True
                                distract_room_type = (
                                    self.room_tracker.get_room_sem_by_point(
                                        distract_obj.get_position()
                                    )
                                )
                                # Distraction should not have associated object category, relational state and room type all the same.
                                if (
                                    dac == ac
                                    and drs == rs
                                    and room_type == distract_room_type
                                ):
                                    distract_success = False
                                    continue
                                self.object_manager.fix(distract_obj)
                                distract_object_state = self.set_random_object_state(
                                    distract_obj
                                )
                                distract_object_state_str = (
                                    "Normal"
                                    if not distract_object_state
                                    else distract_object_state.__name__
                                )
                                print(
                                    f"Distraction: {distract_cat} ({distract_object_state_str}) {distract_relational_state.__name__} {distract_asso_cat} in {distract_room_type}"
                                )
                            else:
                                self.sampling_dict[distract_cat][distract_model][
                                    distract_relational_state.__name__
                                ][asso_cat][asso_model] = False
                                print(
                                    f"Failed to sample: {distract_cat} {distract_model} {distract_relational_state.__name__} {distract_asso_cat} {distract_asso_model}"
                                )

        # Be extra cautious and verify that the variables are set correctly.
        assert target_obj
        assert associated_obj
        assert relational_state
        assert room_type
        assert relational_state in target_obj.states
        assert target_obj.states[relational_state].get_value(associated_obj)

        self.task_raw_category_obs = (
            target_obj.category,
            associated_obj.category,
            room_type,
        )

        self.task_obs = np.concatenate(
            (
                self.search_obj_to_onehot[target_obj.category],
                self.relational_states_to_onehot[relational_state],
                # self.obj_state_to_onehot[object_state],
                self.associated_obj_to_onehot[associated_obj.category],
                self.room_to_onehot[room_type],
            ),
            dtype=np.float32,
        )

    def get_task_obs(self, _):
        return self.task_obs.copy()

    def get_task_raw_category_obs(self, _):
        return self.task_raw_category_obs
