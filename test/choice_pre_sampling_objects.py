from collections import defaultdict
from random import sample, seed

import json
import os
import numpy as np
import pybullet as p
from igibson.object_states import Under, OnTop, Inside
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.assets_utils import get_ig_category_path
from scipy.spatial.transform import Rotation as R
from igibson.utils.utils import NumpyEncoder

from ssg.tasks.choice.choice_constants import (
    EXTRA_OBJECTS,
    NUM_EXTRA_OBJECTS_EACH,
    NUM_EXTRA_OBJECTS_TOTAL,
    OBJECTS_INFO,
)
from ssg.utils.object_utils import import_object, ObjectManager
from ssg.utils.other_utils import retry

tree = lambda: defaultdict(tree)

SCENE_ID = "Rs_int"
DUMP_OUTPUT_DIR = "./out"

TOTAL_SAMPLES = 10000  # Number of samples in the final dump file
GROUP_SAMPLES = 100    # Number of samples in each intermediate dump file


@retry(times=20)
def reset_and_get_dic():
    pre_sampling_dict = {}

    # This is absolutely critical, reset doors
    simulator.scene.reset_scene_objects()

    # Stash all objects off scene
    for obj in all_objects:
        if obj.name in object_manager.stashed_objects:
            continue
        object_manager.release(obj)
        object_manager.stash(obj)

    # Choose support objects and candidate objects
    support_type = np.random.choice(list(support_surfaces.keys()))
    support_model = np.random.choice(list(support_surfaces[support_type].keys()))
    support_objs = list(support_surfaces[support_type][support_model].values())

    candidate_type = np.random.choice(list(choice_objects.keys()))
    candidate_model = np.random.choice(list(choice_objects[candidate_type].keys()))
    candidate_objs = list(choice_objects[candidate_type][candidate_model].values())

    pre_sampling_dict["support_type"] = support_type
    pre_sampling_dict["support_model"] = support_model
    pre_sampling_dict["candidate_type"] = candidate_type
    pre_sampling_dict["candidate_model"] = candidate_model

    # Place support objects
    for obj, offset in zip(support_objs, (-1.25, 1.25)):
        object_manager.unstash(obj)
        obj.set_position_orientation(
            (offset, -1.5, 1.0),
            R.from_euler("xyz", angles=(0, 0, 180), degrees=True).as_quat(),
        )
        # Allow objects to settle so we can sample on them
        for _ in range(5):
            simulator.step()

    # Choose supported states for each support obj
    states = np.random.choice(allowed_states[support_type], 2, replace=False)

    # Sample choice objects
    for candidate_obj, support_obj, state in zip(candidate_objs, support_objs, states):
        object_manager.unstash(candidate_obj)
        # Hack for sampling under
        if state == Under:
            pos = support_obj.get_position()
            pos[2] = candidate_obj.bounding_box[2] / 2
            candidate_obj.set_position(pos)
            for _ in range(5):
                simulator.step()
            success = candidate_obj.states[state].get_value(support_obj)
        else:
            success = candidate_obj.states[state].set_value(
                support_obj, True, use_ray_casting_method=True
            )
            for _ in range(5):
                simulator.step()
            success &= candidate_obj.states[state].get_value(support_obj)

    obj_idxs = np.random.choice(np.arange(len(candidate_objs)), 2, replace=False)
    target_obj_idx = obj_idxs[0]
    distractor_obj_idx = obj_idxs[1]
    target_obj = candidate_objs[target_obj_idx]
    distractor_obj = candidate_objs[distractor_obj_idx]

    # Sample extra objects
    available_extra_objects = sample(extra_objects, len(extra_objects))
    idx = 0
    for extra_obj in available_extra_objects:
        if idx >= NUM_EXTRA_OBJECTS_TOTAL:
            break
        object_manager.unstash(extra_obj)
        # Pick which support object we will put on
        rand_idx = np.random.randint(2)
        support_obj = support_objs[rand_idx]
        # Choose supported state for this support obj
        state = np.random.choice(allowed_states[support_type])
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
        else:
            object_manager.stash(extra_obj)

    # Take another 100 steps and verify that the relational states are still satisfied
    for _ in range(100):
        simulator.step()

    # Fix all objects
    for obj in all_objects:
        if obj.name in object_manager.stashed_objects:
            continue
        object_manager.fix(obj)
        # Save object state
        pre_sampling_dict[obj.name] = obj.get_position_orientation()

    assert target_obj.states[states[target_obj_idx]].get_value(
        support_objs[target_obj_idx]
    )
    assert distractor_obj.states[states[distractor_obj_idx]].get_value(
        support_objs[distractor_obj_idx]
    )
    pre_sampling_dict["target_obj_name"] = target_obj.name
    pre_sampling_dict["target_state"] = states[target_obj_idx].__name__
    pre_sampling_dict["target_support_name"] = support_objs[target_obj_idx].name
    pre_sampling_dict["distractor_obj_name"] = distractor_obj.name
    pre_sampling_dict["distractor_state"] = states[distractor_obj_idx].__name__
    pre_sampling_dict["distractor_support_name"] = support_objs[distractor_obj_idx].name

    print(
        f"Find: {target_obj.category} {states[target_obj_idx].__name__} {support_type}"
    )
    print(
        f"Distraction: {distractor_obj.category} {states[distractor_obj_idx].__name__} {support_type}"
    )
    print("=" * 50)
    return pre_sampling_dict


if __name__ == "__main__":
    # Set seed.
    seed(42)
    np.random.seed(42)

    # Import scene.
    headless = True
    simulator = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_height=512,
        image_width=512,
    )
    scene = InteractiveIndoorScene(
        scene_id=SCENE_ID,
        load_object_categories=["agent", "floor"],
    )
    simulator.import_scene(scene)

    if not headless:
        simulator.viewer.initial_pos = [0.0, 0.7, 1.3]
        simulator.viewer.initial_view_direction = [0.0, -1.0, -0.2]
        simulator.viewer.reset_viewer()

    # Define variables.
    support_surfaces = tree()
    choice_objects = tree()
    extra_support_surfaces = tree()
    extra_objects = []
    all_objects = []
    object_manager = ObjectManager()

    allowed_states = {
        "breakfast_table": [OnTop, Under],
        "shelf": [OnTop, Inside],
    }

    # NOTE: this is important!
    # In the choice task, there's an agent loaded. Here we load in
    # a dummy object so that the name of all other objects match
    # the choice task exactly.
    dummy_obj = import_object(simulator, igibson_category="paper_towel", model="33_0")
    object_manager.stash(dummy_obj)

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
                simulator, igibson_category=category, model=model, scale=scale
            )
            all_objects.append(obj)
            support_surfaces[category][model][idx] = obj

        # Import all tables
        category = "breakfast_table"
        for model, info in OBJECTS_INFO[category].items():
            if info["scale"] is None:
                scale = None
            else:
                scale = np.array(info["scale"])
            obj = import_object(
                simulator,
                igibson_category=category,
                model=model,
                scale=scale,
            )
            all_objects.append(obj)
            support_surfaces[category][model][idx] = obj

        # Import all apples
        category = "apple"
        for model, info in OBJECTS_INFO[category].items():
            obj = import_object(
                simulator,
                igibson_category=category,
                model=model,
                scale=info["scale"],
            )
            bid = obj.get_body_ids()[obj.main_body]
            p.changeDynamics(bid, -1, rollingFriction=100)
            all_objects.append(obj)
            choice_objects[category][model][idx] = obj

        # Import all bowls
        category = "bowl"
        for model, info in OBJECTS_INFO[category].items():
            obj = import_object(
                simulator,
                igibson_category=category,
                model=model,
                scale=info["scale"],
            )
            all_objects.append(obj)
            choice_objects[category][model][idx] = obj

        # Import the rest
        for category in ["gym_shoe"]:
            for model in os.listdir(get_ig_category_path(category)):
                obj = import_object(simulator, igibson_category=category, model=model)
                all_objects.append(obj)
                choice_objects[category][model][idx] = obj

    # Import extra objects
    for category in EXTRA_OBJECTS:
        available_models = (
            EXTRA_OBJECTS[category]
            if EXTRA_OBJECTS[category]
            else os.listdir(get_ig_category_path(category))
        )
        for model in available_models:
            for _ in range(NUM_EXTRA_OBJECTS_EACH):
                obj = import_object(simulator, igibson_category=category, model=model)
                all_objects.append(obj)
                extra_objects.append(obj)

    # Stash all objects off scene
    print("Object list:")
    for obj in all_objects:
        object_manager.stash(obj)
        print(obj.name)
    print("=" * 50)

    skip_existing = True
    num_groups = (TOTAL_SAMPLES + GROUP_SAMPLES - 1) // GROUP_SAMPLES
    for group_idx in range(num_groups):
        print(f"Processing group {group_idx+1} / {num_groups}")
        dump_file_path = os.path.join(
            DUMP_OUTPUT_DIR,
            f"tmp_dump_{group_idx}.json",
        )
        if skip_existing and os.path.isfile(dump_file_path):
            print(f"Skipping: {dump_file_path}")
            continue
        group_dump_dic = {}
        for idx in range(GROUP_SAMPLES):
            group_dump_dic[idx] = reset_and_get_dic()
        with open(dump_file_path, "w") as f:
            json.dump(dict(group_dump_dic), f, cls=NumpyEncoder)

    # Combine all dumped json files.
    print()
    print("Combining dump files...")
    dump_dic = {}
    idx = 0
    for filename in os.listdir(DUMP_OUTPUT_DIR):
        if "tmp_dump_" not in filename:
            continue
        print(filename)
        with open(os.path.join(DUMP_OUTPUT_DIR, filename), "r") as f:
            tmp_dic = json.load(f)
        for d in tmp_dic.values():
            dump_dic[idx] = d
            idx += 1

    with open(os.path.join(DUMP_OUTPUT_DIR, "choice_cache_dump.json"), "w") as f:
        json.dump(dump_dic, f)
