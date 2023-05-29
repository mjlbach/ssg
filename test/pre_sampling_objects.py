from collections import defaultdict
from random import sample, seed

import argparse
import json
import os
import numpy as np
from igibson.object_states import Inside, OnTop, Under
from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.utils import NumpyEncoder

from ssg.tasks.relational_search.relational_search_constants import (
    SEARCH_OBJECTS,
    OBJ_SAMPLING_STATES,
)
from ssg.utils.floor_sampler import sample_on_floor
from ssg.utils.object_utils import import_object, ObjectManager
from ssg.utils.room_assignment import RoomTracker

tree = lambda: defaultdict(tree)

DUMP_OUTPUT_DIR = "./out"


def dump_sample_dict(object_manager, target_obj):
    object_manager.unstash(target_obj)
    target_model = target_obj.model_path.split("/")

    pre_sampling_dict = tree()
    for ac, available_states in OBJ_SAMPLING_STATES.items():
        available_associated_objs = scene.objects_by_category[ac]
        print("=" * 100)
        print(ac)
        print([ao.name for ao in available_associated_objs])
        for ao in available_associated_objs:
            associated_obj = ao
            for relational_state in available_states:
                # Define the target dictionary.
                target_dict = pre_sampling_dict[scene.scene_id][target_model[-2]][
                    target_model[-1]
                ][relational_state.__name__][ao.name]
                # Don't try to sample a combination that previously failed.
                if target_dict["is_success"] == False:
                    continue

                # Hack for sampling under
                if relational_state == Under:
                    pos = ao.get_position()
                    pos[2] = target_obj.bounding_box[2] / 2  # put on floor
                    target_obj.set_position(pos)
                    success = target_obj.states[relational_state].get_value(
                        associated_obj
                    )
                else:
                    success = target_obj.states[relational_state].set_value(
                        ao, True, True
                    )
                if success:
                    target_dict["is_success"] = True
                    room_type = room_tracker.get_room_sem_by_point(
                        target_obj.get_position()
                    )
                    target_dict["room"] = room_type

                    assert target_obj
                    assert associated_obj
                    assert relational_state
                    assert room_type

                    (
                        target_dict["pos"],
                        target_dict["orn"],
                    ) = target_obj.get_position_orientation()
                    print(
                        f"SUCCESS: {target_model[-2]} {target_model[-1]} {relational_state.__name__} {ao.name} in {room_type}",
                    )
                else:
                    target_dict["is_success"] = False
                    print(
                        f"Failed to sample: {target_model[-2]} {target_model[-1]} {relational_state.__name__} {ao.name}"
                    )

    object_manager.stash(target_obj)
    return pre_sampling_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="Rs_int")
    args = parser.parse_args()

    SCENE_ID = args.scene
    print("Working on scene: ", SCENE_ID)

    # Set seed.
    seed(42)
    np.random.seed(42)

    # Import scene.
    sim = Simulator(mode="headless", image_height=512, image_width=512)
    scene = InteractiveIndoorScene(
        scene_id=SCENE_ID,
        not_load_object_categories=[
            "ceilings",
            "carpet",
            "coffee_maker",
            "potted_plant",
            "sofa",
            "straight_chair",
        ],
    )
    sim.import_scene(scene)

    # Define variables.
    search_categories = list(SEARCH_OBJECTS)
    room_tracker = RoomTracker(scene)
    object_manager = ObjectManager()

    # Import all search objects and stash them.
    search_objs = tree()
    for category, models in SEARCH_OBJECTS.items():
        for model in models:
            obj = import_object(
                sim,
                igibson_category=category,
                model=model,
            )
            search_objs[category][model] = obj
    for category in search_objs:
        for model in search_objs[category]:
            object_manager.stash(search_objs[category][model])

    dump_dic = {}
    skip_existing = True
    for category in search_objs:
        for model in search_objs[category]:
            target_obj = search_objs[category][model]
            target_model = target_obj.model_path.split("/")
            dump_file_path = os.path.join(
                DUMP_OUTPUT_DIR,
                f"{SCENE_ID}-{target_model[-2]}-{target_model[-1]}.json",
            )
            if skip_existing and os.path.isfile(dump_file_path):
                print(f"Skipping: {dump_file_path}")
                continue
            dump_dic = dump_sample_dict(object_manager, target_obj)
            with open(
                dump_file_path,
                "w",
            ) as f:
                json.dump(dict(dump_dic), f, cls=NumpyEncoder)

    # Combine all dumped json files.
    print()
    print("Combining dump files...")
    data = tree()
    for filename in os.listdir(DUMP_OUTPUT_DIR):
        if "-" not in filename:
            continue
        print(filename)
        scene_id, cat, name = filename.split("-", 2)
        name = name.split(".")[0]
        with open(os.path.join(DUMP_OUTPUT_DIR, filename), "r") as f:
            tmp = json.load(f)
        data[scene_id][cat][name] = tmp[scene_id][cat][name]

    flattened_cache = {}
    for scene, scene_dic in data.items():
        if scene not in flattened_cache:
            flattened_cache[scene] = {}
        for cat, cat_dic in scene_dic.items():
            for name, name_dic in cat_dic.items():
                for state, state_dic in name_dic.items():
                    for as_name, dic in state_dic.items():
                        if dic["is_success"]:
                            if cat not in flattened_cache[scene]:
                                flattened_cache[scene][cat] = []
                            flattened_cache[scene][cat].append(
                                (
                                    name,
                                    state,
                                    as_name,
                                    dic["pos"],
                                    dic["orn"],
                                    dic["room"],
                                )
                            )
    with open(os.path.join(DUMP_OUTPUT_DIR, "cache_dump.json"), "w") as f:
        json.dump(flattened_cache, f)
