import pybullet as p
import numpy as np
from bddl.object_taxonomy import ObjectTaxonomy
from igibson.objects.articulated_object import URDFObject
from igibson.utils.assets_utils import (
    get_ig_avg_category_specs,
    get_ig_category_path,
    get_ig_model_path,
)
from ssg.tasks.relational_search.relational_search_constants import (
    SEARCH_CATEGORY_STATES,
)

object_taxonomy = ObjectTaxonomy()
avg_category_spec = get_ig_avg_category_specs()

import os


def import_object(
    simulator,
    wordnet_category=None,
    igibson_category=None,
    model=None,
    scale=None,
):
    if wordnet_category:
        categories = object_taxonomy.get_subtree_igibson_categories(wordnet_category)
        igibson_category = np.random.choice(categories)
    else:
        assert igibson_category is not None
    if model:
        pass
    else:
        category_path = get_ig_category_path(igibson_category)
        model_choices = os.listdir(category_path)

        # Randomly select an object model
        model = np.random.choice(model_choices)

    model_path = get_ig_model_path(igibson_category, model)
    filename = os.path.join(model_path, model + ".urdf")
    num_new_obj = len(simulator.scene.objects_by_name)
    obj_name = "{}_{}".format(igibson_category, num_new_obj)

    if scale is None:
        fit_avg_dim_volume = True
    else:
        fit_avg_dim_volume = False

    # create the object and set the initial position to be far-away
    simulator_obj = URDFObject(
        filename,
        name=obj_name,
        category=igibson_category,
        scale=scale,
        model_path=model_path,
        avg_obj_dims=avg_category_spec.get(igibson_category),
        fit_avg_dim_volume=fit_avg_dim_volume,
        texture_randomization=False,
        overwrite_inertial=True,
    )

    # # Load the object into the simulator
    simulator.import_object(simulator_obj)

    # task
    return simulator_obj


class ObjectManager:
    def __init__(self):
        self.constraints = {}
        self.stash_idx = 0
        self.stashed_objects = set()
        
    def force_sleep(self, obj, body_id=None):
        if body_id is None:
            body_id = obj.get_body_ids()[0]

        activationState = p.ACTIVATION_STATE_SLEEP + p.ACTIVATION_STATE_DISABLE_WAKEUP
        p.changeDynamics(body_id, -1, activationState=activationState)


    def stash(self, obj):
        # Reset object state.
        if obj.category in SEARCH_CATEGORY_STATES:
            available_object_states = SEARCH_CATEGORY_STATES[obj.category]
            for obj_state in available_object_states:
                obj.states[obj_state].set_value(False)

        stash_idx = self.stash_idx % 50
        obj.set_position((50 + stash_idx, 50, 0))
        self.force_sleep(obj)
        for instance in obj.renderer_instances:
            instance.hidden = True
        self.stash_idx += 1
        self.stashed_objects.add(obj.name)


    def unstash(self, obj):
        obj.force_wakeup()
        for instance in obj.renderer_instances:
            instance.hidden = False
        if obj.name in self.stashed_objects:
            self.stashed_objects.remove(obj.name)


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


    def unstash_and_place(self, obj, pos_orn):
        self.unstash(obj)
        obj.set_position_orientation(*pos_orn)
        #self.fix(obj) # Do not add constraints!