import logging

import hydra
import numpy as np
import pybullet as p
from igibson.utils import semantics_utils
from igibson.utils.constants import SemanticClass
from scipy.spatial.transform import Rotation as R

import ssg


class ObjectSet:
    def __init__(
        self,
        env,
        features=[
            "local_pos",
            "local_orn_rad",
            "bbox",
            "cat",
            "semantic_class",
            # "is_agent",
            # "interactable",
            # "occupancy",
        ],
        full_observability=False,
    ):
        self.env = env
        self.features = features
        self.full_observability = full_observability

        self.feature_size_map = {
            "pos": 3,  # local position
            "local_pos": 3,  # local position
            "orn": 4,  # local position
            "local_orn": 4,  # local position
            "orn_rad": 3,  # local position
            "local_orn_rad": 3,  # local position
            "bbox": 3,  # x, y, z extent, null if not object
            "cat": 1,  # object, place, room, action
            "semantic_class": 1,  # Semantic class
            "interactable": 1,  # The robot has come within dt meters of the node. The node is an action node and lies in free space. Otherwise 0.
            "is_agent": 1,
            "occupancy": 1,  # ESDF value (voxel occupancy?):1 if the node is a place or action, 0 if it is an object, null if it is a room
        }

        self.obj_types = {
            "object": 0,
            "place": 1,
            "room": 2,
            "action": 3,
            "scene": 4,
        }

        self.obj_dim = 0
        for feature in features:
            self.obj_dim += self.feature_size_map[feature]

        # self.category_mapping = self.generate_mapping()
        self.category_mapping = self.consolidate_mapping()

        self.reset()

    def generate_mapping(self):
        cat_map = semantics_utils.CLASS_NAME_TO_CLASS_ID
        cat_map["room_floor"] = semantics_utils.CLASS_NAME_TO_CLASS_ID["floors"]
        cat_map["agent"] = int(SemanticClass.ROBOTS)
        cat_map["scene"] = 0

        cat_max = max(cat_map.values())
        room_map = semantics_utils.get_room_name_to_room_id(cat_max + 1)
        cat_map.update(room_map)
        return cat_map

    def consolidate_mapping(self):
        rooms = list(self.env.simulator.scene.room_sem_name_to_sem_id)
        items = list(self.env.simulator.scene.objects_by_category)
        category_mapping = {}
        all_categories = ["scene"] + rooms + items
        for idx, item in enumerate(all_categories):
            category_mapping[item] = idx / len(all_categories)

        return category_mapping

    def reset(self):
        self.objects = {}

    def to_numpy(self):
        arr = np.zeros((len(self.objects), self.obj_dim), dtype=np.float32)
        idx = 0
        for obj in self.objects.values():
            offset = 0
            for feature in self.features:
                arr[idx, offset : offset + self.feature_size_map[feature]] = obj[
                    feature
                ]
                offset += self.feature_size_map[feature]
            idx += 1

        return arr

    def update(self, seg):
        bids = set()

        if self.full_observability:
            bids.update(list(self.env.simulator.scene.objects_by_id))
        else:
            body_ids = self.env.simulator.renderer.get_pb_ids_for_instance_ids(seg)
            bids_in_fov = set(np.unique(body_ids)) - {-1}
            agent_bid = self.env.simulator.scene.objects_by_category["agent"][
                0
            ].get_body_ids()[0]

            bids.update(bids_in_fov)
            bids.update([agent_bid])

        for bid in bids:
            if bid in self.env.simulator.scene.objects_by_id:
                obj = self.env.simulator.scene.objects_by_id[bid]
            else:
                logger = logging.getLogger("ssg")
                logger.warning(
                    "invalid pybullet id encountered from instance segmentation"
                )
                continue

            pos, orn, bbox_extent, _ = obj.get_base_aligned_bounding_box(
                fallback_to_aabb=True
            )
            pos = pos.astype(np.float32)
            orn = orn.astype(np.float32)

            # Update attributes
            if bid in self.objects:
                self.objects[bid].update(
                    {
                        "pos": pos,
                        "orn": orn,
                    }
                )
            else:
                if obj.category == "agent":
                    is_agent = True
                    bbox_extent = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                else:
                    is_agent = False

                self.objects[bid] = {
                    "bbox": bbox_extent.astype(np.float32),
                    "pos": pos,
                    "orn": orn,
                    "cat": self.obj_types["object"],
                    "cat_name": obj.category,
                    "semantic_class": self.category_mapping[obj.category],
                    "is_agent": is_agent,
                }

        egocentric_transform = p.invertTransform(
            *self.env.robots[0].get_position_orientation()
        )

        for bid, value in self.objects.items():
            local_pos, local_orn = p.multiplyTransforms(
                *egocentric_transform, value["pos"], value["orn"]
            )
            local_pos = np.array(local_pos, dtype=np.float32)
            self.objects[bid]["local_pos"] = local_pos
            self.objects[bid]["local_orn"] = local_orn
            self.objects[bid]["local_orn_rad"] = R.from_quat(local_orn).as_euler("xyz")


@hydra.main(config_path=ssg.CONFIG_PATH, config_name="config")
def main(cfg):
    from omegaconf import OmegaConf

    from ssg.envs.igibson_env import iGibsonEnv

    env_config = OmegaConf.to_object(cfg)

    env = iGibsonEnv(
        config_file=env_config,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 40.0,
    )

    env.reset()
    # scene_graph = Graph(env)
    obj_set = ObjectSet(env)

    # 10 seconds
    for i in range(100):
        # action = np.array([1, 1])
        action = 1
        state, reward, done, info = env.step(action)
        obs = state
        action = action
        reward = reward

        ins_seg = env.simulator.renderer.render_robot_cameras(modes=("ins_seg"))[0]
        obj_set.update(ins_seg)
        obj_set.to_numpy()


if __name__ == "__main__":
    main()
