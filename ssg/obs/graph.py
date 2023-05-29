import logging
from collections import defaultdict
from enum import Enum

import hydra
import networkx as nx
import numpy as np
import pybullet as p
import torch_geometric.utils as pyg_utils
from igibson.object_states import ContactBodies, Inside, OnTop, Under
from igibson.utils import semantics_utils
from igibson.utils.constants import SemanticClass
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R

import ssg
from ssg.utils.room_constants import TRAVERSABLE_MAP
from ssg.tasks.relational_search.relational_search_constants import UNIFIED_CATEGORICAL_ENCODING

tree = lambda: defaultdict(tree)


class Edge(Enum):
    onTop = "onTop"
    inside = "inside"
    under = "under"
    inHand = "inHand"
    inRoom = "inRoom"
    roomConnected = "roomConnected"


class Graph:
    def __init__(
        self,
        env,
        features=[
            "local_pos",
            "bbox",
            "cat",
            "semantic_class",
            # "is_agent",
            # "interactable",
            # "occupancy",
        ],
        edge_groups={
            "connected": [Edge.inRoom, Edge.roomConnected],
            "inside": [Edge.inside],
            "inHand": [Edge.inHand],
            "onTop": [Edge.onTop],
            "under": [Edge.under],
        },
        full_observability=False,
    ):
        self.env = env
        self.full_observability = full_observability
        self.edge_type_to_group = {}
        self.edge_groups = edge_groups
        for group_label, edge_types in edge_groups.items():
            for edge in edge_types:
                self.edge_type_to_group[Edge(edge)] = group_label

        self.node_types = {
            "object": 0,
            "place": 1,
            "room": 2,
            "action": 3,
            "scene": 4,
        }

        self.edges_to_track = set()
        for edge_type in edge_groups.values():
            self.edges_to_track.update(edge_type)
        self.edges_to_track = list(map(Edge, self.edges_to_track))

        self.features = features
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
            "semantic_class_categorical": UNIFIED_CATEGORICAL_ENCODING['other'].shape[0] ,  # Semantic class
            "interactable": 1,  # The robot has come within dt meters of the node. The node is an action node and lies in free space. Otherwise 0.
            "is_agent": 1,
            "occupancy": 1,  # ESDF value (voxel occupancy?):1 if the node is a place or action, 0 if it is an object, null if it is a room
        }

        self.node_dim = 0
        for feature in features:
            self.node_dim += self.feature_size_map[feature]

        # self.category_mapping = self.generate_mapping()
        self.category_mapping = self.consolidate_mapping()

        ins_map = env.simulator.scene.room_ins_map

        room_instances = np.unique(ins_map)
        self.room_instances = np.delete(room_instances, np.where(room_instances == 0))
        self.room_centroids = np.zeros((self.room_instances.shape[0], 2))
        self.room_instance_centroid_map = {}
        idx = 0
        for _, room_ins_id in env.simulator.scene.room_ins_name_to_ins_id.items():
            if room_ins_id == 0:
                continue
            x, y = np.where(env.simulator.scene.room_ins_map == room_ins_id)
            centroid = np.mean(np.vstack([x, y]).T, axis=0).astype(int)
            self.room_centroids[idx] = centroid
            self.room_instance_centroid_map[idx] = room_ins_id
            idx += 1

        self.room_ins_name_to_sem_name = {}
        for sem_name, instances in self.env.simulator.scene.room_sem_name_to_ins_name.items():
            for instance in instances:
                self.room_ins_name_to_sem_name[instance] = sem_name
        self.reset()

    def get_node_id(self):
        node_id_count = self.node_id_count
        self.node_id_count += 1
        return node_id_count

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
        self.G = nx.Graph()
        self.node_id_count = 0
        self.body_node_ids = {}
        self.room_node_ids = {}

    def add_room_instance(self, room_instance):
        room_type = self.room_ins_name_to_sem_name[room_instance]
        lower, upper = self.env.simulator.scene.get_aabb_by_room_instance(room_instance)
        node_id = self.get_node_id()
        self.room_node_ids[room_instance] = node_id
        self.G.add_node(
            node_id,
            bbox=((upper - lower)).astype(np.float32),
            cat=self.node_types["room"],
            cat_name=room_type,
            semantic_class=self.category_mapping[room_type],
            semantic_class_categorical=UNIFIED_CATEGORICAL_ENCODING.get(room_type, UNIFIED_CATEGORICAL_ENCODING["other"]),
            # interactable=
            # occupancy=
            is_agent=False,
            pos=((upper - lower) / 2 + lower).astype(np.float32),
            orn=np.array([0, 0, 0, 1], dtype=np.float32),
        )
        for instance, instance_node_id in self.room_node_ids.items():
            # Add all edges for object
            if Edge.roomConnected in self.edges_to_track:
                if instance in TRAVERSABLE_MAP[self.env.scene.scene_id][room_instance]:
                    self.G.add_edge(
                        node_id, instance_node_id, relation=Edge.roomConnected
                    )

    def get_adjacency_list(self, threshold=1.0):
        mapping = {}
        arr = np.zeros((len(self.body_node_ids), 3))
        for idx, body_id in enumerate(self.body_node_ids):
            obj = self.env.simulator.scene.objects_by_id[body_id]
            mapping[idx] = body_id
            arr[idx] = obj.get_position()
        distances = distance_matrix(arr, arr)
        xs, ys = np.where((distances < threshold) == True)
        adjacency_list = {mapping[x]: [] for x in xs}
        for x, y in zip(xs, ys):
            if x != y:
                adjacency_list[mapping[x]].append(mapping[y])
        return adjacency_list

    def get_room_instance_by_point(self, pos):
        xy = pos[:2]
        room_instance = self.env.simulator.scene.get_room_instance_by_point(xy)

        # Use centroid
        if room_instance is None:
            x, y = self.env.simulator.scene.world_to_seg_map(xy)
            nearest_room_centroid = np.argmin(
                np.linalg.norm(
                    np.subtract(self.room_centroids, np.array([x, y])), axis=1
                )
            )
            room_instance_id = self.room_instance_centroid_map[nearest_room_centroid]
            room_instance = self.env.simulator.scene.room_ins_id_to_ins_name[
                room_instance_id
            ]
        return room_instance

    def update_bids(self, bids):
        robot = self.env.robots[0]
        # Add or update all nodes in graph
        for bid in bids:
            if bid in self.env.simulator.scene.objects_by_id:
                obj = self.env.simulator.scene.objects_by_id[bid]
            else:
                logger = logging.getLogger("ssg")
                logger.warning(
                    f"[ssg] invalid pybullet id ({bid}) encountered from instance segmentation"
                )
                continue

            if obj.category in ["ceilings", "walls", "floors"]:
                continue

            # Use position/orientaiton of aligned bounding box
            pos, orn, bbox_extent, _ = obj.get_base_aligned_bounding_box(
                fallback_to_aabb=True
            )
            pos = np.array(pos, dtype=np.float32)
            orn = np.array(orn, dtype=np.float32)

            # Get room of object
            room_instance = self.get_room_instance_by_point(pos)

            # Add room to graph
            if room_instance not in self.room_node_ids:
                self.add_room_instance(room_instance)

            # Add or update node
            if bid not in self.body_node_ids:
                node_id = self.get_node_id()
                self.G.add_node(
                    node_id,
                    bbox=bbox_extent.astype(np.float32),
                    cat=self.node_types["object"],
                    semantic_class=self.category_mapping[obj.category],
                    semantic_class_categorical=UNIFIED_CATEGORICAL_ENCODING.get(obj.category, UNIFIED_CATEGORICAL_ENCODING["other"]),
                    cat_name=obj.category,
                    orn=orn,
                    room_instance=room_instance,
                    # interactable=
                    # occupancy=
                    is_agent=(obj.category == "agent"),
                    pos=pos,
                )
                self.body_node_ids[bid] = node_id
            else:
                node_id = self.body_node_ids[bid]
                self.G.nodes[node_id].update(
                    {
                        "pos": pos,
                        "orn": orn,
                        "room_instance": room_instance,
                    }
                )

        # Optimization for under
        close_objects = self.get_adjacency_list(threshold=1.0)
        # print(distance_mapping)
        # for bid in self.body_node_ids:
        #     obj = self.env.simulator.scene.objects_by_id[bid]
        #     print(obj.category, bid)

        # Add or update all edges in the graph. Should be done after updating the nodes
        # Remove all edges for the objects in view
        for bid, node_id in self.body_node_ids.items():
            for edge in list(self.G.edges(node_id)):
                self.G.remove_edge(*edge)

        # Add all edges for the objects in view
        for bid, node_id in self.body_node_ids.items():
            obj = self.env.simulator.scene.objects_by_id[bid]
            room_instance = self.G.nodes[node_id]["room_instance"]

            if Edge.inRoom in self.edges_to_track:
                self.G.add_edge(
                    node_id, self.room_node_ids[room_instance], relation=Edge.inRoom
                )

            if Edge.onTop in self.edges_to_track:
                contact_set = set()
                contact_bodies = obj.states[ContactBodies].get_value()
                for element in contact_bodies:
                    contact_set.add(element.bodyUniqueIdB)
                for contact_bid in contact_set:
                    contact = self.env.simulator.scene.objects_by_id[contact_bid]
                    if (
                        contact_bid in self.body_node_ids
                        and obj.states[OnTop].get_value(contact)
                        and not obj.states[Inside].get_value(contact)
                    ):
                        # print(obj.category, 'ontop', contact.category)
                        # print(self.G.edges)
                        self.G.add_edge(
                            node_id,
                            self.body_node_ids[contact_bid],
                            relation=Edge.onTop,
                        )
                        # print(self.G.edges)
                        # print()

            if Edge.under in self.edges_to_track:
                for relation_bid in self.body_node_ids.keys():
                    if relation_bid == bid or relation_bid not in close_objects[bid]:
                        continue
                    relation_obj = self.env.simulator.scene.objects_by_id[relation_bid]
                    if relation_bid in self.body_node_ids and obj.states[
                        Under
                    ].get_value(relation_obj):
                        # print(obj.category, 'under', relation_obj.category)
                        # print(self.G.edges)
                        self.G.add_edge(
                            node_id,
                            self.body_node_ids[relation_bid],
                            relation=Edge.under,
                        )
                        # print(self.G.edges)
                        # print()

            # To-do, this is kinda hacky
            if Edge.inside in self.edges_to_track:
                contact_set = set()
                contact_bodies = obj.states[ContactBodies].get_value()
                for element in contact_bodies:
                    contact_set.add(element.bodyUniqueIdB)
                for contact_bid in contact_set:
                    contact = self.env.simulator.scene.objects_by_id[contact_bid]
                    if contact_bid in self.body_node_ids and obj.states[
                        Inside
                    ].get_value(contact):
                        # print(obj.category, 'inside', contact.category)
                        # print(self.G.edges)
                        self.G.add_edge(
                            node_id,
                            self.body_node_ids[contact_bid],
                            relation=Edge.inside,
                        )
                        # print(self.G.edges)
                        # print()

            if Edge.inHand in self.edges_to_track and robot.inventory == obj:
                self.G.add_edge(
                    node_id, self.room_node_ids[room_instance], relation=Edge.inHand
                )

        # Update all local positions and orientations of the nodes
        egocentric_transform = p.invertTransform(*robot.get_position_orientation())

        for node, value in self.G.nodes.items():
            local_pos, local_orn = p.multiplyTransforms(
                *egocentric_transform, value["pos"], value["orn"]
            )
            self.G.nodes[node]["local_pos"] = np.array(local_pos, dtype=np.float32)
            self.G.nodes[node]["local_orn"] = local_orn
            self.G.nodes[node]["local_orn_rad"] = R.from_quat(local_orn).as_euler("xyz")

    def update(self, seg):
        robot = self.env.robots[0]

        # Get list of body ids to update
        bids = set()
        if self.full_observability:
            bids.update(list(self.env.simulator.scene.objects_by_id))
        else:
            body_ids = self.env.simulator.renderer.get_pb_ids_for_instance_ids(seg)
            bids_in_fov = set(np.unique(body_ids)) - {-1}
            agent_bid = robot.get_body_ids()[0]

            bids.update(bids_in_fov)
            bids.update([agent_bid])

        self.update_bids(bids)

    def to_pyg(self):
        return pyg_utils.from_networkx(self.G, group_node_attrs=self.features)

    def to_ray(self):
        edges = np.array(self.G.edges)
        nodes = np.zeros([len(self.G.nodes), self.node_dim], dtype=np.float32)
        for id in self.G.nodes:
            start = 0
            for feature in self.features:
                offset = self.feature_size_map[feature]
                nodes[id, start : start + offset] = self.G.nodes[id][feature]
                start += offset

        edge_groups = {group: [] for group in self.edge_groups}
        for (i, j, relation) in self.G.edges.data("relation"):  # type: ignore
            group = self.edge_type_to_group[relation]
            edge_groups[group].append((i, j))

        edges = {}
        for key in edge_groups:
            edges[key] = np.array(list(set(edge_groups[key])), dtype=np.int64)

        out = {"nodes": nodes}
        out.update(edges)
        return out


@hydra.main(config_path=ssg.CONFIG_PATH, config_name="config")
def main(cfg):
    import torch_geometric as pyg
    from omegaconf import OmegaConf

    from ssg.envs.igibson_env import iGibsonEnv
    from ssg.policies.soft_attention import SAM

    env_config = OmegaConf.to_object(cfg)

    env = iGibsonEnv(
        config_file=env_config,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 40.0,
    )

    env.reset()

    #
    gat = SAM(8, 2)

    env.reset()
    # scene_graph = Graph(env)
    scene_graph = Graph(env)

    # 10 seconds
    for _ in range(100):
        # action = np.array([1, 1])
        action = 1
        _, reward, _, _ = env.step(action)
        action = action
        reward = reward

        ins_seg = env.simulator.renderer.render_robot_cameras(modes=("ins_seg"))[0]
        scene_graph.update(ins_seg)
        ray_graph = scene_graph.to_ray()
        sg = scene_graph.to_pyg()
        batch = pyg.data.Batch.from_data_list([sg])
        out = gat(batch)


if __name__ == "__main__":
    main()
