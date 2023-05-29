import numpy as np


class RoomTracker:
    def __init__(self, scene):
        self.scene = scene
        ins_map = scene.room_ins_map

        room_instances = np.unique(ins_map)
        self.room_instances = np.delete(room_instances, np.where(room_instances == 0))
        self.room_centroids = np.zeros((self.room_instances.shape[0], 2))
        self.room_instance_centroid_map = {}
        idx = 0
        for _, room_ins_id in scene.room_ins_name_to_ins_id.items():
            if room_ins_id == 0:
                continue
            x, y = np.where(scene.room_ins_map == room_ins_id)
            centroid = np.mean(np.vstack([x, y]).T, axis=0).astype(int)
            self.room_centroids[idx] = centroid
            self.room_instance_centroid_map[idx] = room_ins_id
            idx += 1

        self.room_ins_name_to_sem_name = {}
        for (
            room_sem_cat,
            rooms,
        ) in scene.room_sem_name_to_ins_name.items():
            for room in rooms:
                self.room_ins_name_to_sem_name[room] = room_sem_cat

    def get_room_instance_by_point(self, pos):
        xy = pos[:2]
        room_instance = self.scene.get_room_instance_by_point(xy)

        # Use centroid
        if room_instance is None:
            x, y = self.scene.world_to_seg_map(xy)
            nearest_room_centroid = np.argmin(
                np.linalg.norm(
                    np.subtract(self.room_centroids, np.array([x, y])), axis=1
                )
            )
            room_instance_id = self.room_instance_centroid_map[nearest_room_centroid]
            room_instance = self.scene.room_ins_id_to_ins_name[room_instance_id]

        return room_instance

    def get_room_sem_by_point(self, point):
        room_instance = self.get_room_instance_by_point(point)
        return self.room_ins_name_to_sem_name[room_instance]
