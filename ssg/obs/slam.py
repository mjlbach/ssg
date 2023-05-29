import cv2
import numpy as np
from igibson.external.pybullet_tools.utils import matrix_from_quat
from igibson.utils import semantics_utils
from igibson.utils.constants import MAX_CLASS_COUNT, SemanticClass
from igibson.utils.utils import convertPointCoordSystem
from numba import njit, prange
from skimage.draw import disk
import os
import ssg


@njit(parallel=False)
def last_nonzero_numba(arr, value):
    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[1]):
            for z in range(arr.shape[2], 0, -1):
                if arr[x, y, z] != 0:
                    value[x, y] = arr[x, y, z]
                    break
    return value


class SimpleSlam:
    def __init__(
        self,
        object_categories=None,
        normalization_map=None,
        resolution=128,
        voxel_size=0.23,
        egocentric=False,
    ):
        self.resolution = resolution
        self.voxel_size = voxel_size
        self.midpoint = resolution // 2
        self.debug = False
        assert self.resolution % 2 == 0

        self.egocentric = egocentric
        if self.egocentric:
            y, x = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
            indices = np.vstack((x.ravel(), y.ravel())).T
            xy_world = (indices - self.midpoint) * self.voxel_size
            xy_world = np.c_[xy_world, np.zeros(xy_world.shape[0])]
            # self.voxel_world_coords = np.c_[xy_world, np.ones(xy_world.shape[0])]

            ogl_vox_centers = convertPointCoordSystem(
                xy_world, from_system="pybullet", to_system="opengl"
            )
            self.voxel_ogl_coords = np.c_[
                ogl_vox_centers, np.ones(ogl_vox_centers.shape[0])
            ]

        cat_map = semantics_utils.CLASS_NAME_TO_CLASS_ID
        cat_map["room_floor"] = cat_map["floors"]
        cat_map["agent"] = int(SemanticClass.ROBOTS)
        self.cat_map = cat_map

        # if object_categories is not None:
        #     remapped_ids = {}
        #     idx = 1
        #     for key in object_categories:
        #         remapped_ids[cat_map[key]] = idx
        #         idx += 1
        #     remapped_ids[0] = 0
        #
        #     normalization_map = remapped_ids
        #
        # if normalization_map:
        #     # Extract out keys and values
        #     k = np.array(list(normalization_map.keys()))
        #     v = np.array(list(normalization_map.values()), dtype=np.float32)
        #
        #     # Get argsort indices
        #     sidx = k.argsort()
        #
        #     self.ks = k[sidx]
        #     self.vs = v[sidx]
        #
        #     self.max_class = np.max(list(normalization_map.values()))
        #     self.normalization_map = normalization_map
        # else:
        self.normalization_map = None

        self.reset()

    @staticmethod
    def last_nonzero(arr, axis, invalid_val=0):
        mask = arr != 0
        val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    def reset(self):
        self.sem_map = np.zeros([self.resolution] * 3, dtype=np.float32)

    def update(self, state, extrinsic):
        sem = state["seg"][:, :, 0].reshape(-1)
        points = state["pc"].reshape(-1, 3)
        # Add a column of ones for the inverse transform
        points = np.c_[points, np.ones(points.shape[0])]

        # Get map indices
        world_points = np.linalg.inv(extrinsic).dot(points.T).T

        out = (world_points * 1 / self.voxel_size).astype(int) + self.midpoint
        out = np.clip(out, 0, self.resolution - 1)

        ### Replace non-visible agent (0.25) with floor (0.75)
        self.sem_map[self.sem_map == self.cat_map["agent"]] = self.cat_map["floors"]
        ## TODO: HACK remove hardback from semantic map
        self.sem_map[self.sem_map == self.cat_map["notebook"]] = self.cat_map["floors"]
        self.sem_map[out[:, 0], out[:, 1], out[:, 2]] = sem

        ### Clip out ceiling
        sem_map = self.sem_map[..., (self.midpoint - 10) : (self.midpoint + 10)]

        value = np.zeros((sem_map.shape[0], sem_map.shape[1], 1), dtype=np.float32)
        sem_map = last_nonzero_numba(sem_map, value)

        if self.normalization_map:
            sem_map = self.vs[np.searchsorted(self.ks, sem_map)]
            sem_map = sem_map / self.max_class
        else:
            sem_map = sem_map / MAX_CLASS_COUNT

        if self.egocentric:
            # Convert coordinates for each voxel in opengl coordinate frame for world to local ogl coordinates
            # Take a grid around the agent, with the number of cells equal to the resolution * resolution
            # of the metric map, and convert each of those 3d grid points to the world frame
            world_points = np.linalg.inv(extrinsic).dot(self.voxel_ogl_coords.T).T

            # Convert global pybullet coordinates to global voxel indices
            voxel_idx = np.divide(world_points, self.voxel_size) + self.midpoint
            voxel_idx = np.clip(voxel_idx, 0, self.resolution - 1)
            voxel_idx = np.round(voxel_idx).astype(int)

            # Debug, print the corners:
            if self.debug:
                debug = voxel_idx.reshape(128, 128, -1)
                # import ipdb; ipdb.set_trace()
                print()
                for x in [0, 127]:
                    for y in [0, 127]:
                        print(f"Remapping {x},{y} to {debug[x, y]}")
                print()

            # Index the semantic map using the voxels
            sem_map = sem_map[voxel_idx[..., 0], voxel_idx[..., 1]]

            # Reshape into image
            sem_map = sem_map.reshape(self.resolution, self.resolution, 1)

        self.last_2d_map = sem_map
        return sem_map

    def dump(self):
        map_path = os.path.join(ssg.ROOT_PATH, "assets/wainscott_0_int_sem_map.npz")
        self.sem_map.dump(map_path)

    def load(self):
        map_path = os.path.join(ssg.ROOT_PATH, "assets/wainscott_0_int_sem_map.npz")
        self.sem_map = np.load(map_path, allow_pickle=True)


class VisitedSlam:
    def __init__(self, resolution=50, voxel_size=0.18, egocentric=False):
        self.resolution = resolution
        assert self.resolution % 2 == 0

        self.voxel_size = voxel_size
        self.midpoint = resolution // 2
        self.debug = False

        self.egocentric = egocentric
        if self.egocentric:
            y, x = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
            indices = np.vstack((x.ravel(), y.ravel())).T
            xy_world = (indices - self.midpoint) * self.voxel_size
            xy_world = np.c_[xy_world, np.zeros(xy_world.shape[0])]
            # self.voxel_world_coords = np.c_[xy_world, np.ones(xy_world.shape[0])]

            ogl_vox_centers = convertPointCoordSystem(
                xy_world, from_system="pybullet", to_system="opengl"
            )
            self.voxel_ogl_coords = np.c_[
                ogl_vox_centers, np.ones(ogl_vox_centers.shape[0])
            ]

        self.reset()

    def reset(self):
        self.visited_map = np.zeros(
            (self.resolution, self.resolution, 1), dtype=np.float32
        )

    def update(self, pos, orn, extrinsic=None):
        # Convert prior agent positions to "past" visitation
        mask = self.visited_map != 0
        self.visited_map[mask] = 0.5

        # Convert agent position to voxel position centered at numpy array midpoint
        if self.debug:
            for x in [20, 108]:
                for y in [20, 108]:
                    rr, cc = disk((x, y), 3)
                    self.visited_map[rr, cc] = 2

        if self.egocentric:
            # Use camera matrix for extracting position
            # Cannot use this directly
            agent_voxel_idx = np.divide(pos, self.voxel_size) + self.midpoint
            agent_voxel_idx = np.clip(agent_voxel_idx, 0, self.resolution - 1)
            agent_voxel_idx = np.round(agent_voxel_idx).astype(int)

            # Draw a disk at the voxel idx derived from world coordinates
            rr, cc = disk((agent_voxel_idx[0], agent_voxel_idx[1]), 3)
            rr = np.clip(rr, 0, self.resolution - 1)
            cc = np.clip(cc, 0, self.resolution - 1)
            self.visited_map[rr, cc] = 1

            # Convert coordinates for each voxel in opengl coordinate frame for world to local ogl coordinates
            # Take a grid around the agent, with the number of cells equal to the resolution * resolution
            # of the metric map, and convert each of those 3d grid points to the world frame
            world_points = np.linalg.inv(extrinsic).dot(self.voxel_ogl_coords.T).T

            # Convert global pybullet coordinates to global voxel indices
            voxel_idx = np.divide(world_points, self.voxel_size) + self.midpoint
            voxel_idx = np.clip(voxel_idx, 0, self.resolution - 1)
            voxel_idx = np.round(voxel_idx).astype(int)

            # Debug, print the corners:
            if self.debug:
                debug = voxel_idx.reshape(128, 128, -1)
                # import ipdb; ipdb.set_trace()
                print()
                for x in [0, 127]:
                    for y in [0, 127]:
                        print(f"Remapping {x},{y} to {debug[x, y]}")
                print()

            # Index the semantic map using the voxels
            visited_map = self.visited_map[voxel_idx[..., 0], voxel_idx[..., 1]]

            # Reshape into image
            visited_map = visited_map.reshape(self.resolution, self.resolution, 1)

        else:
            voxel_idx = (pos * 1 / self.voxel_size).astype(int) + self.midpoint
            voxel_idx = np.clip(voxel_idx, 0, self.resolution - 1)
            orn_vec = np.dot(matrix_from_quat(orn), np.array([5, 0, 0]))
            orn_vec = (voxel_idx + orn_vec).astype(int).tolist()
            self.visited_map = self.visited_map.swapaxes(0, 1).copy()
            self.visited_map = cv2.arrowedLine(
                self.visited_map,
                voxel_idx[:2].tolist(),
                orn_vec[:2],
                color=1,
                tipLength=1,
            )
            self.visited_map = self.visited_map.swapaxes(0, 1).copy()
            visited_map = self.visited_map

        return visited_map


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    from igibson.utils.utils import parse_config

    import ssg
    from ssg.envs.igibson_env import iGibsonEnv

    config_file = os.path.join(ssg.CONFIG_PATH, "config.yaml")
    config = parse_config(config_file)
    config["output"] = ["rgb", "ego_sem_map", "depth", "agent_path"]
    config["use_egocentric_projection"] = True

    env = iGibsonEnv(
        config_file=config,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 40.0,
    )

    env.reset()
    # plt.ion()

    # 10 seconds
    for i in range(20):
        action = np.array(2)
        state, reward, done, info = env.step(action)
        _, axs = plt.subplots(2, 2, tight_layout=True, dpi=170)

        if "rgb" in state:
            axs[0, 0].imshow(state["rgb"])
            axs[0, 0].set_title("rgb")
            axs[0, 0].set_axis_off()

        if "ego_sem_map" in state:
            axs[0, 1].imshow(state["ego_sem_map"])
            axs[0, 1].set_title("Inferred semantic map")
            axs[0, 1].set_axis_off()

        if "agent_path" in state:
            axs[1, 1].imshow(state["agent_path"])
            axs[1, 1].set_title("Agent history")
            axs[1, 1].set_axis_off()

        if "depth" in state:
            axs[1, 0].imshow(state["depth"])
            axs[1, 0].set_title("depth")
            axs[1, 0].set_axis_off()

        plt.show()
