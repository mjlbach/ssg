import numpy as np
import open3d
import open3d as o3d
import open3d.core as o3c

# Fixing the frustum clearing issue

#
# We don't have this functionality supported, but since you are using the python API I guess you can do this manually:
#
# * you first create a frustum volume in the camera coordinate system by unprojecting the imaging plane to a list of discretized depths and reshape it to (N, 3) (only once)
# *then you rigid transform the volume to the world coordinate system (once per iteration)
# * finally you turn it into a Open3D Tensor and feed it to the function in need.


class Slam:
    def __init__(
        self,
        intrinsic,
        features=["color", "seg", "ins_seg"],
        feature_sizes=[(3), (1), (1)],
        device="CUDA:0",
    ):
        self.device = o3c.Device(device)
        voxel_size = 3.0 / 512

        self.features = features
        self.feature_sizes = feature_sizes

        attr_names = ["tsdf", "weight"] + features
        attr_sizes = [(1), (1)] + feature_sizes
        attr_dtypes = [o3c.float32] * len(attr_sizes)

        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=attr_names,
            attr_dtypes=attr_dtypes,
            attr_channels=attr_sizes,
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=self.device,
        )
        self.trunc = voxel_size * 4
        self.depth_max = 5.0
        self.depth_scale = 1.0
        self.intrinsic = o3c.Tensor(intrinsic, o3c.Dtype.Float64)
        self.intrinsic_dev = o3c.Tensor(
            intrinsic, device=self.device, dtype=o3c.float32
        )


    @staticmethod
    def convert_opengl_to_world(ex):
        return np.array(
            [
                [ex[0, 0], ex[0, 1], ex[0, 2], ex[0, 3]],
                [-ex[1, 0], -ex[1, 1], -ex[1, 2], -ex[1, 3]],
                [-ex[2, 0], -ex[2, 1], -ex[2, 2], -ex[2, 3]],
                [0, 0, 0, 1],
            ]
        )

    def integrate(self, extrinsic, depth, features):
        extrinsic = self.convert_opengl_to_world(extrinsic)
        extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float64)
        depth = open3d.t.geometry.Image(np.ascontiguousarray(depth))
        depth = depth.to(self.device)

        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, self.intrinsic, extrinsic, self.depth_scale, self.depth_max
        )
        # Activate them in the underlying hash map (may have been inserted)
        self.vbg.hashmap().activate(frustum_block_coords)

        # Find buf indices in the underlying engine
        buf_indices, _ = self.vbg.hashmap().find(frustum_block_coords)
        o3c.cuda.synchronize()

        voxel_coords, voxel_indices = self.vbg.voxel_coordinates_and_flattened_indices(
            buf_indices
        )
        self.voxel_coords = voxel_coords
        o3c.cuda.synchronize()

        # Now project them to the depth and find association
        # (3, N) -> (2, N)
        extrinsic_dev = extrinsic.to(self.device, o3c.float32)
        xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3, 3:]

        uvd = self.intrinsic_dev @ xyz
        d = uvd[2]
        u = (uvd[0] / d).round().to(o3c.int64)
        v = (uvd[1] / d).round().to(o3c.int64)
        o3c.cuda.synchronize()

        mask_proj = (
            (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (v < depth.rows)
        )

        # For visualization
        self.columns = depth.columns
        self.rows = depth.rows
        self.extrinsic = extrinsic
        self.frustum_block_coords = frustum_block_coords

        v_proj = v[mask_proj]
        u_proj = u[mask_proj]
        d_proj = d[mask_proj]
        depth_readings = (
            depth.as_tensor()[v_proj, u_proj, 0].to(o3c.float32) / self.depth_scale
        )
        sdf = depth_readings - d_proj

        mask_inlier = (
            (depth_readings > 0)
            & (depth_readings < self.depth_max)
            & (sdf >= -self.trunc)
        )

        sdf[sdf >= self.trunc] = self.trunc
        sdf = sdf / self.trunc
        o3c.cuda.synchronize()

        weight = self.vbg.attribute("weight").reshape((-1, 1))
        tsdf = self.vbg.attribute("tsdf").reshape((-1, 1))

        valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
        w = weight[valid_voxel_indices]
        wp = w + 1

        tsdf[valid_voxel_indices] = (
            tsdf[valid_voxel_indices] * w + sdf[mask_inlier].reshape(w.shape)
        ) / (wp)

        ### Add for loop here to iterate over properties
        for idx, feat in enumerate(self.features):
            feat_tensor = o3c.Tensor(
                features[feat], dtype=o3c.float32, device=self.device
            )

            if feat == "color":
                ### Open3D normalize color, need to rescale between 0-255
                feat_tensor *= 255

            feat_tensor_readings = feat_tensor[v_proj, u_proj]

            # Dictionary keys must be in same order as feature sizes
            shape = self.feature_sizes[idx]
            feat_tensor = self.vbg.attribute(feat).reshape((-1, shape))
            feat_tensor[valid_voxel_indices] = feat_tensor_readings[mask_inlier]

        weight[valid_voxel_indices] = wp
        o3c.cuda.synchronize()
        return self.vbg

    def visualize(self, extrinsic=None):
        if extrinsic is None:
            extrinsic = self.extrinsic

        # examples/python/t_reconstruction_system/ray_casting.py
        result = self.vbg.ray_cast(
            block_coords=self.vbg.hashmap().key_tensor(),
            intrinsic=self.intrinsic,
            extrinsic=extrinsic,
            width=self.columns,
            height=self.rows,
            render_attributes=["depth", "normal", "color", "index", "interp_ratio"],
            depth_scale=self.depth_scale,
            depth_min=0,
            depth_max=10,
            weight_threshold=1,
        )

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2)
        # colorized depth
        colorized_depth = o3d.t.geometry.image(result["depth"]).colorize_depth(
            self.depth_scale, 0, 3
        )
        axs[0, 0].imshow(colorized_depth.as_tensor().cpu().numpy())
        axs[0, 0].set_title("depth")

        axs[0, 1].imshow(result["normal"].cpu().numpy())
        axs[0, 1].set_title("normal")

        axs[1, 0].imshow(result["color"].cpu().numpy())
        axs[1, 0].set_title("color via kernel")

        plt.show()

    def visualize_frustum(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.voxel_coords.cpu().numpy())
        o3d.visualization.draw(pcd)
        return


if __name__ == "__main__":
    import os

    import ssg
    from ssg.envs.igibson_env import iGibsonEnv

    o3d.visualization.webrtc_server.enable_webrtc()

    config_file = os.path.join(ssg.CONFIG_PATH, "config.yaml")
    num_cpu = 1

    env = iGibsonEnv(
        config_file=config_file,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 40.0,
    )

    # Get the intrinsic matrix
    intrinsic = env.simulator.renderer.get_intrinsics()
    slam = Slam(intrinsic=intrinsic)

    env.reset()

    # 10 seconds
    for i in range(100):
        rgb, threed, seg, ins_seg = env.simulator.renderer.render_robot_cameras(
            modes=("rgb", "3d", "seg", "ins_seg")
        )
        extrinsics = env.simulator.renderer.V
        vbg = slam.integrate(
            extrinsics,
            depth=-threed[:, :, 2:3],
            features={
                "color": rgb[:, :, :3],
                "seg": seg[..., 0:1],
                "ins_seg": ins_seg[..., 0:1],
            },
        )

        action = np.array([0, 1])
        state, reward, done, info = env.step(action)
        # slam.visualize_frustum()

    # slam.visualize()

    # active_buf_indices = vbg.hashmap().active_buf_indices().to(o3c.int64)
    # key_tensor = vbg.hashmap().key_tensor()[active_buf_indices]
    # vbg.hashmap().activate(key_tensor)
    # buf_indices, _ = vbg.hashmap().find(key_tensor)

    pcd = vbg.extract_point_cloud()

    pcd_block_coordinates = vbg.compute_unique_block_coordinates(pcd)
    vbg.hashmap().activate(pcd_block_coordinates)
    buf_indices, _ = vbg.hashmap().find(pcd_block_coordinates)

    o3c.cuda.synchronize()
    voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(
        buf_indices
    )
    color = vbg.attribute("color").reshape((-1, 3))[voxel_indices]

    # Using active buf indices
    # color.shape = SizeVector[54960128, 3]
    # voxel_coords.shape = SizeVector[54960128, 3]

    # Using pcd to get voxel_coords
    # color.shape = SizeVector[48689152, 3]
    # voxel_coords.shape = SizeVector[48689152, 3]

    # Using pcd
    # pcd.point["colors"].shape = SizeVector[1510288, 3]

    # pcd = o3d.t.geometry.PointCloud()
    # pcd.point["positions"] = voxel_coords
    # pcd.point["colors"] = color / 255

    o3d.visualization.draw(pcd)

    # o3d.visualization.draw(vbg.extract_point_cloud())
