import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def custom_draw_pc(pcd, idx):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.update_renderer()
    vis.poll_events()
    image = vis.capture_screen_float_buffer(False)
    # vis.register_animation_callback(move_forward)
    plt.imsave(os.path.join("3dpc", f"{idx:03d}.png"), np.asarray(image), dpi=1)
    # vis.run()
    vis.destroy_window()


for idx, file in enumerate(os.listdir("point_clouds")):
    pcd = o3d.io.read_point_cloud(
        f"point_clouds/{file}",
    )
    custom_draw_pc(pcd, idx)
