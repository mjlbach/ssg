import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "./floorplan"
CACHE_DIR = "./out"
OUTPUT_DIR = DATA_DIR
SCENE_ID = "Wainscott_0_int"

tree = lambda: collections.defaultdict(tree)

scene_trav_map_size = {
    "Wainscott_0_int": 3000,
    "Merom_0_int": 2200,
    "Benevolence_0_int": 1800,
    "Pomaria_0_int": 2800,
    "Merom_1_int": 2200,
    "Wainscott_1_int": 2800,
    "Rs_int": 1000,
    "Pomaria_1_int": 2800,
    "Benevolence_1_int": 2000,
    "Ihlen_0_int": 2400,
    "Beechwood_0_int": 2400,
    "Benevolence_2_int": 1800,
    "Pomaria_2_int": 1600,
    "Beechwood_1_int": 2400,
    "Ihlen_1_int": 2200,
}

trav_map_resolution = 0.01
trav_map_size = scene_trav_map_size[SCENE_ID]


def world_to_map(xy):
    return (xy / trav_map_resolution + trav_map_size / 2.0).astype(int)


# Load floorplan.
floorplan_file = os.path.join(DATA_DIR, f"{SCENE_ID}.png")
floorplan = plt.imread(floorplan_file)

# Get positions to plot.
data = tree()
for filename in os.listdir(CACHE_DIR):
    if "-" not in filename:
        continue
    scene_id, cat, name = filename.split("-", 2)
    name = name.split(".")[0]
    with open(os.path.join(CACHE_DIR, filename), "r") as f:
        tmp = json.load(f)
    data[scene_id][cat][name] = tmp[scene_id][cat][name]

scene_to_xy = collections.defaultdict(list)
for scene, scene_dic in data.items():
    for cat, cat_dic in scene_dic.items():
        for name, name_dic in cat_dic.items():
            for state, state_dic in name_dic.items():
                for as_name, dic in state_dic.items():
                    as_cat, _ = as_name.rsplit("_", 1)
                    if dic["is_success"]:
                        scene_to_xy[scene].append(dic["pos"][:2])
scene_to_xy = dict(scene_to_xy)
xy = np.array(scene_to_xy[SCENE_ID])
xy = world_to_map(xy)

# Construct heatmap.
heatmap, xedges, yedges = np.histogram2d(
    xy[:, 0], xy[:, 1], bins=150, range=[[0, trav_map_size], [0, trav_map_size]]
)
heatmap = np.ma.masked_where(heatmap == 0, heatmap)

# Visualize heatmap with floorplan.
plt.figure(figsize=(20, 20))
plt.axis("off")
plt.imshow(floorplan, alpha=1, extent=[0, trav_map_size, 0, trav_map_size])
ax = plt.gca()
im = ax.imshow(
    heatmap.T, origin="lower", alpha=0.5, vmin=0, vmax=5, extent=[0, trav_map_size, 0, trav_map_size]
)
im.set_cmap("Reds")

plt.savefig(
    os.path.join(OUTPUT_DIR, f"task_heatmap_{SCENE_ID}.png"),
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)
