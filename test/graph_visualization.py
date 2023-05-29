import matplotlib.pyplot as plt
from pathlib import Path
from pyvis.network import Network
import ssg

def plot_modalities(state):

    # plt.ion()

    _, axs = plt.subplots(2, 2, tight_layout=True, dpi=170)

    if "rgb" in state:
        axs[0, 0].imshow(state["rgb"])
        axs[0, 0].set_title("rgb")
        axs[0, 0].set_axis_off()

    if "ego_sem_map" in state:
        axs[0, 1].imshow(state["ego_sem_map"])
        axs[0, 1].set_title("Inferred semantic map")
        axs[0, 1].set_axis_off()

    if "multichannel_map" in state:
        axs[0, 1].imshow(state["multichannel_map"][..., 0])
        axs[0, 1].set_title("Inferred semantic map")
        axs[0, 1].set_axis_off()

    if "agent_path" in state:
        axs[1, 1].imshow(state["agent_path"])
        axs[1, 1].set_title("Agent history")
        axs[1, 1].set_axis_off()
        axs[1, 1].set_xlim(-5, 5)
        axs[1, 1].set_ylim(-5, 5)

    if "depth" in state:
        axs[1, 0].imshow(state["depth"])
        axs[1, 0].set_title("depth")
        axs[1, 0].set_axis_off()

    plt.show()


def generate_pyvis_graph(env):
    net = Network()
    sg = env.scene_graph.G
    for key, value in sg.nodes.items():
        net.add_node(key, label=value["cat_name"], title=value["pos"].tolist())
    # print(sg.edges)
    for u, v, a in sg.edges(data=True):
        # print(u, v, a)
        net.add_edge(u, v, label=str(a["relation"]))
        # print(f"Node id: {key}")
        # for graph_attribute, graph_value in value.items():
        #     print(f"\t{graph_attribute}: {graph_value}")
        # neighbors = []
        # for element in sg.neighbors(key):
        #     neighbors.append(sg.nodes[element]['cat_name'])
        # print(f"\tNeighbors {neighbors}")
    f = Path(ssg.ROOT_PATH).parent.joinpath("vis.html")
    f = str(f)
    print(f"Writing scene graph to {f}")
    net.write_html(f)


# def write_cytoscape():
# if "scene_graph" in state:
#     with open("/home/michael/cytoscape.json", "w") as f:
#         data = nx.cytoscape_data(env.scene_graph.G)
#         for idx, node in enumerate(data['elements']['nodes']): #type: ignore
#             for element in node['data']:
#                 if type(node['data'][element]) is np.ndarray:
#                     data['elements']['nodes'][idx]['data'][element] = node['data'][element].tolist()  #type: ignore
#
#         for idx, edge in enumerate(data['elements']['edges']): #type: ignore
#             for element in edge['data']:
#                 if type(edge['data'][element]) is np.ndarray:
#                     data['elements']['edges'][idx]['data'][element] = edge['data'][element].tolist()  #type: ignore
#                 elif isinstance(edge['data'][element], Enum):
#                     data['elements']['edges'][idx]['data'][element] = str(edge['data'][element])  #type: ignore
#         json.dump(data, f)

# def draw_nx_graph(env, ax):
#
#     from copy import deepcopy
#
#     sg = deepcopy(env.scene_graph.G)
#     local_pos = nx.get_node_attributes(sg, "local_pos")
#     cat = nx.get_node_attributes(sg, "cat")
#
#     node_labels = {}
#     node_colors = []
#     lpos_labels = {}
#     for lpos in local_pos:
#         lpos_labels[lpos] = local_pos[lpos][:2]
#
#     for node in sg:
#         x, y, _ = local_pos[node]
#         # cat_label = cat[node]
#         # color = env.slam.normalization_map[cat_label]/env.slam.max_class
#         # node_colors.append(color)
#         node_labels[
#             node
#         ] = f"LocalPos: {x:.1f}, {y:.2f}\nID: {node}\n\n"
#
#     positions = {
#         obj: (pose[1], -pose[0]) for obj, pose in sg.nodes.data("pos")
#     }
#     nx.draw(
#         sg,
#         pos=positions,
#         ax=ax,
#         # node_color=node_colors,
#         # labels=node_labels,
#         font_size=6,
#         arrowsize=5,
#         node_size=50,
#     )
