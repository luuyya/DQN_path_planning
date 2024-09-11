import osmnx as ox
import matplotlib.pyplot as plt

# 下载未简化的路网数据
place_name = "Zhuhai, China"
G = ox.graph_from_place(place_name, network_type='drive', simplify=False)

# 查看未简化前的节点和边的数量
original_node_count = len(G.nodes)
original_edge_count = len(G.edges)
print(f"Original graph (not simplified): {original_node_count} nodes, {original_edge_count} edges")

# 保存未简化的路网数据到本地文件
ox.save_graphml(G, filepath="zhuhai_drive_unsimplified.graphml")

# 执行节点简化
G_simplified = ox.simplify_graph(G)

# 查看简化后的节点和边的数量
simplified_node_count = len(G_simplified.nodes)
simplified_edge_count = len(G_simplified.edges)
print(f"Simplified graph: {simplified_node_count} nodes, {simplified_edge_count} edges")

# 保存简化后的路网数据到本地文件
ox.save_graphml(G_simplified, filepath="zhuhai_drive_simplified.graphml")

# 可视化简化后的路网数据
fig, ax = ox.plot_graph(G_simplified, bgcolor='black', node_color='white', edge_color='white', node_size=0, edge_linewidth=0.5)
plt.show()
