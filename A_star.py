import heapq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 读取 GraphML 文件
G = nx.read_graphml('zhuhai_drive_simplified.graphml')

# 确保所有节点坐标是浮点数
for node in G.nodes():
    G.nodes[node]['x'] = float(G.nodes[node]['x'])
    G.nodes[node]['y'] = float(G.nodes[node]['y'])

# 定义启发式函数，使用欧几里得距离
def heuristic(node1, node2):
    x1, y1 = G.nodes[node1]['x'], G.nodes[node1]['y']
    x2, y2 = G.nodes[node2]['x'], G.nodes[node2]['y']
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# 手动实现A*算法
def a_star_algorithm(graph, start, goal):
    # 优先级队列，存储节点及其优先级（优先级为路径总代价 + 启发式估计）
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # 跟踪从起点到每个节点的最优路径的实际代价
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0
    
    # 跟踪从起点到当前节点的最优路径
    came_from = {}
    
    # 跟踪从起点到终点的最优路径的总代价
    f_score = {node: float('inf') for node in graph.nodes()}
    f_score[start] = heuristic(start, goal)
    
    while open_set:
        # 取出优先级最高的节点
        current = heapq.heappop(open_set)[1]
        
        # 如果当前节点是终点，重建路径
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # 返回从起点到终点的路径
        
        # 遍历邻居
        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # 如果没有路径

# 设定起点和终点
start_node = '1338091065'
end_node = '1792735344'

# 运行 A* 算法获取路径
path = a_star_algorithm(G, start_node, end_node)

# 设置颜色和边宽
node_color_map = []
edge_color_map = []
node_size_map = []
edge_width_map = []

for node in G:
    if node == start_node:
        node_color_map.append('green')
        node_size_map.append(100)  # 起点加粗
    elif node == end_node:
        node_color_map.append('red')
        node_size_map.append(100)  # 终点加粗
    else:
        node_color_map.append('gray')
        node_size_map.append(0)  # 其他节点不显示

for edge in G.edges():
    if (edge in zip(path, path[1:])) or (edge[::-1] in zip(path, path[1:])):
        edge_color_map.append('blue')  # 路径用蓝色标出
        edge_width_map.append(2.0)  # 路径上的边加粗
    else:
        edge_color_map.append('gray')  # 其他边为灰色
        edge_width_map.append(0.5)  # 默认边宽度为0.5

# 获取节点的位置信息
pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}

# 绘制图
# plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_color=node_color_map, edge_color=edge_color_map, with_labels=False, node_size=node_size_map, width=edge_width_map, arrows = False)
plt.show()