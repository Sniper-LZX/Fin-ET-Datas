import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_directed_graph(matrix, save_path=None):
    """
    根据输入的邻接矩阵绘制有向图并保存或显示图像。

    参数:
        matrix (np.array): 邻接矩阵，表示节点间的权重。
        save_path (str, optional): 图像保存路径。如果为 None，则直接显示图像。

    返回:
        None
    """
    matrix = np.array(matrix)
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows系统黑体
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 创建有向图
    G = nx.DiGraph()
    num_nodes = matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    # 添加边及其权重
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = matrix[i][j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    # 使用弹簧布局确定节点位置
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=55)

    # 绘制节点和标签
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # 计算边宽和颜色
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    widths = [(w / max_weight) * 5 for w in weights]  # 线宽按权重比例缩放

    # 绘制带箭头的边
    edges = nx.draw_networkx_edges(
        G, pos, edgelist=edges,
        width=widths, arrowstyle='-|>', arrowsize=15,
        edge_color=weights, edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=max_weight
    )

    # 添加边权重标签
    edge_labels = {(u, v): f"{w:.3f}" for u, v, w in G.edges(data='weight')}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, rotate=False)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max_weight))
    sm.set_array([])
    plt.colorbar(sm, label='边权重', shrink=0.8)

    # 显示图形标题
    plt.title("事件关联图可视化", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    else:
        plt.show()

    # 关闭图像以释放内存
    plt.close()
