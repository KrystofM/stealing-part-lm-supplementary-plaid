import numpy as np
from bounders.algos.bellman_ford import bellman_ford


def bellman_ford_bounder(low, high, constraints):
    NTOK = len(low)
    base_top_token, _ = constraints[0]

    # Create graph for max distance
    # Source will be an extra vertex NTOK
    vertices = list(range(NTOK + 1))
    edges = []
    for token, bias in constraints:
        for i in range(NTOK):
            if i != token:
                #     x[t] - x[i] >= bias[i] - bias[t]
                #     x[i] <= x[t] - (bias[i] - bias[t])
                a = bias[i] - bias[token]
                edges.append((token, i, -a))

    # import networkx as nx
    # import matplotlib.pyplot as plt

    # # Create a directed graph
    # G = nx.DiGraph()

    # # Add edges with weights
    # for edge in edges:
    #     start_vertex, end_vertex, weight = edge
    #     G.add_edge(start_vertex, end_vertex, weight=weight)

    # # Position nodes using a layout for better visualization
    # pos = nx.spring_layout(G)

    # # Draw the graph
    # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # # Display the graph
    # plt.title("Directed Graph for Bellman-Ford Algorithm")
    # plt.show()
    # x[s] = 0
    s = NTOK
    for i in range(NTOK):
        # x[i] <= x[s] + 1
        edges.append((s, i, 1))
        # x[s] <= x[i]
        edges.append((i, s, 0))
    # x[base_top_token] >= 1
    # x[s] <= x[base_top_token] - 1
    edges.append((base_top_token, s, -1))

    dist_max = bellman_ford(vertices, edges, NTOK)
    # print(f"dist_max: {dist_max}")

    # Create graph for min distance
    # We're operating in negative space, so we'll flip the signs of the x's
    vertices = list(range(NTOK + 1))
    edges = []
    for token, bias in constraints:
        for i in range(NTOK):
            if i != token:
                #     x[t] - x[i] <= bias[i] - bias[t]
                #     -x[i] <= -x[t] + (bias[i] - bias[t])
                a = bias[i] - bias[token]
                edges.append((i, token, -a))

    # -x[s] = 0
    s = NTOK
    for i in range(NTOK):
        # -x[i] >= -1
        # -x[s] <= -x[i] + 1
        edges.append((i, s, 1))
        # -x[i] <= -x[s]
        edges.append((s, i, 0))
    # -x[base_top_token] <= -1
    # -x[base_top_token] <= -x[s] - 1
    edges.append((s, base_top_token, -1))

    dist_min = -bellman_ford(vertices, edges, NTOK)
    # print(f"dist_min: {dist_min}")

    eps = 1e-6
    #    assert np.all(dist_min <= dist_max)
    assert np.all(dist_min >= -eps)
    #    assert np.all(dist_max <= 1 + eps)

    low = np.maximum(low, dist_min[0:NTOK])
    high = np.minimum(high, dist_max[0:NTOK])

    return {"low": low, "high": high}
