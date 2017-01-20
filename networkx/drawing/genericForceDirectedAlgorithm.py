#!/usr/bin/env python3.5

import sys
import pdb
import json
import random
import cProfile
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from PoulpeAPI import extract_save_json, all_metadata_files, separate_nodes_and_edges, get_compound_nodes, set_nodes_position

# DB_PATH = sys.argv[1]

random.seed(42)
# graph = sum([json.load(open(f)) for f in all_metadata_files()], [])
# nodes, edges = separate_nodes_and_edges(graph)
# lonely_nodes = [node for node in nodes if node['data']['id'] not in [edge['data']['source'] for edge in edges]
#                 and node['data']['id'] not in [edge['data']['target'] for edge in edges]]
# fixed_nodes = []
#
#
# def apply_layout(list_all_json_data, fixed_nodes, lonely_nodes):
#     '''Return the json list of graph to cytoscape with the list
#     of all nodes and edges
#     '''
#     graph = nx.Graph()
#     nodes, edges = separate_nodes_and_edges(list_all_json_data)
#     for node in nodes:
#         node['position']['movable'] = True
#     closed_nodes = [node for node in nodes if not node['data']['open']]
#     compound_nodes = get_compound_nodes(closed_nodes)
#     relevant_nodes = [node for node in nodes if node not in fixed_nodes and node not in closed_nodes and node not in lonely_nodes]
#
#     fixed_nodes += [node for node in compound_nodes if not node['position']['movable']]
#     relevant_nodes += [node for node in compound_nodes if node['position']['movable']]
#
#     pos = {node['data']['id']: [node['position']['x'], node['position']['y']] for node in fixed_nodes + relevant_nodes}
#     size = {node['data']['id']: 30 for node in fixed_nodes + relevant_nodes}
#
#     for node_1, node_2 in itertools.permutations(fixed_nodes + relevant_nodes, r=2):
#         if node_1['data']['id'] in node_2['data']['parent-file']:
#             graph.add_edge(node_1['data']['id'], node_2['data']['id'], weight=1)
#         elif node_1['data']['type'] not in ['file', 'country'] \
#                 and node_1['data']['type'] == node_2['data']['type'] \
#                 and node_1['data']['parent-file'] == node_2['data']['parent-file']:
#             graph.add_edge(node_1['data']['id'], node_2['data']['id'], weight=3)
#
#     if relevant_nodes:
#         x, y, qualityGraph = spat_force_atlas_2(graph, pos=pos, size=size, edge_weight_influence=2)
#
#         plt.plot(qualityGraph)
#         plt.show()
#         plt.plot(x, y, 'ro')
#         plt.show()

    #     nodes_position = ForceAtlas2(graph, pos=pos, iterations=15 * len(relevant_nodes),
    #                                  fixed=[node['data']['id'] for node in fixed_nodes],
    #                                  size=size, adjustBySize=True, logAttraction=True)
    #     set_nodes_position(nodes_position, lonely_nodes, closed_nodes, relevant_nodes)
    #
    #     for node in closed_nodes + lonely_nodes + relevant_nodes:
    #         if not node['data']['id'].isupper():
    #             extract_save_json(node['data']['id'], [node] + [edge for edge in edges if node['data']['id'] == edge['data']['source']])
    # return(fixed_nodes + closed_nodes + lonely_nodes + relevant_nodes + edges)


cos(x), arccos(x), cos(arccos(x))
sin(x), arcsin(x), sin(arcsin(x))
tan(x), arctan(x), tan(arctan(x))

def spat_force_atlas_2(G, pos=None, fixed=None, size=None, gravity=1,
                       strong_gravity=False, scaling_ratio=50,
                       edge_weight_influence=1, log_attraction=False,
                       dissuade_hubs=False, anti_collision=False,
                       center=[0,0], iterations=2000):
    sorted_nodes = sorted(G)
    num_nodes = len(G) # + 1
    Ones = np.ones((len(G), 1))

    weight = np.array(nx.to_numpy_matrix(G, nodelist=sorted_nodes, weight='weight'))
    weight = weight**edge_weight_influence
    adjacency = np.where(weight, 1, 0)
    dissuade = (1 + np.sum(adjacency, axis=1)).reshape((num_nodes, 1)) @ Ones.T
    degree = dissuade * (Ones @ (1 + np.sum(adjacency, axis=1)).reshape((1, num_nodes)))
    # gravity = np.concatenate((Ones, np.zeros((num_nodes, num_nodes - 1))), axis=1)
    # gravity_op = np.ones((num_nodes, num_nodes)) - gravity
    x = np.array([[pos[node][0] if pos and node in pos and pos[node][0] else random.random() * num_nodes for node in sorted_nodes]]).reshape((num_nodes, 1)) # .insert(0, [0])
    y = np.array([[pos[node][1] if pos and node in pos and pos[node][1] else random.random() * num_nodes for node in sorted_nodes]]).reshape((num_nodes, 1)) # .insert(0, [0])
    # print(x)
    # print(y)

    # np.set_printoptions(threshold=np.inf)
    # print('weight:')
    # print(weight)
    # print(np.shape(weight))
    # print()

    def force_atlas_2(distance):
        # np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=3)
        # print('distance: ' + str(distance))
        # print()
        # print('weight: ' + str(weight))
        # print()
        # print('degree: ' + str(degree))
        # print()
        # print('np.where(distance > 1, scaling_ratio * degree / distance, scaling_ratio * degree):')
        # print(np.where(distance > 1, scaling_ratio * degree / distance, scaling_ratio * degree))
        # print()
        # print('distance * weight:')
        # print(distance * weight)
        # print()
        return np.where(distance > 0.1, scaling_ratio * degree / distance, scaling_ratio * degree / 0.1) - distance * weight
    return spacialization(sorted_nodes, x, y, adjacency, iterations, force_atlas_2)


def spacialization(sorted_nodes, x, y, adjacency, iterations, get_force):
    Ones = np.ones((len(x), 1))
    Q = np.inf
    α = len(x)
    i = 0
    qualityGraph = []
    while True and i < iterations:
        Δx = (x @ Ones.T) - (Ones @ x.T)
        Δy = (y @ Ones.T) - (Ones @ y.T)
        dist = Δx**2 + Δy**2
        f = get_force(np.sqrt(dist))
        # print('f:')
        # print(f)
        # pdb.set_trace()
        δx = np.sum(Δx * f, axis=1).reshape((len(x),1))
        δy = np.sum(Δy * f, axis=1).reshape((len(x),1))
        # print('max: ' + str(max(map(np.max, [δx, δy]))))
        while True and i < iterations and max(map(np.max, [δx * α, δy * α])) > 1:
            x_2 = x + α * δx
            y_2 = y + α * δy
            # print(x_2)
            # print(y_2)
            Δx_2 = (x_2 @ Ones.T) - (Ones @ x_2.T)
            Δy_2 = (y_2 @ Ones.T) - (Ones @ y_2.T)
            dist_2 = np.sqrt(Δx_2**2 + Δy_2**2)
            Q_2 = np.sum(dist_2 * adjacency) / np.sum(dist_2)
            # print('Q: ' + str(Q))
            # print('Q_2: ' + str(Q_2))
            if Q_2 < Q:
                qualityGraph.append(Q_2)
                Q = Q_2
                break
            qualityGraph.append(Q_2)
            # print('iteration: ' + str(i))
            i += 1
            α /= 2
        if i != iterations:
            x = x_2
            y = y_2
            if max(map(np.max, [δx * α, δy * α])) < 1:
                break
        # print('iteration: ' + str(i))
        i += 1
    return {sorted_nodes[i]: [x[i][0], y[i][0]] for i in range(len(x))}, qualityGraph


# spatialized_graph = apply_layout(graph, fixed_nodes, lonely_nodes)
# nodes, edges = separate_nodes_and_edges(spatialized_graph)
# xList = []
# yList = []
# for node in nodes:
#     xList.append(node['position']['x'])
#     yList.append(node['position']['y'])
# plt.plot(xList, yList, 'ro')
# plt.show()
