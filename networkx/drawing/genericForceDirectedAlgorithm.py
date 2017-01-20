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

random.seed(42)


def spat_force_atlas_2(G, pos=None, fixed=None, size=None, gravity=1,
                       strong_gravity=False, scaling_ratio=50,
                       edge_weight_influence=1, log_attraction=False,
                       dissuade_hubs=False, anti_collision=False,
                       center=[0, 0], iterations=2000):
    sorted_nodes = sorted(G)
    num_nodes = len(G)
    ones = np.ones((len(G), 1))

    weight = np.array(nx.to_numpy_matrix(G, nodelist=sorted_nodes,
                                         weight='weight'))
    weight = weight**edge_weight_influence
    adjacency = np.where(weight, 1, 0)
    dissuade = (1 + np.sum(adjacency, axis=1)).reshape((num_nodes, 1)) @ ones.T
    degree = dissuade * (ones @
                         (1 +
                          np.sum(adjacency, axis=1)).reshape((1, num_nodes)))
    y = np.array([[pos[node][1]
                   if pos and node in pos and pos[node][1]
                   else random.random() * num_nodes
                   for node in sorted_nodes]]).reshape((num_nodes, 1))

    def force_atlas_2(distance):
        np.set_printoptions(precision=3)
        return np.where(distance > 0.1, scaling_ratio * degree / distance,
                        scaling_ratio * degree / 0.1) - distance * weight
    return spacialization(sorted_nodes, x, y, adjacency, iterations,
                          force_atlas_2)


def spacialization(sorted_nodes, x, y, adjacency, iterations, get_force):
    ones = np.ones((len(x), 1))
    Q = np.inf
    α = len(x)
    i = 0
    qualityGraph = []
    while i < iterations:
        Δx = (x @ ones.T) - (ones @ x.T)
        Δy = (y @ ones.T) - (ones @ y.T)
        dist = Δx**2 + Δy**2
        f = get_force(np.sqrt(dist))
        δx = np.sum(Δx * f, axis=1).reshape((len(x), 1))
        δy = np.sum(Δy * f, axis=1).reshape((len(x), 1))
        while i < iterations:
            x_2 = x + α * δx
            y_2 = y + α * δy
            Δx_2 = (x_2 @ ones.T) - (ones @ x_2.T)
            Δy_2 = (y_2 @ ones.T) - (ones @ y_2.T)
            dist_2 = np.sqrt(Δx_2**2 + Δy_2**2)
            Q_2 = np.sum(dist_2 * adjacency) / np.sum(dist_2)
            if Q_2 < Q:
                qualityGraph.append(Q_2)
                Q = Q_2
                break
            qualityGraph.append(Q_2)
            i += 1
            α /= 2
        if i != iterations:
            x = x_2
            y = y_2
            if max(map(np.max, [δx * α, δy * α])) < 1:
                break
        i += 1
    return {sorted_nodes[i]: [x[i][0], y[i][0]] for i in range(len(x))},\
        qualityGraph
