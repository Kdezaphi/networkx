#!/usr/bin/env python3.5

import pdb
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

random.seed(42)


class TestGeneric(object):
    numpy = 1  # nosetests attribute, use nosetests -a 'not numpy' to skip test

    @classmethod
    def setupClass(cls):
        global numpy
        try:
            import numpy
        except ImportError:
            raise SkipTest('numpy not available.')

    def setUp(self):
        self.Gp2 = nx.path_graph(2)
        self.Gp3 = nx.path_graph(3)
        self.Gp4 = nx.path_graph(4)
        self.Gp5 = nx.path_graph(5)
        self.Gc3 = nx.cycle_graph(3)
        self.Gi = nx.cycle_graph(3)
        self.Gi.add_edge(2, 3)
        self.Gc4 = nx.cycle_graph(4)
        self.Gg10 = nx.grid_2d_graph(10, 10)

    def launchTest(self):
        self.posGp2, self.qualGp2 = spat_force_atlas_2(self.Gp2)
        self.posGp3, self.qualGp3 = spat_force_atlas_2(self.Gp3)
        self.posGp4, self.qualGp4 = spat_force_atlas_2(self.Gp4)
        self.posGp5, self.qualGp5 = spat_force_atlas_2(self.Gp5)
        self.posGc3, self.qualGc3 = spat_force_atlas_2(self.Gc3)
        self.posGc3_2fixed,\
            self.qualGc3_2fixed = spat_force_atlas_2(self.Gc3,
                                                     pos={0: [0, 400],
                                                          1: [0, -400],
                                                          2: [3000, 0]},
                                                     fixed=[0, 1])
        self.posGi, self.qualGi = spat_force_atlas_2(self.Gc3,
                                                     pos={0: [0, 400],
                                                          1: [0, -400],
                                                          2: [693, 0],
                                                          3: [-693, 0]},
                                                     fixed=[0, 1, 2])
        self.posGc4, self.qualGc4 = spat_force_atlas_2(self.Gc4)
        self.posGg10, self.qualGg10 = spat_force_atlas_2(self.Gg10)


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
    weight **= edge_weight_influence
    adjacency = np.where(weight, 1, 0)
    dissuade = (1 + np.sum(adjacency, axis=1)).reshape((num_nodes, 1)) @ ones.T
    degree = dissuade * (ones @
                         (1 +
                          np.sum(adjacency, axis=1)).reshape((1, num_nodes)))
    x = np.array([[pos[node][0]
                   if pos and node in pos and pos[node][0]
                   else random.random() * num_nodes
                   for node in sorted_nodes]]).reshape((num_nodes, 1))
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
            if max(map(np.max, [δx * α, δy * α])) < 0.01:
                break
        i += 1
    return {sorted_nodes[i]: [x[i][0], y[i][0]] for i in range(len(x))},\
        qualityGraph
