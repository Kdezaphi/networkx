#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import random
import numpy as np
import networkx as nx


def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


def spat_force_atlas_2(G, k=0, g=1,
                       strong_gravity=False,
                       edge_weight_influence=1,
                       log_attraction=False,
                       dissuade_hubs=False,
                       pos=None,
                       fixed=None,
                       iterations=np.inf,
                       displacement_min=1,
                       weight='weight',
                       scale=1.0,
                       center=None,
                       dim=2):
    """Position nodes using Force Atlas 2 force-directed algorithm.

    Parameters
    ----------
    G : NetworkX graph or list of nodes

    k : float  optional (default=2 beyond 100 nodes else 10)
        Scalar to adjust repulsion force. The more it is, the more
        the repulsion is stronger.

    g : float  optional (default=1)
        Scalar to adjust gravity force. The more it is, the more nodes
        are attracted to the center.

    strong_gravity : boolean  optional (default=False)
        Attract more the nodes that are distant frmo the center by removing
        the division by the distance.

    edge_weight_influence : float  optional (default=1)
        If the edges are weighted, power the edges by this value.

    log_attraction : boolean  optional (default=False)
        Use logarithm attraction force instead of proportionnal.

    dissuade_hubs : boolean  optional (default=False)
        Divide the attraction by the degree plus one
        for nodes it point to.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    fixed : list or None  optional (default=None)
        Nodes to keep fixed at initial position.

    iterations : int  optional (default=inf)
        Number of maximum iterations. The algorithm stop when the
        maximum displacement is under displacement_min or when the
        number of iterations is reach.

    displacement_min : float  optional (default=0.1)
        Scalar to set the minimum displacement of the maximum displacement
        to stop the algorithm.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : float  optional (default=1.0)
        Scale factor for positions. The nodes are positioned
        in a box of size [-scale, scale] x [-scale, scale].

    center : array-like or None  optional
        Coordinate pair around which to center the layout.

    dim : int  optional (default=2)
        Dimension of layout

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spat_force_atlas_2(G)
    """
    G, center = _process_params(G, center, dim)
    num_nodes = len(G)
    if num_nodes > 1:
        if not k:
            k = 2 if num_nodes > 100 else 10
        ones = np.ones((num_nodes, 1))

        movable = np.array([[1] if not fixed or node not in fixed else [0]
                            for node in G])

        coord = np.array([pos[node] if pos and node in pos and any(pos[node])
                          else [random.random() * num_nodes
                          for i in range(dim)] for node in G])

        weight = np.asarray(nx.to_numpy_matrix(G, weight=weight))
        weight **= edge_weight_influence
        adjacency = np.where(weight, 1, 0)
        dissuade = (1 + np.sum(adjacency, axis=1)).reshape((num_nodes, 1)) @\
            ones.T
        degree = np.vstack(((1 + np.sum(adjacency, 1)),
                            dissuade *
                            (ones @ (1 + np.sum(adjacency,
                                                1)).reshape((1, num_nodes)))))
        degree[1:] = k * degree[1:]

        def force_atlas_2(distance, Δ_norm):
            f_gravity = degree[0]
            if not strong_gravity:
                f_gravity /= distance[0]
            f_gravity = f_gravity.reshape((num_nodes, 1)) * Δ_norm[0]

            f_repulsion = degree[1:] / distance[1:]
            f_repulsion = f_repulsion.reshape((num_nodes, num_nodes, 1)) *\
                Δ_norm[1:]

            if log_attraction:
                f_attraction = np.log(distance[1:])
            else:
                f_attraction = distance[1:] * weight
            if dissuade_hubs:
                f_attraction /= dissuade
            f_attraction = f_attraction.reshape((num_nodes, num_nodes, 1)) *\
                Δ_norm[1:]

            return np.sum(f_attraction, axis=1) -\
                np.sum(f_repulsion, axis=1) +\
                f_gravity

        return spacialization(G, coord, adjacency, movable, displacement_min,
                              iterations, force_atlas_2, center, scale, dim)
    return {}


def spacialization(G, coord, adjacency, movable, displacement_min, iterations,
                   get_force, center, scale, dim):

    num_nodes = G.number_of_nodes()
    Δ = np.vstack([[coord]] * num_nodes) -\
        np.transpose(np.vstack(([[coord]] * num_nodes)), (1, 0, 2))
    dist = np.linalg.norm(Δ, axis=2)
    dist = np.where(dist < 0.01, 0.01, dist)
    Q = np.inf
    α = 1
    i = 0
    while True and i < iterations:
        Δ = np.vstack((np.array([center - coord]),
                       np.vstack([[coord]] * num_nodes) -
                       np.transpose(np.vstack(([[coord]] * num_nodes)),
                                    (1, 0, 2))))
        dist = np.linalg.norm(Δ, axis=2)
        dist = np.where(dist < 0.01, 0.01, dist)
        Δ_norm = Δ / dist.reshape((num_nodes + 1, num_nodes, 1))
        δ = get_force(dist, Δ_norm) * movable
        while True and i < iterations:
            coord_2 = coord + δ * α
            Δ_2 = np.vstack([[coord_2]] * num_nodes) -\
                np.transpose(np.vstack(([[coord_2]] * num_nodes)), (1, 0, 2))
            dist_2 = np.linalg.norm(Δ_2, axis=2)
            dist_2 = np.where(dist_2 < 0.01, 0.01, dist_2)
            Q_2 = np.sum(dist_2 * adjacency) / np.sum(dist_2)
            if Q_2 < Q:
                Q = Q_2
                coord = coord_2
                print('update quality {}: {}'.format(i, max(map(np.max,
                                                                map(np.abs,
                                                                    δ * α)))))
                break
            if max(map(np.max, map(np.abs, δ * α))) <= displacement_min:
                break
            print('displacement_max {}: {}'.format(i, max(map(np.max,
                                                              map(np.abs,
                                                                  δ * α)))))
            i += 1
            α /= 2
        # print('displacement_max {}: {}'.format(i, max(map(np.max,
        #                                                   map(np.abs,
        #                                                       δ * α)))))
        i += 1
        if max(map(np.max, map(np.abs, δ * α))) <= displacement_min:
            break
    if len(coord) > 1:
        coord_max = np.array([max(coord[:, i]) for i in range(dim)])
        coord_diff = np.array([coord_max[i] - min(coord[:, i])
                               for i in range(dim)])
        coord = (coord + coord_diff - coord_max) / max(coord_diff) *\
            scale + center
        pos = dict(zip(G, coord))
    elif len(G) == 1:
        pos = {node: center for node in G}
    return pos
