import functools
import tensorflow as tf
import numpy as np
from sklearn import neighbors


def _compute_connectivity(positions, radius, add_self_edges):
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return senders, receivers


def _compute_connectivity_for_batch(positions, n_node, radius, add_self_edges):
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
    receivers_list = []
    senders_list = []
    n_edge_list = []
    num_nodes_in_previous_graphs = 0

    for positions_graph_i in positions_per_graph_list:
        senders_graph_i, receivers_graph_i = _compute_connectivity(
            positions_graph_i, radius, add_self_edges)

        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

        num_nodes_graph_i = len(positions_graph_i)
        num_nodes_in_previous_graphs += num_nodes_graph_i

    senders = np.concatenate(senders_list, axis=0).astype(np.int32)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
    n_edge = np.stack(n_edge_list).astype(np.int32)

    return senders, receivers, n_edge


def compute_connectivity_for_batch_pyfunc(positions, n_node, radius, add_self_edges=True):
    partial_fn = functools.partial(_compute_connectivity_for_batch, add_self_edges=add_self_edges)
    senders, receivers, n_edge = tf.py_function(partial_fn, [positions, n_node, radius], [tf.int32, tf.int32, tf.int32])
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge.set_shape(n_node.get_shape())
    return senders, receivers, n_edge
