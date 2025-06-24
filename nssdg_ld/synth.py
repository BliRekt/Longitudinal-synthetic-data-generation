import networkx as nx
from collections import defaultdict
from nssdg_ld.base_model import CategoricalBaseModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from kneed import KneeLocator



def extract_consensus_graph(samples, threshold=0.6, num_nodes=2):
    total = len(samples)
    edge_freq = defaultdict(int)
    for graph, _, _ in samples:
        for edge in graph.edges:
            edge_freq[edge] += 1

    consensus_graph = nx.DiGraph()
    consensus_graph.add_nodes_from(range(num_nodes))
    for edge, count in edge_freq.items():
        if count / total >= threshold:
            consensus_graph.add_edge(*edge)

    return consensus_graph

def extract_segment_cpds(data, allocation, graph, num_categories):
    segment_cpds = []
    segment_lengths = []

    for node in range(data.shape[0]):
        segments = CategoricalBaseModel.get_node_segments(None, allocation, node)
        parents = list(graph.predecessors(node))

        for (start, end) in segments:
            child = data[node, start+1:end]
            if parents:
                parent_data = [data[p, start:end-1] for p in parents]
                parent_configs = np.vstack(parent_data)
                counts = np.zeros((num_categories,) * (len(parents)+1))
                for i in range(child.shape[0]):
                    idx = tuple(parent_configs[:, i]) + (child[i],)
                    counts[idx] += 1
                with np.errstate(divide='ignore', invalid='ignore'):
                    cpds = counts / counts.sum(axis=-1, keepdims=True)
                    cpds = np.nan_to_num(cpds)
                vec = cpds.flatten()
            else:
                counts = np.bincount(child, minlength=num_categories)
                vec = counts / np.sum(counts)

            segment_cpds.append(vec)
            segment_lengths.append(end - start)

    # Pad all CPDs to same length, then append segment length
    max_len = max(len(vec) for vec in segment_cpds)
    padded_cpds = [
        np.concatenate([v, np.zeros(max_len - len(v)), [l]])
        for v, l in zip(segment_cpds, segment_lengths)
    ]

    return np.array(padded_cpds)

def cluster_cpds(cpds, max_k=10):
    cpds_no_length = cpds[:, :-1]  

    # Normalize for distance-based clustering
    cpds_normalized = normalize(cpds_no_length, norm='l2')
    n_samples = len(cpds_normalized)

    inertias = []
    for k in range(1, min(max_k, n_samples) + 1):
        km = KMeans(n_clusters=k, random_state=0).fit(cpds_normalized)
        inertias.append(km.inertia_)

    if len(inertias) == 1:
        k_opt = 1
    else:
        k_opt = find_elbow_point(inertias)

    k_opt = min(k_opt, n_samples)

    # Final KMeans with optimal cluster count
    final_model = KMeans(n_clusters=k_opt, random_state=0).fit(cpds_normalized)
    labels = final_model.labels_
    return final_model, labels


def find_elbow_point(inertias):
    """
    Finds the 'elbow' point in the inertia curve using the 'knee' method.
    """

    x = range(1, len(inertias) + 1)
    kneedle = KneeLocator(x, inertias, curve='convex', direction='decreasing')
    
    elbow = kneedle.elbow
    if elbow is None:
        elbow = 1  # fallback to 1 cluster if no elbow found
    return elbow

def compute_transition_matrix(labels):
    transitions = defaultdict(int)
    for i in range(len(labels) - 1):
        transitions[(labels[i], labels[i+1])] += 1

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    n = len(unique_labels)
    matrix = np.zeros((n, n))
    for (i, j), count in transitions.items():
        matrix[label_to_idx[i], label_to_idx[j]] = count
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix, label_to_idx

def generate_synthetic_sequence(start_probs, transition_matrix, segment_cpds, cluster_lengths,
                                num_nodes, sequence_length, labels, consensus_graph, num_categories=5):
    synthetic = [[] for _ in range(num_nodes)]
    total_time = 0

    current_cluster = np.random.choice(len(start_probs), p=start_probs)

    while total_time < sequence_length:
        #  Retry until we get a non-empty cluster or bail out
        retries = 0
        cluster_cpds = segment_cpds[labels == current_cluster]
        while len(cluster_cpds) == 0 and retries < 10:
            current_cluster = np.random.choice(len(start_probs), p=start_probs)
            cluster_cpds = segment_cpds[labels == current_cluster]
            retries += 1

        if len(cluster_cpds) == 0:
            break  # Bail out if still empty

        #  Sample segment length
        possible_lengths = cluster_lengths[current_cluster]
        seg_len = int(min(np.random.choice(possible_lengths), sequence_length - total_time))

        #  Sample a CPD vector and reshape
        sampled_vec = cluster_cpds[np.random.choice(len(cluster_cpds))]
        cpd_vector = sampled_vec[:-1]  # last value is segment length
        cpd_shape = tuple([num_categories] * int(np.log(len(cpd_vector)) / np.log(num_categories)))
        segment_cpd = cpd_vector.reshape(cpd_shape)

        #  Sample values for this segment
        for t in range(seg_len):
            for node in range(num_nodes):
                parents = list(consensus_graph.predecessors(node))
                if parents:
                    if total_time + t - 1 < 0:
                        probs = np.ones(num_categories) / num_categories
                    else:
                        try:
                            parent_vals = tuple(synthetic[p][total_time + t - 1] for p in parents)
                            parent_vals = tuple(min(v, num_categories - 1) for v in parent_vals)
                            probs = segment_cpd[parent_vals].flatten()
                            if len(probs) != num_categories:
                                probs = probs[:num_categories]
                        except IndexError:
                            probs = np.ones(num_categories) / num_categories
                else:
                    probs = segment_cpd.flatten()

                #  Handle invalid distributions
                if probs.sum() == 0 or np.isnan(probs).any():
                    probs = np.ones(num_categories) / num_categories
                else:
                    probs = probs / np.sum(probs)

                #  Final safety check
                if len(probs) != num_categories:
                    raise ValueError(f"Expected probs of length {num_categories}, got {len(probs)}")

                value = np.random.choice(num_categories, p=probs)
                synthetic[node].append(value)

        total_time += seg_len

        #  Sample next cluster
        if total_time < sequence_length:
            trans_probs = transition_matrix[current_cluster]
            if trans_probs.sum() == 0 or np.isnan(trans_probs).any():
                current_cluster = np.random.choice(len(transition_matrix))
            else:
                current_cluster = np.random.choice(len(transition_matrix), p=trans_probs)

    return np.array(synthetic)