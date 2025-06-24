import numpy as np
from scipy.special import gammaln

class CategoricalBaseModel:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.alpha = None  # Will be initialized dynamically based on data

    def compute_local_marginal_likelihood(self, data_subset, parent_set):
        m = data_subset.shape[1]
        if m == 0:
            return 0.0

        child_data = data_subset[-1, :]

        all_values = data_subset.flatten() if parent_set else child_data
        max_val = int(np.max(all_values)) + 1
        alpha = np.ones(max_val)  # symmetric Dirichlet prior

        if not parent_set:
            counts = np.bincount(child_data, minlength=max_val)
            return np.sum(gammaln(counts + alpha)) - gammaln(np.sum(counts + alpha))

        parent_data = data_subset[:-1, :]

        if len(parent_set) == 1:
            joint_counts = np.zeros((max_val, max_val))
            for p, c in zip(parent_data[0], child_data):
                joint_counts[p, c] += 1
        else:
            parent_states = np.zeros(parent_data.shape[1], dtype=int)
            for i in range(parent_data.shape[1]):
                state = 0
                for j, val in enumerate(parent_data[:, i]):
                    state += val * (max_val ** j)
                parent_states[i] = state

            num_parent_states = max(parent_states) + 1
            joint_counts = np.zeros((num_parent_states, max_val))
            for ps, cs in zip(parent_states, child_data):
                joint_counts[ps, cs] += 1

        log_ml = 0
        for ps in range(joint_counts.shape[0]):
            counts = joint_counts[ps, :]
            log_ml += np.sum(gammaln(counts + alpha)) - gammaln(np.sum(counts + alpha))

        return log_ml

    def compute_bde_score(self, data, graph, allocation):
        log_score = 0.0

        for node in range(self.num_nodes):
            parents = list(graph.predecessors(node))
            segments = self.get_node_segments(allocation, node)

            for start, end in segments:
                if end - start <= 1:
                    continue

                child_data = data[node, start+1:end].reshape(1, -1)

                if parents:
                    parent_data = np.vstack([data[p, start:end-1] for p in parents])
                    if graph.has_edge(node, node):
                        self_data = data[node, start:end-1].reshape(1, -1)
                        parent_data = np.vstack([parent_data, self_data]) if parent_data.size else self_data
                    data_subset = np.vstack([parent_data, child_data])
                else:
                    if graph.has_edge(node, node):
                        parent_data = data[node, start:end-1].reshape(1, -1)
                        data_subset = np.vstack([parent_data, child_data])
                    else:
                        data_subset = child_data

                if data_subset.shape[1] < 5:
                    continue

                parent_set = parents.copy()
                if graph.has_edge(node, node):
                    parent_set.append(node)

                log_score += self.compute_local_marginal_likelihood(data_subset, parent_set)

                if len(parent_set) > 0:
                    parent_data_subset = data_subset[:-1, :]
                    if parent_data_subset.shape[0] > 0:
                        log_score -= self.compute_local_marginal_likelihood(parent_data_subset, [])

        return log_score

    def get_node_segments(self, allocation, node):
        segments = []
        current_segment_start = 0
        current_segment_value = allocation[node, 0]

        for t in range(1, allocation.shape[1]):
            if allocation[node, t] != current_segment_value:
                segments.append((current_segment_start, t))
                current_segment_start = t
                current_segment_value = allocation[node, t]

        segments.append((current_segment_start, allocation.shape[1]))
        return segments

