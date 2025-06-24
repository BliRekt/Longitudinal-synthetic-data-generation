import numpy as np
import networkx as nx
from collections import defaultdict
from nssdg_ld.base_model import CategoricalBaseModel
from nssdg_ld.allocation import Allocation
from nssdg_ld.graph_operations import GraphOperations


def count_valid_graph_neighbors(graph, fan_in):
    """
    Count valid single-edge mutations (add, remove, reverse) under a fan-in constraint
    """
    nodes = list(graph.nodes())
    num_adds = 0
    num_removes = len(graph.edges())
    num_reverses = 0

    for u in nodes:
        for v in nodes:
            if u != v:
                if not graph.has_edge(u, v) and len(list(graph.predecessors(v))) < fan_in:
                    num_adds += 1

    for u, v in graph.edges():
        if u != v and not graph.has_edge(v, u) and len(list(graph.predecessors(u))) < fan_in:
            num_reverses += 1

    return num_adds + num_removes + num_reverses


class MCMC:
    def __init__(self, num_nodes, max_segments=10):
        self.base_model = CategoricalBaseModel(num_nodes)
        self.graph_ops = GraphOperations(num_nodes)
        self.allocation_ops = Allocation(num_nodes, max_segments)
        self.num_nodes = num_nodes

    def mcmc_inference(self, data, num_iterations=10000, burn_in=1000):
        """
        Run MCMC to jointly infer graph structure and changepoint segmentations.
        Uses alternating proposals between structure and allocation moves.
        Returns: posterior edge probabilities, mean number of segments, MAP allocation, full samples.
        """
        num_time_points = data.shape[1] - 1

        # Initialize: fully self-looped graph (X_t depends on X_{t-1})
        current_graph = nx.DiGraph()
        current_graph.add_nodes_from(range(self.num_nodes))
        for node in range(self.num_nodes):
            current_graph.add_edge(node, node)

        # Initial segmentation: 1 segment per node
        current_allocation = np.zeros((self.num_nodes, num_time_points), dtype=int)
        current_K = np.ones(self.num_nodes, dtype=int)

        # Initial BDe score
        current_score = self.base_model.compute_bde_score(data, current_graph, current_allocation)

        # Tracking samples
        samples = []
        graph_counts = defaultdict(int)
        allocation_samples = []
        K_samples = []
        scores = []

        for iteration in range(num_iterations):
            # 50% chance: propose structure move
            if np.random.rand() < 0.5:
                new_graph = self.graph_ops.propose_graph_move(current_graph)
                if new_graph is not None:
                    new_score = self.base_model.compute_bde_score(data, new_graph, current_allocation)

                    # MH acceptance ratio includes proposal symmetry |N(G)| terms
                    n_old = count_valid_graph_neighbors(current_graph, fan_in=3)
                    n_new = count_valid_graph_neighbors(new_graph, fan_in=3)
                    proposal_ratio = n_old / n_new if n_new > 0 else 1.0
                    log_acceptance_ratio = (new_score - current_score) + np.log(proposal_ratio + 1e-10)
                    acceptance_ratio = np.exp(min(0, log_acceptance_ratio))  # min(0, ...) prevents overflow

                    if np.random.rand() < min(1.0, acceptance_ratio):
                        current_graph = new_graph
                        current_score = new_score

            # 50% chance: propose changepoint move
            else:
                new_allocation, new_K, move_ratio = self.allocation_ops.propose_allocation_move(current_allocation, current_K)
                new_score = self.base_model.compute_bde_score(data, current_graph, new_allocation)

                # RJMCMC acceptance ratio includes birth/death/realloc move ratio
                log_acceptance_ratio = (new_score - current_score) + np.log(move_ratio + 1e-10)
                acceptance_ratio = np.exp(min(0, log_acceptance_ratio))
                if np.random.rand() < min(1.0, acceptance_ratio):
                    current_allocation = new_allocation
                    current_K = new_K
                    current_score = new_score

            # Record samples after burn-in and thinning
            if iteration >= burn_in and iteration % 10 == 0:
                samples.append((current_graph.copy(), current_allocation.copy(), current_K.copy()))
                graph_counts[tuple(sorted(current_graph.edges()))] += 1
                allocation_samples.append(current_allocation.copy())
                K_samples.append(current_K.copy())
                scores.append(current_score)

        # Compute posterior edge probabilities
        total_samples = len(samples)
        edge_probs = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                edge_probs[(i, j)] = sum(1 for s in samples if s[0].has_edge(i, j)) / total_samples

        # Segment summary
        mean_K = np.mean(K_samples, axis=0)
        best_idx = np.argmax(scores)
        map_allocation = allocation_samples[best_idx]

        return edge_probs, mean_K, map_allocation, samples
