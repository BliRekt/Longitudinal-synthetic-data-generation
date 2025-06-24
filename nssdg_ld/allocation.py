import numpy as np
import math

class Allocation:
    def __init__(self, num_nodes, max_segments=10):
        self.num_nodes = num_nodes
        self.max_segments = max_segments

    def propose_allocation_move(self, current_allocation, current_K):
        """
        Propose a changepoint move (birth, death, or reallocation) for one randomly chosen node.
        Returns:
            new_allocation: updated segment allocation matrix [nodes x time]
            new_K: updated number of segments per node
            acceptance_ratio: prior x proposal ratio
        """
        new_allocation = current_allocation.copy()
        new_K = current_K.copy()
        
        # Select random node for changepoint move
        node = np.random.randint(self.num_nodes)
        current_K_node = current_K[node]

        # Decide move type based on number of segments
        if current_K_node == 1:
            move_type = 'birth'
        elif current_K_node == self.max_segments:
            move_type = 'death'
        else:
            move_type = np.random.choice(['birth', 'death', 'reallocate'], 
                                         p=[0.45, 0.45, 0.1])

        # BIRTH MOVE: Add a new changepoint
        if move_type == 'birth':
            proposed_K_node = current_K_node + 1
            possible_positions = [
                t for t in range(1, current_allocation.shape[1])
                if current_allocation[node, t] == current_allocation[node, t-1]
            ]
            if not possible_positions:
                return current_allocation, current_K, 1.0

            # Select new changepoint
            new_change_point = np.random.choice(possible_positions)
            current_val = current_allocation[node, new_change_point - 1]

            new_allocation[node, new_change_point:] = current_val + 1
            new_K[node] = proposed_K_node

            # Spacing-aware prior ratio
            b_j = new_change_point
            b_jm1 = b_j - 1
            b_jp1 = current_allocation.shape[1]
            for t in range(b_j + 1, current_allocation.shape[1]):
                if current_allocation[node, t] > current_val:
                    b_jp1 = t
                    break
            numerator = (b_jp1 - b_j) * (b_j - b_jm1)
            denominator = (b_jp1 - b_jm1)
            spacing_prior = numerator / denominator if denominator > 0 else 1.0

            # Truncated Poisson on number of segments
            λ = 1.0
            count_prior = (math.exp(-λ) / math.factorial(proposed_K_node - 1)) / \
                        (math.exp(-λ) / math.factorial(current_K_node - 1))

            prior_ratio = count_prior * spacing_prior

            forward_prob = 0.45 / len(possible_positions)
            backward_prob = 0.45 / proposed_K_node
            proposal_ratio = forward_prob / backward_prob if backward_prob > 0 else 1.0

            return new_allocation, new_K, prior_ratio * proposal_ratio


        # REALLOCATE MOVE: Move a changepoint
        elif move_type == 'death':
            if current_K_node <= 1:
                return current_allocation, current_K, 1.0

            change_points = [
                t for t in range(1, current_allocation.shape[1])
                if current_allocation[node, t] != current_allocation[node, t - 1]
            ]
            if not change_points:
                return current_allocation, current_K, 1.0

            cp_to_remove = np.random.choice(change_points)
            current_val = current_allocation[node, cp_to_remove]

            b_j = cp_to_remove
            b_jm1 = max([t for t in change_points if t < b_j], default=0)
            b_jp1 = min([t for t in change_points if t > b_j], default=current_allocation.shape[1])

            numerator = (b_jp1 - b_j)
            denominator = (b_jp1 - b_j) * (b_j - b_jm1)
            spacing_prior = numerator / denominator if denominator > 0 else 1.0

            λ = 1.0
            count_prior = (math.exp(-λ) / math.factorial(current_K_node - 2)) / \
                        (math.exp(-λ) / math.factorial(current_K_node - 1)) if current_K_node > 1 else 1.0

            prior_ratio = count_prior * spacing_prior

            # Remove changepoint
            mask = new_allocation[node, :] > current_val
            new_allocation[node, mask] -= 1
            new_K[node] = current_K_node - 1

            forward_prob = 0.45 / len(change_points)
            backward_prob = 0.45 / (current_allocation.shape[1] - new_K[node])
            proposal_ratio = forward_prob / backward_prob if backward_prob > 0 else 1.0

            return new_allocation, new_K, prior_ratio * proposal_ratio

                # REALLOCATE MOVE: Move an existing changepoint slightly
        elif move_type == 'reallocate':
            change_points = [
                t for t in range(1, current_allocation.shape[1])
                if current_allocation[node, t] != current_allocation[node, t - 1]
            ]
            if not change_points:
                return current_allocation, current_K, 1.0

            # Select one changepoint and a nearby shift
            cp = np.random.choice(change_points)
            shift = np.random.choice([-1, 1])
            new_cp = cp + shift

            if new_cp <= 0 or new_cp >= current_allocation.shape[1]:
                return current_allocation, current_K, 1.0
            if current_allocation[node, new_cp] == current_allocation[node, new_cp - 1]:
                return current_allocation, current_K, 1.0

            # Move changepoint by shifting values
            temp_val = current_allocation[node, cp]
            if shift == -1:
                new_allocation[node, new_cp:cp] = temp_val
            else:
                new_allocation[node, cp:new_cp] = temp_val - 1

            return new_allocation, new_K, 1.0  # Neutral move ratio
