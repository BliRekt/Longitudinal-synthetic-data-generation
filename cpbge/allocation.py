import numpy as np

class Allocation:
    def __init__(self, num_nodes, max_segments=10):
        self.num_nodes = num_nodes
        self.max_segments = max_segments

    def propose_allocation_move(self, current_allocation, current_K):
        """
        Propose a new changepoint allocation for one node.
        There are three move types:
          - birth: add a changepoint (more segments)
          - death: remove a changepoint (fewer segments)
          - reallocate: shift an existing changepoint
        """

        new_allocation = current_allocation.copy()
        new_K = current_K.copy()
        
        # Randomly select a node
        node = np.random.randint(self.num_nodes)
        current_K_node = current_K[node]
        
        # # Decide move type
        if current_K_node == 1:
            move_type = 'birth'  
        elif current_K_node == self.max_segments:
            move_type = 'death'  
        else:
            move_type = np.random.choice(['birth', 'death', 'reallocate'], 
                                        p=[0.45, 0.45, 0.1])
        
        if move_type == 'birth':
            new_K_node = current_K_node + 1
            # Find time points where a changepoint can be added
            possible_positions = []
            for t in range(1, current_allocation.shape[1]):
                if current_allocation[node, t] == current_allocation[node, t-1]:
                    possible_positions.append(t)
            
            if not possible_positions:
                return current_allocation, current_K, 1.0  # No valid positions
            
            # Randomly select a new changepoint location
            new_change_point = np.random.choice(possible_positions)
            current_val = current_allocation[node, new_change_point-1]
            # Update allocation: increment segment IDs after the new changepoint
            new_allocation[node, new_change_point:] = current_val + 1
            new_K[node] = new_K_node
            
            # Determine where new_change_point sits
            b_j = new_change_point
            b_jm1 = new_change_point - 1
            b_jp1 = current_allocation.shape[1]
            for t in range(new_change_point + 1, current_allocation.shape[1]):
                if current_allocation[node, t] > current_val:
                    b_jp1 = t
                    break

            numerator = (b_jp1 - b_j) * (b_j - b_jm1)
            denominator = (b_jp1 - b_jm1)

            if denominator == 0:
                return current_allocation, current_K, 1.0  # Prevent division by zero

            # Prior ratio (encourages evenly spaced changepoints)
            prior_ratio = (2 * current_K_node * (2 * current_K_node + 1)) / ((current_allocation.shape[1] - 2) ** 2)
            prior_ratio *= numerator / denominator

            # Proposal ratio
            forward_prob = 0.45 / len(possible_positions)
            backward_prob = 0.45 / new_K_node
            proposal_ratio = forward_prob / backward_prob if backward_prob > 0 else 1.0

            
            return new_allocation, new_K, prior_ratio * proposal_ratio
            
        elif move_type == 'death':
            # Remove a change-point
            if current_K_node <= 1:
                return current_allocation, current_K, 1.0  # No change
            
            # Find change-points for this node
            change_points = []
            for t in range(1, current_allocation.shape[1]):
                if current_allocation[node, t] != current_allocation[node, t-1]:
                    change_points.append(t)
            
            if not change_points:
                return current_allocation, current_K, 1.0  # No change-points to remove
            
            # Select a change-point to remove
            cp_to_remove = np.random.choice(change_points)
            current_val = current_allocation[node, cp_to_remove]

            b_j = cp_to_remove
            b_jm1 = max([t for t in change_points if t < b_j], default=0)
            b_jp1 = min([t for t in change_points if t > b_j], default=current_allocation.shape[1])

            numerator = (b_jp1 - b_j)
            denominator = (b_jp1 - b_j) * (b_j - b_jm1)

            if denominator == 0:
                return current_allocation, current_K, 1.0  # Prevent div-by-zero

            prior_ratio = (current_allocation.shape[1] - 2) / (2 * (2 * new_K[node] + 1))
            prior_ratio *= numerator / denominator

            # Update allocation by removing the change-point
            mask = new_allocation[node, :] > current_val
            new_allocation[node, mask] -= 1
            new_K[node] = current_K_node - 1
            
            # Proposal ratio (death vs birth)
            forward_prob = 0.45 / len(change_points)
            possible_new_positions = current_allocation.shape[1] - new_K[node]
            backward_prob = 0.45 / max(1, possible_new_positions)
            proposal_ratio = forward_prob / backward_prob if backward_prob > 0 else 1.0
            
            return new_allocation, new_K, prior_ratio * proposal_ratio
            
        else:  # reallocate
            # Find change-points for this node
            change_points = []
            for t in range(1, current_allocation.shape[1]):
                if current_allocation[node, t] != current_allocation[node, t-1]:
                    change_points.append(t)
            
            if not change_points:
                return current_allocation, current_K, 1.0 
            
            # Select a change-point to move
            cp_to_move = np.random.choice(change_points)
            
            # Find neighboring change-points
            sorted_cps = sorted(change_points)
            idx = sorted_cps.index(cp_to_move)
            prev_cp = sorted_cps[idx-1] if idx > 0 else 0
            next_cp = sorted_cps[idx+1] if idx < len(sorted_cps)-1 else current_allocation.shape[1]
            
            # Propose new position uniformly between neighbors
            possible_positions = list(range(prev_cp+1, next_cp))
            if cp_to_move in possible_positions:
                possible_positions.remove(cp_to_move)
            
            if not possible_positions:
                return current_allocation, current_K, 1.0
            
            # Choose a new time to move the changepoint to
            new_cp = np.random.choice(possible_positions)
            
            # Temporarily adjust allocation
            temp_allocation = current_allocation.copy()
            segment_val = temp_allocation[node, cp_to_move]
            
            if new_cp < cp_to_move:
                # Moving left
                temp_allocation[node, new_cp:cp_to_move] = segment_val
            else:
                # Moving right
                temp_allocation[node, cp_to_move:new_cp] = temp_allocation[node, cp_to_move-1]
            
            # Normalize segment labels to ensure they’re 0 to K-1
            _, new_allocation[node, :] = np.unique(temp_allocation[node, :], return_inverse=True)
            
            # Identify the old and new changepoint positions
            b_j = cp_to_move
            b_j_star = new_cp
            b_jm1 = prev_cp
            b_jp1 = next_cp

            numerator = (b_jp1 - b_j_star) * (b_j_star - b_jm1)
            denominator = (b_jp1 - b_j) * (b_j - b_jm1)

            if denominator == 0:
                return current_allocation, current_K, 1.0  # Avoid div-by-zero

            prior_ratio = numerator / denominator
            proposal_ratio = 1.0

            
            return new_allocation, new_K, prior_ratio * proposal_ratio