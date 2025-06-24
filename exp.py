#Generate dataset
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('..')) 
from cpbge.data import generate_synthetic_data_2node, generate_synthetic_data_4node

num_individuals = 2
num_timepoints = 100
all_data = []
all_true_cps = []   # True change points (optional, for validation)

def generate_synthetic_data_2node_multicps(num_time_points=100, epsilon=0.5, snr=50):
    """Modified: 2-node network (X → X and X → Y) with 5 change points"""
    np.random.seed(42)
    m = num_time_points
    X = np.zeros(m)
    Y = np.zeros(m)
    phi_X = np.random.normal(0, 1, m)
    phi_Y = np.random.normal(0, 1, m)

    X[0] = np.random.normal(0, 1)
    beta = np.ones(m)

    # Define 5 equal segments
    segment_edges = np.linspace(0, m, 6, dtype=int)  # 6 points = 5 segments
    change_points = segment_edges[1:-1]  # inner edges are the CPs

    # Assign different beta values to each segment
    beta[segment_edges[0]:segment_edges[1]] = 1
    beta[segment_edges[1]:segment_edges[2]] = -2
    beta[segment_edges[2]:segment_edges[3]] = 0
    beta[segment_edges[3]:segment_edges[4]] = -2
    beta[segment_edges[4]:segment_edges[5]] = 1

    # Generate X (autoregressive)
    for t in range(1, m):
        X[t] = np.sqrt(1 - epsilon ** 2) * X[t - 1] + epsilon * phi_X[t]

    # Scale noise
    sigma_beta_X = np.std(beta * X)
    c = sigma_beta_X / snr

    # Generate Y (influenced by beta * X + noise)
    for t in range(m - 1):
        Y[t + 1] = beta[t] * X[t] + c * phi_Y[t + 1]

    data = np.vstack([X, Y])
    true_cps = [change_points.tolist(), change_points.tolist()]  # CPs for both X and Y
    return data, true_cps


# Uncomment the next line to generate a 4-node dataset
# data, true_cps = generate_synthetic_data_4node(num_time_points=num_timepoints)
# node_names = ["X", "Y", "Z", "W"]
# Uncomment the next lines to generate a 2-node dataset
# data, true_cps = generate_synthetic_data_2node_multicps(num_time_points=num_timepoints)
# node_names = ["X", "Y"]

for i in range(num_individuals):
    data_i, true_cps_i = generate_synthetic_data_2node_multicps(num_timepoints)
    all_data.append(data_i)
    all_true_cps.append(true_cps_i)

node_names = ["X", "Y"]

#Find change points
# This script performs MCMC inference on synthetic data, identifies change points,
# and splits the data into segments based on these change points.
import pandas as pd
from cpbge.mcmc import MCMC
from cpbge.plotting import plot_time_series_with_changepoints
from collections import defaultdict
from tqdm import tqdm  # optional for progress bar


def split_dataframe_at_change_points(df, change_points):
    """
    Splits a DataFrame into segments at the given change points.

    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        change_points (list of lists): Nested list of change points (e.g., [[33, 65], [33, 65]])

    Returns:
        list of pd.DataFrame: A list of DataFrame segments.
    """
    # Step 1: Flatten and deduplicate change points
    flat_points = sorted(set(point for sublist in change_points for point in sublist))
    
    # Step 2: Add start and end bounds
    indices = [0] + flat_points + [len(df)]
    
    # Step 3: Split the DataFrame
    segments = [df.iloc[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
    
    return segments

def filter_close_changepoints(cp_list, min_distance, total_length):
    """
    Remove changepoints that would result in segments shorter than min_distance,
    including the first and last segments.

    Parameters:
        cp_list (list[int]): Sorted list of changepoint indices.
        min_distance (int): Minimum allowed segment length.
        total_length (int): Total length of the time series.

    Returns:
        list[int]: Filtered changepoints.
    """
    if not cp_list:
        return []

    cp_list = sorted(cp_list)
    filtered = []

    prev_cp = 0
    for cp in cp_list:
        if cp - prev_cp >= min_distance:
            filtered.append(cp)
            prev_cp = cp
        # else: skip cp because the segment would be too short

    # Ensure final segment is also long enough
    if filtered and total_length - filtered[-1] < min_distance:
        filtered.pop()

    return filtered

all_edge_probs = []
all_allocations = []
all_K = []
all_cps = []

def print_learned_structure(edge_probs, mean_K, allocation, node_names=None, min_segment_length=5):
    """Prints edges and change points using readable node labels."""
    if node_names is None:
        node_names = [str(i) for i in range(len(mean_K))]

    print("Learned Edge Probabilities (edges with prob > 0.5):")
    for (u, v), prob in edge_probs.items():
        if prob > 0.5:
            print(f"  Edge {node_names[u]} → {node_names[v]}: {prob:.3f}")

    print("Changepoints:")
    for node in range(allocation.shape[0]):
        raw_cp = np.where(allocation[node, 1:] != allocation[node, :-1])[0].tolist()
        filtered_cp = filter_close_changepoints(raw_cp, min_segment_length, data_i.shape[1])
        cps= filtered_cp

        print(f"  Node {node_names[node]}: {filtered_cp} (K = {len(filtered_cp) + 1})")

    return cps

for data_i in tqdm(all_data):
    print(f"\n=== Individual {i} ===")
    
    model = MCMC(num_nodes=data_i.shape[0], max_segments=10)
    edge_probs, mean_K, allocation, samples = model.mcmc_inference(
        data_i, num_iterations=500000, burn_in=1000
    )

    cps_per_node = print_learned_structure(edge_probs, mean_K, allocation, node_names=node_names, min_segment_length=5)
    cps_per_node = [cps_per_node] * len(node_names)
    print("Learned Change Points:", cps_per_node)
    # Collect outputs
    all_edge_probs.append(edge_probs)
    all_allocations.append(allocation)
    all_K.append(mean_K)
    all_cps.append(cps_per_node)

    plot_time_series_with_changepoints(data_i, cps_per_node, title=f"Individual {i} Learned Change Points")


    from collections import defaultdict

def extract_bayesian_structure(edge_probs, node_names, threshold=0.5):
    """Builds child-to-parents mapping for nodes with edge probability > threshold."""
    child_parents = defaultdict(list)
    for (u, v), prob in edge_probs.items():
        if prob > threshold:
            child_parents[v].append(u)

    bayesian_structure = [
        [node_names[child], [node_names[parent] for parent in parents]]
        for child, parents in child_parents.items()
    ]
    return bayesian_structure


def apply_temporal_shift(df, bayesian_structure):
    """Adds t-1 columns for self-loops in the Bayesian structure."""
    updated_structure = []

    for child, parents in bayesian_structure:
        new_parents = []
        for parent in parents:
            if parent == child:
                lagged_col = f"{parent}_t-1"
                if lagged_col not in df.columns:
                    print(f"Creating lagged column for self-parenting node: {child}")
                    df[lagged_col] = df[parent].shift(1)
                new_parents.append(lagged_col)
            else:
                new_parents.append(parent)
        updated_structure.append([child, new_parents])

    df = df.dropna().reset_index(drop=True)
    return df, updated_structure

all_structures = {}
all_shifted_data = {}
all_individual_segments = {}

for i, (data_i, edge_probs_i, learnt_cps_i) in enumerate(zip(all_data, all_edge_probs, all_cps)):
    indiv_id = f"indiv_{i}"
    print(f"\n=== Processing {indiv_id} ===")
    
    # Step 1: Extract structure
    structure = extract_bayesian_structure(edge_probs_i, node_names)
    print("Structure:", structure)

    # Step 2: Create DataFrame
    data_df = pd.DataFrame(data_i.T, columns=node_names)

    # Step 3: Apply temporal shift
    shifted_df, updated_structure = apply_temporal_shift(data_df.copy(), structure)
    print("Updated Structure:", updated_structure)

    # Step 4: Segment based on change points
    segments = split_dataframe_at_change_points(shifted_df, learnt_cps_i)
    for seg_idx, segment in enumerate(segments):
        print(f"Segment {seg_idx} ({indiv_id}):")
        print(segment.head())

    # Save using individual ID as key
    all_structures[indiv_id] = updated_structure
    all_shifted_data[indiv_id] = shifted_df
    all_individual_segments[indiv_id] = segments
    
# Get conditional probability tables (CPTs) for the Bayesian network
# Use DataSynthesizer1 to describe the datasets and convert description to CPTs
from collections import defaultdict
from DataSynthesizer1.DataDescriber import DataDescriber
from DataSynthesizer1.lib.utils import display_bayesian_network
import pandas as pd
import json
import os

epsilon = 1.0

# Structure: grouped_by_var[individual_id][variable] = [df, df, ...]
grouped_by_var = defaultdict(lambda: defaultdict(list))

def parse_conditional_probabilities(cond_probs, structure):
    """Same as your original parsing function"""
    tables = {}
    parent_map = {child: parents[0] if parents else None for child, parents in structure}

    for child_var, value in cond_probs.items():
        parent_var = parent_map.get(child_var)

        if isinstance(value, list):  # Marginal
            df = pd.DataFrame({
                child_var: list(range(len(value))),
                f"P({child_var})": value
            })
            tables[child_var] = df

        elif isinstance(value, dict) and parent_var:  # Conditional
            records = []
            for parent_val, probs in value.items():
                parent_val_index = int(parent_val.strip("[]"))
                row = {parent_var: parent_val_index}
                row.update({
                    f"P({child_var}={i} | {parent_var}={parent_val_index})": p
                    for i, p in enumerate(probs)
                })
                records.append(row)

            df = pd.DataFrame(records)
            tables[child_var] = df

        else:
            raise ValueError(f"Unsupported format for variable '{child_var}'")

    return tables

# Assuming you have a dictionary: all_individual_segments = {indiv_id: [segment1, segment2, ...]}
for indiv_id, segments in all_individual_segments.items():
    structure = all_structures[indiv_id]

    for i, segment in enumerate(segments):
        if indiv_id == "indiv_0":
            # Use updated structure and ensure correct lagged segment is used
            describer = DataDescriber(bayesian_network=structure, category_threshold=16)
            print(describer.bayesian_network)
            print(segment)
            # Save this specific segment with lagged columns
            segment.to_csv("temp_segment.csv", index=False)
            try:
                describer.describe_dataset_in_correlated_attribute_mode(
                    "temp_segment.csv",
                    epsilon=epsilon
                )
            except Exception as e:
                print("Describe failed with error:", e)
                import traceback
                traceback.print_exc()
        # describer.describe_dataset_in_correlated_attribute_mode("temp_segment.csv", k=2, epsilon=epsilon, attribute_to_is_candidate_key = {col: False for col in segment.columns})
            json_filename = f"dataset_description_{indiv_id}_segment_{i}.json"
            describer.save_dataset_description_to_file(json_filename)
            display_bayesian_network(describer.bayesian_network)
        else:
            from DataSynthesizer1.lib.PrivBayes import construct_noisy_conditional_distributions
            describer = DataDescriber(bayesian_network=structure, category_threshold=16)
            describer.describe_dataset_in_independent_attribute_mode("temp_segment.csv",epsilon=epsilon)
            df_encoded = describer.encode_dataset_into_binning_indices()
            describer.data_description['bayesian_network'] = structure
            describer.data_description['conditional_probabilities'] = construct_noisy_conditional_distributions(
                structure, df_encoded, epsilon / 2)

        with open(json_filename) as f:
            data = json.load(f)
        

        cond_probs = data["conditional_probabilities"]
        tables = parse_conditional_probabilities(cond_probs, structure)

        # Store and clean
        for var, df in tables.items():
            df = df[[col for col in df.columns if col.startswith("P(")]]
            grouped_by_var[indiv_id][var].append(df)
