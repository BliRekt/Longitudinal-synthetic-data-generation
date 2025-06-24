import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import KBinsDiscretizer
from nssdg_ld.mcmc import MCMC
from nssdg_ld.allocation import Allocation
from datasets.two_node import generate_synthetic_2_node_data

def compute_structure_auc(edge_probs, true_adj):
    num_nodes = true_adj.shape[0]
    pred_adj = np.zeros((num_nodes, num_nodes))
    for (i, j), prob in edge_probs.items():
        pred_adj[i, j] = prob
    return roc_auc_score(true_adj.flatten(), pred_adj.flatten())

def extract_changepoints_from_allocation(allocation_row):
    return [t for t in range(1, len(allocation_row)) if allocation_row[t] != allocation_row[t - 1]]

def evaluate_changepoints(true_cps, pred_cps, m, tolerance_ratio=0.15):
    tolerance = int(tolerance_ratio * m)
    matched_pred = set()
    tp = 0
    for true_cp in sorted(set(true_cps)):
        for pred_cp in sorted(set(pred_cps)):
            if abs(true_cp - pred_cp) <= tolerance and pred_cp not in matched_pred:
                tp += 1
                matched_pred.add(pred_cp)
                break
    fp = len(pred_cps) - tp
    fn = len(true_cps) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall

def ib_discretize_from_continuous(X, Y, init_bins=10, max_clusters=5):
    """
    Discretize continuous X and Y using quantile binning and compress X via clustering over p(y|x).
    Returns: X_cat (compressed), Y_cat (quantile-binned)
    """
    # Step 1: Quantile binning
    kbin = KBinsDiscretizer(n_bins=init_bins, encode='ordinal', strategy='quantile')
    x_binned = kbin.fit_transform(X.reshape(-1, 1)).flatten().astype(int)
    y_binned = kbin.fit_transform(Y.reshape(-1, 1)).flatten().astype(int)

    # Step 2: Estimate joint distribution p(x,y)
    pxy = np.zeros((init_bins, init_bins))
    for xb, yb in zip(x_binned, y_binned):
        pxy[xb, yb] += 1
    px = pxy.sum(axis=1, keepdims=True)
    py_x = np.divide(pxy, px, where=px != 0)  # conditional p(y|x)

    # Step 3: Cluster p(y|x) rows to group x into clusters
    distances = squareform(pdist(py_x, metric='euclidean'))  # can change to 'cosine' or 'kl'
    inertias = []
    best_k = 2
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, n_init=10).fit(py_x)
        inertias.append(kmeans.inertia_)

    # Optional: Elbow method (fallback if unclear: use k=3)
    diffs = np.diff(inertias)
    elbow = np.argmin(diffs) + 2 if len(diffs) > 1 else 3

    kmeans = KMeans(n_clusters=elbow, n_init=10).fit(py_x)
    x_cluster_map = {i: label for i, label in enumerate(kmeans.labels_)}
    x_clustered = np.array([x_cluster_map[b] for b in x_binned])
    return x_clustered, y_binned



def run_experiment_on_multiple_datasets(group_ids, m=100, epsilon=0.5, snr=20, seed_offset=0):
    f1_scores = []
    auc_scores = []

    for gid in group_ids:
        X, Y, beta, true_cps = generate_synthetic_2_node_data(
            m=m, epsilon=epsilon, snr=snr,
            group_id=gid % 5,
            seed=seed_offset + gid
        )

        X_cat, Y_cat = ib_discretize_from_continuous(X, Y)
        data = np.vstack([X_cat, Y_cat])

        true_adj = np.array([
            [1, 1],
            [0, 0]
        ])

        mcmc = MCMC(num_nodes=2)
        edge_probs, mean_K, map_allocation, _ = mcmc.mcmc_inference(data, num_iterations=5000, burn_in=1000)

        auc = compute_structure_auc(edge_probs, true_adj)
        auc_scores.append(auc)

        pred_cps = extract_changepoints_from_allocation(map_allocation[1])
        f1, _, _ = evaluate_changepoints(true_cps, pred_cps, m)
        f1_scores.append(f1)

    auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)

    print(f"Structure AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"Changepoint F1: {f1_mean:.4f} ± {f1_std:.4f}")
    

if __name__ == '__main__':
    group_ids = list(range(5))
    run_experiment_on_multiple_datasets(group_ids)
        
