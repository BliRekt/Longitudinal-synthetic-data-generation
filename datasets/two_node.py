import numpy as np

def generate_synthetic_2_node_data(m=100, epsilon=0.5, snr=20, group_id=0, seed=0):
    np.random.seed(seed)
    X = np.zeros(m+1)
    Y = np.zeros(m+1)
    phi_x = np.random.normal(0, 1, m+1)
    phi_y = np.random.normal(0, 1, m+1)

    # Define changepoints by group
    if group_id == 0:
        true_cps = [m // 3, 2 * m // 3]
    elif group_id == 1:
        true_cps = [m // 2]
    elif group_id == 2:
        true_cps = [m // 4, m // 2, 3 * m // 4]
    elif group_id == 3:
        true_cps = [m * i // 5 for i in range(1, 5)]
    elif group_id == 4:
        true_cps = [4 * m // 5]
    else:
        true_cps = []

    beta = np.ones(m+1)
    for i in range(len(true_cps)):
        start = true_cps[i]
        end = true_cps[i+1] if i+1 < len(true_cps) else m
        beta[start:end] *= -1 if i % 2 == 0 else 1

    for t in range(m):
        X[t+1] = np.sqrt(1 - epsilon**2) * X[t] + epsilon * phi_x[t+1]

    sigma_hat = np.std(beta[1:m+1] * X[1:m+1])
    c = sigma_hat / snr

    for t in range(m):
        Y[t+1] = beta[t+1] * X[t] + c * phi_y[t+1]

    return X[1:], Y[1:], beta[1:], true_cps