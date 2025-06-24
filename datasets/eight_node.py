import numpy as np
import pandas as pd

def generate_8_node_dataset(m=40, epsilon=0.5, noise_scale=0.5, seed=0, group_id=0):
    """  
    Parameters:
    - m (int): number of time steps
    - epsilon (float): AR(1) noise strength for HeartRate
    - noise_scale (float): standard deviation for Gaussian noise in all other nodes
    - seed (int): random seed for reproducibility
    
    Returns:
    - df (DataFrame): time series data for all 8 nodes
    - beta (ndarray): (n_vars x m) matrix of beta values over time
    - cp1, cp2 (int): changepoint indices (1/3 and 2/3 of time)
    """
    np.random.seed(seed)
    n_vars = 8 
    X = np.zeros((n_vars, m + 1))  # Time series data storage (with t=0 included)
    phi = np.random.normal(0, 1, (n_vars, m + 1))  # Independent Gaussian noise

    # Define changepoints based on group_id
    if group_id == 0:
        true_cps = [m // 2]
    elif group_id == 1:
        true_cps = [4 * m // 5]
    elif group_id == 2:
        true_cps = [m // 3, 2 * m // 3]
    elif group_id == 3:
        true_cps = [m // 4, m // 2, 3 * m // 4]
    elif group_id == 4:
        true_cps = [m * i // 5 for i in range(1, 5)]
    else:
        true_cps = []

   # Construct piecewise beta matrix
    beta = np.ones((n_vars, m + 1))
    segments = [0] + true_cps + [m]
    for i in range(len(segments) - 1):
        start, end = segments[i], segments[i + 1]
        beta[:, start:end] = (-1) ** i  # alternate +1, -1

   # Initialize root nodes
    X[0, 0] = np.random.normal(0, 1)   # HeartRate (autoregressive root)
    X[4, 0] = np.random.normal(0, 0.5) # Temperature (noise-only variable)

    # HeartRate (node 0): AR(1) process
    for t in range(m):
        X[0, t + 1] = np.sqrt(1 - epsilon**2) * X[0, t] + epsilon * phi[0, t + 1]

    # Generate dependent nodes
    for t in range(m):
        X[1, t + 1] = beta[1, t + 1] * X[0, t] + noise_scale * phi[1, t + 1]  # BloodPressure
        X[2, t + 1] = beta[2, t + 1] * X[0, t] + noise_scale * phi[2, t + 1]  # RespiratoryRate
        X[3, t + 1] = beta[3, t + 1] * X[2, t] + noise_scale * phi[3, t + 1]  # OxygenSaturation
        X[4, t + 1] = noise_scale * phi[4, t + 1]                            # Temperature
        X[5, t + 1] = beta[5, t + 1] * X[4, t] + noise_scale * phi[5, t + 1]  # WhiteBloodCell
        X[6, t + 1] = beta[6, t + 1] * (0.5 * X[4, t] + 0.5 * X[5, t]) + noise_scale * phi[6, t + 1]  # CRP
        X[7, t + 1] = beta[7, t + 1] * (0.5 * X[4, t] + 0.5 * X[6, t]) + noise_scale * phi[7, t + 1]  # GlucoseLevel

    # Variable names (healthcare-inspired)
    variables = [
        "HeartRate", "BloodPressure", "RespiratoryRate", "OxygenSaturation",
        "Temperature", "WhiteBloodCell", "CRP", "GlucoseLevel"
    ]

    # Create DataFrame: drop initial t=0, add time column
    df = pd.DataFrame(X[:, 1:].T, columns=variables)

    return df, beta[:, 1:], true_cps 