import numpy as np

def generate_synthetic_data_2node(num_time_points=41, epsilon=0.5, snr=3):
    """Figure 1a: 2-node network (X → X and X → Y)"""
    np.random.seed(42)
    m = num_time_points
    X = np.zeros(m)
    Y = np.zeros(m)
    phi_X = np.random.normal(0, 1, m)
    phi_Y = np.random.normal(0, 1, m)

    X[0] = np.random.normal(0, 1)
    beta = np.ones(m)
    third = m // 3
    beta[third:(2 * third)] = -1

    for t in range(1, m):
        X[t] = np.sqrt(1 - epsilon ** 2) * X[t - 1] + epsilon * phi_X[t]

    sigma_beta_X = np.std(beta * X)
    c = sigma_beta_X / snr

    for t in range(m - 1):
        Y[t + 1] = beta[t] * X[t] + c * phi_Y[t + 1]

    data = np.vstack([X, Y])
    true_cps = [[third, 2 * third], [third, 2 * third]]
    return data, true_cps

def generate_synthetic_data_4node(num_time_points=100, epsilon=0.5, snr=50):
    """Figure 1b: 4-node network (X → X, X → Y, X → W, X → Z)"""
    np.random.seed(42)
    m = num_time_points
    X = np.zeros(m)
    Y = np.zeros(m)
    W = np.zeros(m)
    Z = np.zeros(m)
    phi_X = np.random.normal(0, 1, m)
    phi_Y = np.random.normal(0, 1, m)
    phi_W = np.random.normal(0, 1, m)
    phi_Z = np.random.normal(0, 1, m)

    X[0] = np.random.normal(0, 1)
    beta_Y = np.ones(m)
    beta_W = np.ones(m)
    beta_Z = np.ones(m)
    third = m // 3
    beta_Y[third:(2 * third)] = -1
    beta_W[third:(2 * third)] = -1
    beta_Z[third:(2 * third)] = -1

    for t in range(1, m):
        X[t] = np.sqrt(1 - epsilon ** 2) * X[t - 1] + epsilon * phi_X[t]

    def get_noise_coeff(beta):
        return np.std(beta * X) / snr

    cY = get_noise_coeff(beta_Y)
    cW = get_noise_coeff(beta_W)
    cZ = get_noise_coeff(beta_Z)

    for t in range(m - 1):
        Y[t + 1] = beta_Y[t] * X[t] + cY * phi_Y[t + 1]
        W[t + 1] = beta_W[t] * X[t] + cW * phi_W[t + 1]
        Z[t + 1] = beta_Z[t] * X[t] + cZ * phi_Z[t + 1]

    data = np.vstack([X, Y, W, Z])
    true_cps = [[third, 2 * third]] * 4
    return data, true_cps

def generate_synthetic_data_4node_sinusoidal(num_time_points=41, c_X=0.25, c_Y=0.25, c_W=0.25, c_Z=0.25):
    """Figure 1c: 4-node sinusoidal network as described in the paper"""
    np.random.seed(42)
    m = num_time_points
    
    # Initialize variables
    X = np.zeros(m)
    Y = np.zeros(m)
    W = np.zeros(m)
    Z = np.zeros(m)
    
    # Initial values
    X[0] = np.random.normal(0, 1)
    Y[0] = np.random.normal(0, 1)
    W[0] = np.random.normal(0, 1)
    Z[0] = np.random.normal(0, 1)
    
    # Generate time series as described in Eq. (10) of the paper
    for t in range(m-1):
        X[t+1] = c_X * np.random.normal(0, 1)
        Y[t+1] = c_Y * np.random.normal(0, 1)
        W[t+1] = W[t] + (2*np.pi)/m + c_W * np.random.normal(0, 1)
        Z[t+1] = c_X * X[t] + c_Y * Y[t] + np.sin(W[t]) + c_Z * np.random.normal(0, 1)
    
    data = np.vstack([X, Y, W, Z])
    
    # The paper doesn't specify true change points for this network
    true_cps = [[], [], [], []] 
    
    return data, true_cps

def generate_synthetic_data_raf(num_time_points=41, epsilon=0.5, snr=10):
    #Figure 1d: RAF pathway with feedback loop on pip3
    np.random.seed(42)
    m = num_time_points
    nodes = ['pip3', 'pip2', 'plcg', 'raf', 'mek', 'erk', 'pkc', 'pka', 'jnk', 'p38', 'akt']
    n = len(nodes)
    data = np.zeros((n, m))
    noise = np.random.normal(0, 1, (n, m))

    data[0, 0] = np.random.normal(0, 1)  # pip3[0]
    for t in range(1, m):
        data[0, t] = np.sqrt(1 - epsilon ** 2) * data[0, t - 1] + epsilon * noise[0, t]

    cps = [np.sort(np.random.choice(range(6, m-5), size=np.random.randint(1, 3), replace=False)).tolist() for _ in range(1, n)]
    beta_vals = [np.ones(m) for _ in range(1, n)]
    for i, cp in enumerate(cps):
        for idx in cp:
            beta_vals[i][idx:] *= -1

    def get_coeff(beta, parent):
        return np.std(beta * parent) / snr

    for i in range(1, n):
        parent_idx = np.random.choice(i)  
        beta = beta_vals[i - 1]
        c = get_coeff(beta, data[parent_idx])
        for t in range(m - 1):
            data[i, t + 1] = beta[t] * data[parent_idx, t] + c * noise[i, t + 1]

    true_cps = [[0]] + cps
    return data, true_cps

