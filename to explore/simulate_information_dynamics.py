from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import utils as ut
import warnings
warnings.filterwarnings("ignore")

SEED = 2023

consensus_mat = scipy.io.loadmat(
    "Consensus_Connectomes.mat",
    simplify_cells=True,
    squeeze_me=True,
    chars_as_strings=True,
)

connectivity = ut.spectral_normalization(
    1, consensus_mat["LauConsensus"]["Matrices"][2][0]
)

n_nodes = connectivity.shape[0]
delta = 0.01
tau = 0.02

G = 0.74
duration = 1
noise_strength = 0.05
trials = 1
dynamics = np.zeros((n_nodes, int(duration / delta) + 1, trials))
noise = np.zeros_like(dynamics)
for trial in range(trials):
    rng = np.random.default_rng(seed=SEED+trial)
    for i in range(int(duration / delta) + 1):
        noise[:, i, trial] = rng.normal(0, noise_strength, (n_nodes,))   
    dynamics[:,:,trial] = ut.simulate_dynamical_system_parallel(
        adjacency_matrix=connectivity,
        coupling=G,
        dt=delta,
        timeconstant=tau,
        input_matrix=noise[:,:,trial],
        duration=duration,
    )

data = Data(dynamics, dim_order="psr")
network_analysis = MultivariateTE()
settings = {'cmi_estimator': 'OpenCLKraskovCMI',
            'gpuid': 0,
            'debug': True,
            'max_lag_sources': 5,
            'min_lag_sources': 1,
            }
results = network_analysis.analyse_network(settings=settings, data=data)