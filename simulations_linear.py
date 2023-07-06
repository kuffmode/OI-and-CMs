import warnings
import numpy as np
from msapy import msa
import utils as ut
import scipy
import pandas as pd
from netneurotools import datasets
import networkx as nx
from numba import prange
import bct
warnings.filterwarnings("ignore")
SEED = 2023
rng = np.random.default_rng(seed=SEED)


networks = ['null'] # 'HumanSC', 'macaqueSC', 'mouseSC', 'netneuro_mouse', 'lattice', 'sneppen', 'null'
for network in networks:
    if network == 'HumanSC':
        consensus_mat = scipy.io.loadmat('Consensus_Connectomes.mat',simplify_cells=True,squeeze_me=True,chars_as_strings=True)
        connectivity = consensus_mat['LauConsensus']['Matrices'][2][0]
        
    elif network == 'netneuro_mouse':
        connectivity = datasets.fetch_connectome("mouse")['conn']

    elif network == 'lattice':
        consensus_mat = scipy.io.loadmat('Consensus_Connectomes.mat',simplify_cells=True,squeeze_me=True,chars_as_strings=True)
        connectivity = consensus_mat['LauConsensus']['Matrices'][2][0]
        G = nx.from_numpy_array(connectivity)
        connectivity = nx.to_numpy_array(nx.lattice_reference(G,seed=SEED))

    elif network == 'sneppen':
        consensus_mat = scipy.io.loadmat('Consensus_Connectomes.mat',simplify_cells=True,squeeze_me=True,chars_as_strings=True)
        connectivity = consensus_mat['LauConsensus']['Matrices'][2][0]
        G = nx.from_numpy_array(connectivity)
        connectivity = nx.to_numpy_array(nx.random_reference(G,seed=SEED))
    
    elif network == 'null':
        consensus_mat = scipy.io.loadmat('Consensus_Connectomes.mat',simplify_cells=True,squeeze_me=True,chars_as_strings=True)
        connectivity = consensus_mat['LauConsensus']['Matrices'][2][0]
        connectivity = bct.null_model_und_sign(connectivity,seed=SEED)[0]
        
    else:
        connectivity = np.loadtxt(f"{network}")
    
    connectivity = ut.spectral_normalization(1,connectivity)
    N_NODES = len(connectivity)


    NOISE_STRENGTH = 0.05
    DELTA = 0.01
    TAU = 0.02
    G =0.79  # 0.74 for Human, 0.64 for macaque, 0.79 for mouse, 0.1 for the "misfitted" network
    DURATION = 1

    N_TRIALS = 10
    N_CORES = -1



    input_tensor = rng.normal(0, NOISE_STRENGTH, (N_NODES, int(DURATION/DELTA)+1, N_TRIALS))
    all_trials = np.zeros((len(connectivity), len(connectivity), N_TRIALS))
    lesion_params = {"adjacency_matrix": connectivity,
                    "model_kwargs":{"timeconstant": TAU,
                    "dt": DELTA,
                    "coupling": G,
                    "duration": DURATION}}

    for trial in prange(N_TRIALS):
        lesion_params["input"] = input_tensor[:, :, trial]
        ci_mat = msa.estimate_causal_influences(
            elements=list(range(N_NODES)),
            objective_function=ut.lesion_simple_nodes,
            objective_function_params=lesion_params,
            n_permutations=1_000,
            n_cores=N_CORES,
            parallelize_over_games=False,
            permutation_seed=trial,
        )
        ci_mat.to_pickle(f"results/{network}_causal_modes_l_w_{len(connectivity)}_{trial}.pickle")