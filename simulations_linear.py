import warnings
import numpy as np
from msapy import msa
import utils as ut
import scipy
import pandas as pd

from numba import prange
warnings.filterwarnings("ignore")
SEED = 2023
rng = np.random.default_rng(seed=SEED)


SPECIES = 'macaqueSC' # 'HumanSC','macaqueSC','mouseSC'

if SPECIES == 'HumanSC':
    consensus_mat = scipy.io.loadmat('Consensus_Connectomes.mat',simplify_cells=True,squeeze_me=True,chars_as_strings=True)
    connectivity = ut.spectral_normalization(1,consensus_mat['LauConsensus']['Matrices'][2][0])
    
else:
    connectivity = ut.spectral_normalization(1,np.loadtxt(f"results/communication_models/{SPECIES}"))
N_NODES = len(connectivity)


NOISE_STRENGTH = 0.05
DELTA = 0.01
TAU = 0.02
G =0.64  #0.74 for Human, 0.64 for macaque, 0.79 for mouse
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
    ci_mat.to_pickle(f"results/{SPECIES}_causal_modes_l_w_{len(connectivity)}_{trial}.pickle")