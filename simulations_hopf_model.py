import warnings
import numpy as np
from msapy import msa
import utils as ut
import scipy
from numba import prange

from neurolib.models.hopf import HopfModel
import neurolib.utils.functions as func

warnings.filterwarnings("ignore")


SEED = 2023
rng = np.random.default_rng(seed=SEED)

consensus_mat = scipy.io.loadmat(
    "Consensus_Connectomes.mat",
    simplify_cells=True,
    squeeze_me=True,
    chars_as_strings=True,
)
connectivity = ut.spectral_normalization(
    1, consensus_mat["LauConsensus"]["Matrices"][0][0]
)
fiber_lengths = consensus_mat["LauConsensus"]["Matrices"][0][1]

NOISE_STRENGTH = 0.05


N_TRIALS = 1
N_CORES = -1

N_NODES = len(connectivity)

all_trials = np.zeros((len(connectivity), len(connectivity), N_TRIALS))
lesion_params = {
    "adjacency_matrix": connectivity,
    "fiber_lengths": fiber_lengths / 50,
    "model_kwargs": {
        "noise_strength": NOISE_STRENGTH,
        "SEED": SEED,
        "K_gl": 5.8,
        "a": 0.15,
    },
}

for trial in prange(N_TRIALS):
    ci_mat = msa.estimate_causal_influences(
        elements=list(range(N_NODES)),
        objective_function=ut.lesion_hopf,
        objective_function_params=lesion_params,
        n_permutations=1_000,
        n_cores=N_CORES,
        parallelize_over_games=False,
        permutation_seed=trial,
    )
    ci_mat.to_pickle(
        f"results/causal_modes_hopf_{len(connectivity)}_scaled_delay.pickle"
    )
