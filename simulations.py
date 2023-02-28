import warnings
import numpy as np
from msapy import msa
import utils as ut
import netneurotools.datasets
warnings.filterwarnings("ignore")
SEED = 2023
rng = np.random.default_rng(seed=SEED)

T = 100
NOISE_STRENGTH = 0.5
N_TRIALS = 20
N_CORES = 100
human = netneurotools.datasets.fetch_connectome("human_struct_scale125")
connectivity = ut.spectral_normalization(0.9, human["conn"])

N_NODES = len(connectivity)

input_tensor = rng.normal(0, NOISE_STRENGTH, (N_NODES, T, N_TRIALS))
all_trials = np.zeros((len(connectivity), len(connectivity), N_TRIALS))
lesion_params = {"adjacency_matrix": connectivity}

for trial in range(N_TRIALS):
    lesion_params["input"] = input_tensor[:, :, trial]
    ci_mat = msa.estimate_causal_influences(
        elements=list(range(N_NODES)),
        objective_function=ut.lesion_simple_nodes,
        objective_function_params=lesion_params,
        n_permutations=1_000,
        n_cores=N_CORES,
        parallelize_over_games=True,
        permutation_seed=trial,
    )
    ci_mat.to_pickle(f"causal_modes_linear_weighted_{len(connectivity)}_{trial}.pickle")
