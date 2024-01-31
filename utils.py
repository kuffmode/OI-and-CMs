import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from copy import deepcopy
from neurolib.models.hopf import HopfModel
from scipy.linalg import expm
import networkx as nx
from scipy.stats import pearsonr, spearmanr
 
@njit
def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """The identity function. It's for the linear case and I literally stole it from Fabrizio:
        https://github.com/fabridamicelli/echoes/blob/master/echoes/utils.py

    Args:
        x (Union[float, np.ndarray]): input. can be a float or an np array.

    Returns:
        Union[float, np.ndarray]: output. will be whatever the input is!
    """
    return x


@njit
def tanh(x: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """Computes the hyperbolic tangent of the input. Again, I stole this from Fabrizio:
    https://github.com/fabridamicelli/echoes/blob/master/echoes/utils.py

    Args:
        x (Union[float, int, np.ndarray]): input. can be a float or an np array.

    Returns:
        Union[float, np.ndarray]: output, squashed between -1 and 1.
    """
    return np.tanh(x)


@njit
def simulate_dynamical_system(adjacency_matrix:np.ndarray,
                 input_matrix:np.ndarray, 
                 coupling:float=1, 
                 dt:float=0.001, 
                 duration:int=10, 
                 timeconstant:float=0.01,
                 function:Callable = identity, 
                 )->np.ndarray:
    """Simulates a dynamical system described by the given paramteres.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix (N,N; duh)
        input_matrix (np.ndarray): Input of shape (N, T) where N is the number of nodes and T is the number of time steps.
        coupling (float, optional): The coupling strength between each node (scales the adjacency_matrix). Defaults to 1.
        dt (float, optional): The time step of the simulation. Defaults to 0.001.
        duration (int, optional): The duration of the simulation in seconds. Defaults to 10.
        timeconstant (float, optional): The time constant of the nodes, I think it's the same as the 'relaxation time'. Defaults to 0.01.
        function (Callable, optional): The activation function. Defaults to identity, which means it's linear.

    Returns:
        np.ndarray: The state of the dynamical system at each time step so again, the shape is (N, T)
    """
    
    N = input_matrix.shape[0]
    T = np.arange(1, duration/dt + 1) # holds time steps
    X = np.zeros((N, len(T)+1)) # holds variable x
    

    dt = dt/timeconstant
    for timepoint in range(input_matrix.shape[1] - 1):
        
        X[:, timepoint + 1] = ((1 - dt) * X[:, timepoint]) + dt * function(
            (coupling * adjacency_matrix) @ X[:, timepoint] + input_matrix[:, timepoint])
    return X

@njit()
def simulate_dynamical_system_parallel(adjacency_matrix:np.ndarray,
                       input_matrix:np.ndarray, 
                       coupling:float=1, 
                       dt:float=0.001, 
                       duration:int=10, 
                       timeconstant:float=0.01,
                       function:Callable = identity, 
                       )->np.ndarray:
    # TODO: Just add this to the above function as an argument! Parallel = True/False
    
    N = input_matrix.shape[0]
    T = np.arange(1, duration/dt + 1) # holds time steps
    X = np.zeros((N, len(T)+1)) # holds variable x

    dt = dt/timeconstant
    for timepoint in range(len(T)):
        for node in prange(N):
            if timepoint == 0:
                X[node, timepoint] = input_matrix[node, timepoint]
            else:
                inputs = np.dot(coupling * adjacency_matrix[node, :], X[:, timepoint-1]) + input_matrix[node, timepoint]
                X[node, timepoint] = (1 - dt) * X[node, timepoint-1] + dt * function(inputs)

    return X


def sar_model(adjacency_matrix:np.ndarray, omega:float) -> np.ndarray:
    """Computes the spatial autoregressive (SAR) model for the given adjacency matrix and spatial lag parameter (I think!).

    Args:
        adjacency_matrix (np.ndarray): Self explanatory.
        omega (float): The spatial lag parameter of the SAR model.

    Returns:
        np.ndarray: The SAR model matrix (N, N)
    """
    N = adjacency_matrix.shape[0]
    sar = np.linalg.inv(np.eye(N) - omega * adjacency_matrix) @ np.linalg.inv(np.eye(N) - omega * adjacency_matrix.T)
    return sar

def lesion_simple_nodes(
    complements: Tuple,
    adjacency_matrix: np.ndarray,
    index: int,
    input: np.ndarray,
    model: Callable = simulate_dynamical_system,
    model_kwargs:dict={},
) -> np.ndarray:

    lesioned_connectivity = deepcopy(adjacency_matrix)
    for target in complements:
        lesioned_connectivity[:, target] = 0.0
        lesioned_connectivity[target, :] = 0.0

    dynamics = model(adjacency_matrix = lesioned_connectivity,
                     input_matrix=input,
                     **model_kwargs)
    lesioned_signal = dynamics[index]
    return lesioned_signal


def lesion_hopf(
    complements: Tuple,
    adjacency_matrix: np.ndarray,
    index: int,
    model_kwargs:dict={},
) -> np.ndarray:


    lesioned_connectivity = adjacency_matrix.copy()
    
    for target in complements:
        lesioned_connectivity[:, target] = 0.0
        lesioned_connectivity[target, :] = 0.0

        
    model = HopfModel(Cmat = lesioned_connectivity, Dmat = np.zeros_like(adjacency_matrix))
    model.params['sigma_ou'] = model_kwargs['noise_strength']
    model.params['seed'] = model_kwargs['SEED']
    model.params['duration'] = 5 * 100
    model.params['K_gl'] = model_kwargs['K_gl']
    model.params['a'] = model_kwargs['a']
    model.run()
    
    lesioned_signal = model.x[index]
    return lesioned_signal


def find_density(adjacency_matrix: np.ndarray) -> float:
    return np.where(adjacency_matrix != 0, 1, 0).sum() / adjacency_matrix.shape[0] ** 2

def communicability_centrality(adjacency_matrix: np.ndarray) -> np.ndarray:

    # adopted from "communicability_wei" function of the netneurotools python package. See here:
    # https://netneurotools.readthedocs.io/en/latest/
    
    row_sum = adjacency_matrix.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adjacency_matrix @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = expm(for_expm)
    return np.diag(cmc)

def parametrized_communicability(adjacency_matrix: np.ndarray, scaling: float = 0.5) -> np.ndarray:
    
    row_sum = adjacency_matrix.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    for_expm = square_sqrt @ adjacency_matrix @ square_sqrt
    for_expm = scaling * for_expm
    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0
    return cmc

def linear_attenuation_model(adjacency_matrix: np.ndarray, normalize:bool = True, alpha: float = 0.5) -> np.ndarray:
    """This is the dynamical model behind Katz centrality. It's very simiar to communicability but instead of an exponential discount on the longer walks, it's linear.
    The discount factor alpha should be between 0 and the spectral radius of the adjacancy matrix. See here for more information:
    https://arxiv.org/abs/2307.02449

    Args:
        adjacency_matrix (np.ndarray): weighted, directed, undirected matrix of shape (N, N). I haven't implemented it but for binary matrices, just do the last part and skip the
        normalization step. 
        **
        important note by Gorka about the directed graphs: 
        be careful because these measures expect that A_{ij} = 1 if j --> i,
        which is the opposite of the convention in graph theory that, A_{ij} = 1 if i --> j.
        So … if you are defining your adjacency matrices following the graph convention by default, make sure you feed the function with the transpose, A^T
        **
        alpha (float): attenuation factor of the influence. Default is 0.5 assuming your adjacency matrix is normalized to a spectral radius of 1.

    Returns:
        np.ndarray: A matrix of shape (N, N) that describes the influence of each node on every other node given walks of all lengths.
    """
    if normalize:
        # normalize input matrix (took this part from wei_communicability function of netneurotools)
        row_sum = adjacency_matrix.sum(1)
        neg_sqrt = np.power(row_sum, -0.5)
        square_sqrt = np.diag(neg_sqrt)
        adjacency_matrix = square_sqrt @ adjacency_matrix @ square_sqrt
        
    # calculate linear attenuation matrix from the normalized adjacency matrix
    N = len(adjacency_matrix)
    lam = np.linalg.inv(np.eye(N)-(alpha*adjacency_matrix))
    return lam


def gt_centrality(complements, graph):
    lesioned = graph.copy()
    lesioned.remove_nodes_from(complements)
    largest_cc = [len(c) for c in sorted(nx.connected_components(lesioned), key=len, reverse=True)]
    if len(largest_cc) == 0:
        return 0
    else:
        return int(largest_cc[0])


def gt_eff(complements, graph):
    lesioned = graph.copy()
    lesioned.remove_nodes_from(complements)
    return float(nx.global_efficiency(lesioned))

def minmax_normalize(
    data: Union[pd.DataFrame, np.ndarray]
) -> Union[pd.DataFrame, np.ndarray]:
    return (data - data.min()) / (data.max() - data.min())


def log_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.log(adjacency_matrix), neginf=0, posinf=0)

def log_minmax_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    lognorm_adjacency_matrix = minmax_normalize(log_normalize(adjacency_matrix))
    np.fill_diagonal(lognorm_adjacency_matrix,0.)
    return np.where(lognorm_adjacency_matrix!=1.,lognorm_adjacency_matrix,0.)

def spectral_normalization(
    target_radius: float, adjacency_matrix: np.ndarray
) -> np.ndarray:
    spectral_radius = np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))
    return adjacency_matrix * target_radius / spectral_radius


def threshold(lower_threshold: int, adjacency_matrix: np.ndarray) -> np.ndarray:
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix = adjacency_matrix.fillna(0)

    lower = np.percentile(adjacency_matrix, lower_threshold)
    upper = np.percentile(adjacency_matrix, 100 - lower_threshold)
    adjacency_matrix[(adjacency_matrix < upper) & (adjacency_matrix > lower)] = 0.0

    return adjacency_matrix


def event_maker(
    n_units: int,
    timesteps: int,
    probability: float = 1,
    rng: np.random.Generator = np.random.default_rng(seed=2023),
) -> np.ndarray:

    if probability < 1:
        input = rng.choice(
            [0, 1], p=[1 - probability, probability], size=(n_units, timesteps)
        )
        input = input.astype(float)
    else:
        input = np.zeros((n_units, timesteps))
        for node in range(n_units):
            event_timepoints = rng.integers(0, timesteps, 1)
            input[node, event_timepoints] += 1

    return input


def brain_plotter(
    data: np.ndarray,
    coordinates: np.ndarray,
    axis: plt.Axes,
    view: Tuple[int, int] = (90, 180),
    size: int = 20,
    cmap: any = "viridis",
    scatter_kwargs=Optional[None],
) -> plt.Axes:
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}

    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=data,
        cmap=cmap,
        s=size,
        **scatter_kwargs
    )
    axis.view_init(*view)
    axis.axis("off")
    scaling = np.array([axis.get_xlim(), axis.get_ylim(), axis.get_zlim()])
    axis.set_box_aspect(tuple(scaling[:, 1] / 1.2 - scaling[:, 0]))
    return axis


def make_influence_ratio(difference_matrix: pd.DataFrame, axis: int = 0) -> pd.Series:
    positives = (difference_matrix > 0).sum(axis) / len(difference_matrix)
    negatives = (difference_matrix < 0).sum(axis) / len(difference_matrix)
    return positives - negatives


def check_symmetric(adjacency_matrix: np.ndarray, tol: float = 1e-8) -> bool:
    return np.all(np.abs(adjacency_matrix - adjacency_matrix.T) < tol)


def preprocess_for_surface_plot(original_values: pd.DataFrame,
                                hemispheres: list,
                                correct_labels: list,) -> pd.DataFrame:
    """Preprocesses a DataFrame of original values for surface plot visualization.

    This function takes a DataFrame of original values, a list of hemisphere labels,
    and a list of correct labels, and preprocesses the DataFrame for surface plot
    visualization. The function generates new labels for the DataFrame based on the
    hemisphere labels and the original labels, and adds a numbered suffix to repeating
    labels. The function then creates a new DataFrame with the new labels and the
    original values, and reorders the rows of the new DataFrame based on the correct
    labels.

    Args:
        original_values (pd.DataFrame): A DataFrame of original values.
        hemispheres (list): A list of hemisphere labels.
        correct_labels (list): A list of correct labels.

    Returns:
        pd.DataFrame: A preprocessed DataFrame with new labels and reordered rows.
    """
    labels = original_values.index
    new_label = ['ctx-' + hemispheres[i] + '-' + labels[i] for i in range(len(labels))]
    word_count = {}
    lausanne_labels = []
    for word in new_label:
        if '_' in word:
            lausanne_labels.append(word)
            word_count[word] = int(word.split('_')[-1])
        else:
            if word not in word_count:
                word_count[word] = 1
                lausanne_labels.append(word + '_1')
            else:
                word_count[word] += 1
                lausanne_labels.append(word + '_' + str(word_count[word]))
    new_df = pd.DataFrame(data=original_values.values,index=lausanne_labels)
    return new_df.reindex(index=correct_labels)


def sort_by_fc_module(adjacency_matrix, fc_modules):
    unique_modules = list(set(fc_modules))  # Extract the unique strings
    mapping = {string: i for i, string in enumerate(unique_modules, start=1)}

    # Convert your list of strings to a list of integers
    fc_modules_integers = [mapping[string] for string in fc_modules]
    sorted_nodes = sorted(zip(fc_modules_integers, range(len(fc_modules_integers))))
    sorted_labels, sorted_indices = zip(*sorted_nodes)
    sorted_strings = [fc_modules[i] for i in sorted_indices]
    sorted_strings.reverse()

    sorted_adjacency_matrix = adjacency_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_indices = np.flip(np.array(sorted_labels).reshape(-1, 1))
    community_changes = [i for i in range(1, len(sorted_labels)) if sorted_labels[i] != sorted_labels[i-1]]
    community_changes.reverse()
    return sorted_adjacency_matrix, sorted_indices, sorted_strings, community_changes


def community_extractor(target_community, sorted_labels, sorted_adjacency_matrix):

    # Find the indices where the community label matches the target_community
    target_indices = [index for index, label in enumerate(sorted_labels) if label == target_community]

    # Create a subnetwork for the target community
    community_subnetwork = sorted_adjacency_matrix[np.ix_(target_indices, target_indices)]
    return community_subnetwork


def plot_community_colorbar(fig, ax_heatmap, community_data, cmap):
    # Get the position of the heatmap axis
    pos = ax_heatmap.get_position()
    # Define the width of the color bar
    colorbar_width = pos.width * 0.08
    # Create a new axis for the community color bar to the right of the heatmap
    ax_colorbar = fig.add_axes([(pos.x1)-0.017 + colorbar_width, pos.y0, colorbar_width, pos.height])
    # Plot the community data as a vertical color bar
    ax_colorbar.imshow(community_data, aspect='auto', cmap=cmap, origin='lower')
    # Hide the axis ticks
    ax_colorbar.set_xticks([])
    ax_colorbar.set_yticks([])
    
def discrete_cmap(N, base_cmap=None):
    """
    This directly came from: 
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    Create an N-bin discrete colormap from the specified input map
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)