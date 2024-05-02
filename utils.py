import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
from neurolib.models.hopf import HopfModel
from scipy.linalg import expm
import seaborn as sns


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
def simulate_dynamical_system(
    adjacency_matrix: np.ndarray,
    input_matrix: np.ndarray,
    coupling: float = 1,
    dt: float = 0.001,
    duration: int = 10,
    timeconstant: float = 0.01,
    function: Callable = identity,
) -> np.ndarray:
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
    T = np.arange(1, duration / dt + 1)  # holds time steps
    X = np.zeros((N, len(T) + 1))  # holds variable x

    dt = dt / timeconstant
    for timepoint in range(input_matrix.shape[1] - 1):
        X[:, timepoint + 1] = ((1 - dt) * X[:, timepoint]) + dt * function(
            (coupling * adjacency_matrix) @ X[:, timepoint] + input_matrix[:, timepoint]
        )
    return X


@njit()
def simulate_dynamical_system_parallel(
    adjacency_matrix: np.ndarray,
    input_matrix: np.ndarray,
    coupling: float = 1,
    dt: float = 0.001,
    duration: int = 10,
    timeconstant: float = 0.01,
    function: Callable = identity,
) -> np.ndarray:
    # TODO: Just add this to the above function as an argument! Parallel = True/False

    N = input_matrix.shape[0]
    T = np.arange(1, duration / dt + 1)  # holds time steps
    X = np.zeros((N, len(T) + 1))  # holds variable x

    dt = dt / timeconstant
    for timepoint in range(len(T)):
        for node in prange(N):
            if timepoint == 0:
                X[node, timepoint] = input_matrix[node, timepoint]
            else:
                inputs = (
                    np.dot(coupling * adjacency_matrix[node, :], X[:, timepoint - 1])
                    + input_matrix[node, timepoint]
                )
                X[node, timepoint] = (1 - dt) * X[node, timepoint - 1] + dt * function(
                    inputs
                )

    return X


def sar_model(
    adjacency_matrix: np.ndarray, omega: float, normalize=False
) -> np.ndarray:
    """Computes the spatial autoregressive (SAR) model for the given adjacency matrix and spatial lag parameter.

    Args:
        adjacency_matrix (np.ndarray): Self explanatory.
        omega (float): The spatial lag parameter of the SAR model.
        normalize: (bool, optional): Whether to strength-normalize the adjacency matrix. Defaults to False.

    Returns:
        np.ndarray: The SAR model matrix (N, N)
    """
    if normalize:
        row_sum = adjacency_matrix.sum(1)
        neg_sqrt = np.power(row_sum, -0.5)
        square_sqrt = np.diag(neg_sqrt)
        adjacency_matrix = square_sqrt @ adjacency_matrix @ square_sqrt

    N = adjacency_matrix.shape[0]
    sar = np.linalg.inv(np.eye(N) - omega * adjacency_matrix) @ np.linalg.inv(
        np.eye(N) - omega * adjacency_matrix.T
    )
    return sar


def lesion_simple_nodes(
    complements: Tuple,
    adjacency_matrix: np.ndarray,
    index: int,
    input: np.ndarray,
    model: Callable = simulate_dynamical_system,
    model_kwargs: dict = None,
) -> np.ndarray:
    """
    Lesions the given nodes and simulates the dynamics of the system given the lesion.

    Args:
        complements (Tuple): Which nodes to lesion, comes from MSA.
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.
        index (int): Which node to track, also comes from MSA.
        input (np.ndarray): Input matrix of shape (N, T) where N is the number of nodes and T is the number of time steps.
                            This is basically the gaussian noise but precomputed for two reasons: 1. replicability, and 2. efficiency.
        model (Callable, optional): Model of local dynamics. Defaults to simulate_dynamical_system.
        model_kwargs (dict, optional): kwargs for the model. Defaults to None.

    Returns:
        np.ndarray: Resulted activity of the target node given the lesion. Shape is (T,)
    """

    lesioned_connectivity = deepcopy(adjacency_matrix)
    for target in complements:
        lesioned_connectivity[:, target] = 0.0
        lesioned_connectivity[target, :] = 0.0

    dynamics = model(
        adjacency_matrix=lesioned_connectivity, input_matrix=input, **model_kwargs
    )
    lesioned_signal = dynamics[index]
    return lesioned_signal


def lesion_hopf(
    complements: Tuple,
    adjacency_matrix: np.ndarray,
    index: int,
    model_kwargs: dict = {},
) -> np.ndarray:
    """
    Lesions the Hopf model. It's really close to the `lesion_simple_nodes` function but I wanted to keep it separate because Hopf needed some extra stuff and I was lazy to do nice coding!

    Args:
        complements (Tuple): Which nodes to lesion, comes from MSA.
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.
        index (int): Which node to track, also comes from MSA.
        model_kwargs (dict, optional): kwargs for the model. Defaults to None.

    Returns:
        np.ndarray: Resulted activity of the target node (only the x state) given the lesion. Shape is (T,)
    """

    lesioned_connectivity = adjacency_matrix.copy()

    for target in complements:
        lesioned_connectivity[:, target] = 0.0
        lesioned_connectivity[target, :] = 0.0

    model = HopfModel(Cmat=lesioned_connectivity, Dmat=np.zeros_like(adjacency_matrix))
    model.params["sigma_ou"] = model_kwargs["noise_strength"]
    model.params["seed"] = model_kwargs["SEED"]
    model.params["duration"] = 5 * 100
    model.params["K_gl"] = model_kwargs["K_gl"]
    model.params["a"] = model_kwargs["a"]
    model.run()

    lesioned_signal = model.x[index]
    return lesioned_signal


def find_density(adjacency_matrix: np.ndarray) -> float:
    """Finds the density of the given adjacency matrix. It's the ratio of the number of edges to the number of possible edges.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the network.

    Returns:
        float: The density of the network.
    """
    return np.where(adjacency_matrix != 0, 1, 0).sum() / adjacency_matrix.shape[0] ** 2


def communicability_centrality(adjacency_matrix: np.ndarray) -> np.ndarray:
    """It's a measure of the influence of a node on all other nodes in the network. Didn't implement the binary version of it so this works only for the weighted networks at the moment.

    Args:
        adjacency_matrix (np.ndarray): Weighted, directed, undirected matrix of shape (N, N).

    Returns:
        np.ndarray: Communicability centrality of each node in the network. Shape is (N,)
    """
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


def parametrized_communicability(
    adjacency_matrix: np.ndarray, normalize: bool = True, scaling: float = 0.5
) -> np.ndarray:
    """Also known as scaled communicability. It computes communicability but also scales the decay rate by a factor of `scaling`.
    The smaller the decay rate, the quicker it assumes the walks to be subsiding. The scaling factor should be between 0 and the spectral radius of the adjacancy matrix.
    See here for more information:
    https://arxiv.org/abs/2307.02449
    Works for weighted and binary networks but make sure not to normalize the binary networks.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.
        **
        important note by Gorka Zamora Lopez about the directed graphs:
            be careful because these measures expect that A_{ij} = 1 if j --> i,
            which is the opposite of the convention in graph theory that, A_{ij} = 1 if i --> j.
            So … if you are defining your adjacency matrices following the graph convention by default, make sure you feed the function with the transpose, A^T
        **
        normalize (bool, optional): Make this false for binary networks. Defaults to True.
        scaling (float, optional): This modulates the discount factor. Defaults to 0.5. Assuming your adjacency matrix is normalized to a spectral radius of 1.

    Returns:
        np.ndarray: scaled communicability matrix of shape (N, N)
    """
    # adopted from "communicability_wei" function of the netneurotools python package. See here:
    # https://netneurotools.readthedocs.io/en/latest/

    if normalize:
        row_sum = adjacency_matrix.sum(1)
        neg_sqrt = np.power(row_sum, -0.5)
        square_sqrt = np.diag(neg_sqrt)
        adjacency_matrix = square_sqrt @ adjacency_matrix @ square_sqrt

    adjacency_matrix *= scaling
    cmc = expm(adjacency_matrix)
    cmc[np.diag_indices_from(cmc)] = 0.0
    return cmc


def linear_attenuation_model(
    adjacency_matrix: np.ndarray, normalize: bool = True, alpha: float = 0.5
) -> np.ndarray:
    """This is the dynamical model behind Katz centrality. It's very simiar to communicability but instead of an exponential discount on the longer walks, it's linear.
    The discount factor alpha should be between 0 and the spectral radius of the adjacancy matrix. See here for more information:
    https://arxiv.org/abs/2307.02449
    Works for weighted and binary networks but make sure not to normalize the binary networks.

    Args:
        adjacency_matrix (np.ndarray): weighted, directed, undirected matrix of shape (N, N). I haven't implemented it but for binary matrices, just do the last part and skip the
        normalization step.
        **
        important note by Gorka Zamora Lopez about the directed graphs:
            be careful because these measures expect that A_{ij} = 1 if j --> i,
            which is the opposite of the convention in graph theory that, A_{ij} = 1 if i --> j.
            So … if you are defining your adjacency matrices following the graph convention by default, make sure you feed the function with the transpose, A^T
        **
        normalize (bool, optional): Make this false for binary networks. Defaults to True.
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
    lam = np.linalg.inv(np.eye(N) - (alpha * adjacency_matrix))
    return lam


def minmax_normalize(
    data: Union[pd.DataFrame, np.ndarray],
) -> Union[pd.DataFrame, np.ndarray]:
    """Normalizes data between 0 and 1.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): Data to be normalized. Can be a DataFrame or an np array but in both cases it should be at most 2D.

    Returns:
        Union[pd.DataFrame, np.ndarray]: Normalized data with the same shape as the input.
    """
    return (data - data.min()) / (data.max() - data.min())


def log_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    """Returns the logarithm of the data (adjacency_matrix) but also takes care of the infinit values.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network. Technically can be any matrix but I did it for the adjacency matrices.

    Returns:
        np.ndarray: Normalized data with the same shape as the input.
    """
    return np.nan_to_num(np.log(adjacency_matrix), neginf=0, posinf=0)


def log_minmax_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    """It first takes the logarithm of the data and then normalizes it between 0 and 1. It also takes care of the infinit values and those nasty things.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network. Technically can be any matrix but I did it for the adjacency matrices.

    Returns:
        np.ndarray: Normalized data with the same shape as the input.
    """
    lognorm_adjacency_matrix = minmax_normalize(log_normalize(adjacency_matrix))
    np.fill_diagonal(lognorm_adjacency_matrix, 0.0)
    return np.where(lognorm_adjacency_matrix != 1.0, lognorm_adjacency_matrix, 0.0)


def spectral_normalization(
    target_radius: float, adjacency_matrix: np.ndarray
) -> np.ndarray:
    """Normalizes the adjacency matrix to have a spectral radius of the target_radius. Good to keep the system stable.

    Args:
        target_radius (float): A value below 1.0. It's the spectral radius that you want to achieve. But use 1.0 if you're planning to change the global coupling strength somewhere.
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.

    Returns:
        np.ndarray: Normalized adjacency matrix with the same shape as the input.
    """
    spectral_radius = np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))
    return adjacency_matrix * target_radius / spectral_radius


def threshold(lower_threshold: int, adjacency_matrix: np.ndarray) -> np.ndarray:
    """thresholds the adjacency matrix. It's a simple percentile based thresholding (doing lower < 0 < 100-lower) so the graph might end up being disconnected.

    Args:
        lower_threshold (int): the percentage of the values to be kept. Like 5 means keep the top 5% of the values from both sides of the distribution and remove the middle 95%.
        adjacency_matrix (np.ndarray): The adjacency matrix of the network.

    Returns:
        np.ndarray: thresholded adjacency matrix with the same shape as the input.
    """
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
    """makes some random events. It's a simple function that generates a matrix of shape (n_units, timesteps) where each row is a node and each column is a time step.
        It then flips some of the steps to 1 based on the probability. If the probability is 1, it just puts a 1 at a random time step for each node.

    Args:
        n_units (int): number of nodes in the network.
        timesteps (int): number of time steps.
        probability (float, optional): probability of having an event for each node. Defaults to 1.
        rng (np.random.Generator, optional): random generator. Defaults to np.random.default_rng(seed=2023).

    Returns:
        np.ndarray: events matrix of shape (n_units, timesteps)
    """

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
    """plots the 3D scatter plot of the brain. It's a simple function that takes the data, the coordinates, and the axis and plots the brain.
    It's a modified version the netneurotools python package but you can give it the axis to plot in. See here:
    https://netneurotools.readthedocs.io/en/latest/

    Args:
        data (np.ndarray): the values that need to be mapped to the nodes. Shape is (N,)
        coordinates (np.ndarray): 3D coordinates fo each node. Shape is (N, 3)
        axis (plt.Axes): Which axis to plot in. This means you have to already have a figure and an axis to plot in.
        view (Tuple[int, int], optional): Which view to look at. Defaults to (90, 180).
        size (int, optional): Size of the nodes. Defaults to 20.
        cmap (any, optional): Color map. Defaults to "viridis" which I don't like but you do you.
        scatter_kwargs (_type_, optional): kwargs for the dots. Defaults to Optional[None].

    Returns:
        plt.Axes: matplotlib axis with the brain plotted.
    """
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}

    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=data,
        cmap=cmap,
        s=size,
        **scatter_kwargs,
    )
    axis.view_init(*view)
    axis.axis("off")
    scaling = np.array([axis.get_xlim(), axis.get_ylim(), axis.get_zlim()])
    axis.set_box_aspect(tuple(scaling[:, 1] / 1.2 - scaling[:, 0]))
    return axis


def make_influence_ratio(difference_matrix: pd.DataFrame, axis: int = 0) -> pd.Series:
    """
    Calculates the influence ratio for each element in a DataFrame along a specified axis. Like the differece between communicability and LAM or something like that.

    Args:
        difference_matrix (pd.DataFrame): A DataFrame containing numerical differences between entities. Simply matrix1 - matrix2.
        axis (int, optional): Which axis, translates to incoming vs outgoing in directed networks. Defaults to 0.

    Returns:
        pd.Series: influence ratio for each element along the specified axis.
    """
    positives = (difference_matrix > 0).sum(axis) / len(difference_matrix)
    negatives = (difference_matrix < 0).sum(axis) / len(difference_matrix)
    return positives - negatives


def check_symmetric(adjacency_matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Checks whether the adjacency matrix, as a square NumPy array is symmetric within a given tolerance.


    Args:
        adjacency_matrix (np.ndarray): the adjacency matrix of the network.
        tol (float, optional): the error tolerance. Defaults to 1e-8.

    Returns:
        bool: A boolean value indicating whether the adjacency matrix is symmetric.
    """
    return np.all(np.abs(adjacency_matrix - adjacency_matrix.T) < tol)


def preprocess_for_surface_plot(
    original_values: pd.DataFrame,
    hemispheres: list,
    correct_labels: list,
) -> pd.DataFrame:
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
    new_label = ["ctx-" + hemispheres[i] + "-" + labels[i] for i in range(len(labels))]
    word_count = {}
    lausanne_labels = []
    for word in new_label:
        if "_" in word:
            lausanne_labels.append(word)
            word_count[word] = int(word.split("_")[-1])
        else:
            if word not in word_count:
                word_count[word] = 1
                lausanne_labels.append(word + "_1")
            else:
                word_count[word] += 1
                lausanne_labels.append(word + "_" + str(word_count[word]))
    new_df = pd.DataFrame(data=original_values.values, index=lausanne_labels)
    return new_df.reindex(index=correct_labels)


def sort_by_fc_module(
    adjacency_matrix: np.ndarray, fc_modules: list
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Sorts the adjacency matrix and based on the FC modules. It also returns the sorted indices, strings, and the borders of the modules.


    Args:
        adjacency_matrix (np.ndarray): adjacency matrix of the network.
        fc_modules (list): list of the FC modules assigned to the nodes. The shape of the list should be (N,)

    Returns:
        Tuple[np.ndarray, np.ndarray, list, list]: sorted adjacency matrix, sorted indices, sorted strings, and the borders of the modules.
    """
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
    borders = [
        i
        for i in range(1, len(sorted_labels))
        if sorted_labels[i] != sorted_labels[i - 1]
    ]
    borders.insert(len(borders), len(sorted_adjacency_matrix))
    borders.insert(0, 0)
    return sorted_adjacency_matrix, sorted_indices, sorted_strings, borders


def in_out_community_influence(
    sorted_influence: np.ndarray, one_community_borders: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """splits the influence matrix into inside and outside community influences.
    Inside ones are within module influences so shape is (N, N) and outside ones are between module influences so shape is (M, N) with M is the length of the original matrix.

    Args:
        sorted_influence (np.ndarray): already sorted by FC communities influence matrix, or any other matrix but this is how I used it.
        one_community_borders (Tuple[int, int]): borders of a single community. It's a tuple of two integers. So you have to loop through communities and call this function for each.
    Example:
    for index in range(1,10):
        target_community = sorted_ci[borders[index-1]:borders[index],borders[index-1]:borders[index]]
        within_community_influence[index-1] = target_community.mean()

        between_community_data,_ = ut.in_out_community_influence(sorted_ci, (borders[index-1],borders[index]))
        between_community_influence[index-1] = between_community_data.mean()

    Returns:
        Tuple[np.ndarray, np.ndarray]: matrices of influence within the module with shape (N, N) and from/to module (M, N).
    """
    inside_community = np.arange(len(sorted_influence)) >= one_community_borders[0]
    inside_community &= np.arange(len(sorted_influence)) < one_community_borders[1]

    outside_community = ~inside_community

    inside_to_outside = sorted_influence[outside_community][:, inside_community]
    outside_to_inside = sorted_influence[inside_community][:, outside_community]
    return inside_to_outside, outside_to_inside


def plot_community_colorbar(
    fig: plt.Figure,
    ax_heatmap: plt.Axes,
    community_data: list,
    cmap: matplotlib.colors.Colormap,
) -> None:
    """Plots the community color bar to the right of the heatmap. Trust me this took me a while to figure out.

    Args:
        fig (plt.Fig): which figure to plot in.
        ax_heatmap (plt.axis): what axis to plot the color bar next to.
        community_data (list): the community labels of the nodes. Shape is (N,)
        cmap (matplotlib.colors.Colormap): which color map to use. Shape is (N,) but remember the number of colors should be the same as the number of communities. Or just use 'Set3' or something from seaborn.
    """
    # Get the position of the heatmap axis
    pos = ax_heatmap.get_position()
    # Define the width of the color bar
    colorbar_width = pos.width * 0.08
    # Create a new axis for the community color bar to the right of the heatmap
    ax_colorbar = fig.add_axes(
        [(pos.x1) - 0.017 + colorbar_width, pos.y0, colorbar_width, pos.height]
    )
    # Plot the community data as a vertical color bar
    ax_colorbar.imshow(community_data, aspect="auto", cmap=cmap, origin="lower")
    # Hide the axis ticks
    ax_colorbar.set_xticks([])
    ax_colorbar.set_yticks([])


def discrete_cmap(
    N: int, base_cmap: matplotlib.colors.Colormap = None
) -> matplotlib.colors.Colormap:
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


def community_plotter(
    adj_matrix: np.ndarray,
    module_labels: list,
    fig: plt.Figure,
    ax: plt.Axes,
    heatmap_kwargs: dict,
    module_colors: Optional[Union[list, np.ndarray]] = None,
    line_width: Optional[float] = 0.5,
    line_color: Optional[str] = "#232324",
) -> plt.Axes:
    """plots the adjacency matrix with the community borders.
    It's a simple function that takes the adjacency matrix, the module labels, and the axis and plots the adjacency matrix with the community borders.

    Args:
        adj_matrix (np.ndarray): unsorted adjacency matrix of the network.
        module_labels (list): module labels of the nodes. Shape is (N,)
        fig (plt.Figure): which figure to plot in.
        ax (plt.Axes): which axis to plot in.
        heatmap_kwargs (dict): kwargs for the heatmap.
        module_colors (Optional[Union[list, np.ndarray]], optional): colormap for the module labels. Defaults to None.
        line_width (Optional[float], optional): width of the border line. Defaults to 0.5.
        line_color (_Optional[str]): hex code of the line color, or matplotlib labels like 'black'. Defaults to "#232324" which is a nice black.

    Returns:
        plt.Axes: Axis with the adjacency matrix plotted.
    """

    sorted_adjacency_matrix, sorted_indices, _, borders = sort_by_fc_module(
        adj_matrix, module_labels
    )
    sns.heatmap(sorted_adjacency_matrix, ax=ax, **heatmap_kwargs)
    if module_colors is not None:
        fig.tight_layout()
        plot_community_colorbar(fig, ax, sorted_indices, module_colors)

    for border in borders:
        ax.axhline(y=border, color=line_color, lw=line_width)
        ax.axvline(x=border, color=line_color, lw=line_width)

    return ax
