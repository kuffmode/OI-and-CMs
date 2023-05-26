import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from copy import deepcopy
from neurolib.models.hopf import HopfModel

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


@njit
def simulate_delayed_linear_system(adjacency_matrix: np.ndarray,
                            delay_matrix: np.ndarray,
                            input_matrix: np.ndarray,
                            coupling: float = 1,
                            dt: float = 0.001,
                            duration: int = 10,
                            timeconstant: float = 0.01) -> np.ndarray:
    """Simulates a linear system of differential equations described by the given parameters.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix (N,N)
        delay_matrix (np.ndarray): The delay matrix (N,N)
        input_matrix (np.ndarray): Input of shape (N, T) where N is the number of nodes and T is the number of time steps.
        coupling (float, optional): The coupling strength between each node (scales the adjacency_matrix). Defaults to 1.
        dt (float, optional): The time step of the simulation. Defaults to 0.001.
        duration (int, optional): The duration of the simulation in seconds. Defaults to 10.
        timeconstant (float, optional): The time constant of the nodes. Defaults to 0.01.

    Returns:
        np.ndarray: The state of the dynamical system at each time step, of shape (N, T)
    """
    N = input_matrix.shape[0]
    T = np.arange(1, duration/dt + 1) # holds time steps
    X = np.zeros((N, len(T)+1)) # holds variable x

    dt = dt/timeconstant
    for timepoint in range(input_matrix.shape[1] - 1):
        # Compute the delayed input matrix
        delayed_input_matrix = np.zeros_like(input_matrix)
        for i in range(N):
            for j in range(N):
                if delay_matrix[i, j] > 0:
                    delayed_input_matrix[i, timepoint - int(delay_matrix[i, j]/dt)] = input_matrix[j, timepoint]

        # Compute the state update
        X[:, timepoint + 1] = ((1 - dt) * X[:, timepoint]) + dt * (
            coupling * adjacency_matrix @ X[:, timepoint] + delayed_input_matrix[:, timepoint])

    return X

@njit
def kuramoto_model(adjacency_matrix:np.ndarray, dt:float, T:int, omega:np.ndarray, initial_theta:np.ndarray, coupling:float)->np.ndarray:
    """
    Computes the activity of the Kuramoto nodes over time for the given connectivity matrix, time step, duration, natural frequencies, initial values, and coupling strength.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of which the Kuramoto model is plugged in. Should be (N, N)
        dt (float): The time step of the simulation.
        T (int): The duration of the simulation.
        omega (np.ndarray): The natural frequency of the Kuramoto nodes. Should be (N, 1)
        initial_theta (np.ndarray): The initial values of the Kuramoto nodes. Should be (N, 1)
        coupling (float): The coupling strength of the Kuramoto model.

    Returns:
        np.ndarray: The activity of the Kuramoto nodes over time. Will be (N, T)
        NOTE: Already passed through the sine function!
    """
    
    N = adjacency_matrix.shape[0]
    n_steps = int(T / dt)
    theta = np.zeros((N, n_steps))
    theta[:, 0] = initial_theta
    
    for t in range(1, n_steps):
        dtheta = omega + coupling * np.array([np.sum(adjacency_matrix[i,:] * np.sin(theta[:, t-1] - theta[i, t-1])) for i in prange(N)])
        theta[:, t] = theta[:, t-1] + dtheta * dt
    
    return np.sin(theta)


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
    fiber_lengths: np.ndarray,
    index: int,
    model_kwargs:dict={},
) -> np.ndarray:

    model_kwargs

    lesioned_connectivity = deepcopy(adjacency_matrix)
    lesioned_delay = deepcopy(fiber_lengths)
    
    for target in complements:
        lesioned_connectivity[:, target] = 0.0
        lesioned_connectivity[target, :] = 0.0
        lesioned_delay[:, target] = 0.0
        lesioned_delay[target, :] = 0.0
        
    model = HopfModel(Cmat = lesioned_connectivity, Dmat = lesioned_delay)
    model.params['sigma_ou'] = model_kwargs['noise_strength']
    model.params['seed'] = model_kwargs['SEED']
    model.params['duration'] = 1 * 200
    model.params['K_gl'] = model_kwargs['K_gl']
    model.params['a'] = model_kwargs['a']
    model.run()
    
    lesioned_signal = model.x[index]
    return lesioned_signal


def find_density(adjacency_matrix: np.ndarray) -> float:
    return np.where(adjacency_matrix != 0, 1, 0).sum() / adjacency_matrix.shape[0] ** 2


def minmax_normalize(
    data: Union[pd.DataFrame, np.ndarray]
) -> Union[pd.DataFrame, np.ndarray]:
    return (data - data.min()) / (data.max() - data.min())


def log_normalize(adjacency_matrix: np.ndarray):
    return np.nan_to_num(np.log(adjacency_matrix), neginf=0, posinf=0)


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
    
# @njit
# def simple_dynamical_system(
#     adjacency_matrix: np.ndarray,
#     input_matrix: np.ndarray,
#     function: Callable = identity,
#     coupling:float = 1.0,
# ) -> np.ndarray:
#     X = np.zeros((input_matrix.shape[0], input_matrix.shape[1]))
#     for timepoint in range(input_matrix.shape[1] - 1):
#         X[:, timepoint + 1] = function(
#             (coupling * adjacency_matrix) @ X[:, timepoint] + input_matrix[:, timepoint]
#         )

#     return X