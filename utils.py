import numpy as np
from numba import njit, prange
import pandas as pd
from typing import Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from copy import deepcopy


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
    return np.tanh(x)


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


@njit
def simulate_dynamical_system(adjacency_matrix:np.ndarray,
                 input_matrix:np.ndarray, 
                 coupling:float=1, 
                 dt:float=0.001, 
                 duration:int=10, 
                 timeconstant:float=0.01,
                 function:Callable = identity, 
                 )->np.ndarray:
    
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



@njit(fastmath=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result



@njit()
def balloonWindkessel(z, sampling_rate, alpha=0.32, kappa=0.65, gamma=0.41, tau=0.98, rho=0.34, V0=0.02):
    """
    Computes the Balloon-Windkessel transformed BOLD signal
    Numerical method (for integration): Runge-Kutta 2nd order method (RK2)

    z:          Measure of neuronal activity (space x time 2d array, or 1d time array)
    sampling_rate: sampling rate, or time step (in seconds)
    alpha:      Grubb's exponent
    kappa:      Rate of signal decay (in seconds)
    gamma:      Rate of flow-dependent estimation (in seconds)
    tau:        Hemodynamic transit time (in seconds)
    rho:        Resting oxygen extraction fraction
    V0:         resting blood vlume fraction

    RETURNS:
    BOLD:       The transformed BOLD signal (from neural/synaptic activity)
    s:          Vasodilatory signal
    f:          blood inflow
    v:          blood volume
    q:          deoxyhemoglobin content
    """

    if z.ndim == 2:
        timepoints = z.shape[1]
    else:
        timepoints = len(z)
        z.shape = (1, len(z))

    dt = sampling_rate

    # Constants
    k1 = 7 * rho
    k2 = 2
    k3 = 2 * rho - 0.2

    # Create lambda function to calculate E, flow
    E = lambda x: 1.0 - (1.0 - rho) ** (1.0 / x)  # x is f, in this case
    # Create lambda function to calculate y, the BOLD signal
    y = lambda q1, v1: V0 * (k1 * (1.0 - q1) + k2 * (1.0 - q1 / v1) + k3 * (1.0 - v1))

    # initialize empty matrices to integrate through
    BOLD = np.zeros(z.shape)
    s = np.zeros(z.shape)  # vasodilatory signal
    f = np.zeros(z.shape)  # blood inflow
    v = np.zeros(z.shape)  # blood volume
    q = np.zeros(z.shape)  # deoxyhemoglobin content

    # Set initial conditions
    s[:, 0] = 0.0
    f[:, 0] = 1.0
    v[:, 0] = 1.0
    q[:, 0] = 1.0
    BOLD[:, 0] = y(q[:, 0], v[:, 0])

    ## Obtain mean value of z, and then calculate steady state of variables prior to performing HRF modeling
    # z_mean = np.mean(z, axis=1)
    z_mean = np_apply_along_axis(np.mean,1,z)
    # Run loop until an approximate steady state is reached
    for t in range(timepoints - 1):

        # 1st order increments (regular Euler)
        # s_k1 = z_mean - (1.0/kappa)*s[:,t] - (1.0/gamma)*(f[:,t] - 1.0)
        s_k1 = z_mean - (kappa) * s[:, t] - (gamma) * (f[:, t] - 1.0)
        f_k1 = s[:, t]
        v_k1 = (f[:, t] - v[:, t] ** (1.0 / alpha)) / tau
        q_k1 = (f[:, t] * E(f[:, t]) / rho - (v[:, t] ** (1.0 / alpha)) * q[:, t] / v[:, t]) / tau

        # Compute intermediate values (Euler method)
        s_a = s[:, t] + s_k1 * dt
        f_a = f[:, t] + f_k1 * dt
        v_a = v[:, t] + v_k1 * dt
        q_a = q[:, t] + q_k1 * dt

        # 2nd order increments (RK2 method)
        # s_k2 = z_mean - (1.0/kappa)*s_a - (1.0/gamma)*(f_a - 1.0)
        s_k2 = z_mean - (kappa) * s_a - (gamma) * (f_a - 1.0)
        f_k2 = s_a
        v_k2 = (f_a - v_a ** (1.0 / alpha)) / tau
        q_k2 = (f_a * E(f_a) / rho - (v_a ** (1.0 / alpha)) * q_a / v_a) / tau

        # Compute RK2 increment
        s[:, t + 1] = s[:, t] + (.5 * (s_k1 + s_k2)) * dt
        f[:, t + 1] = f[:, t] + (.5 * (f_k1 + f_k2)) * dt
        v[:, t + 1] = v[:, t] + (.5 * (v_k1 + v_k2)) * dt
        q[:, t + 1] = q[:, t] + (.5 * (q_k1 + q_k2)) * dt

        BOLD[:, t + 1] = y(q[:, t + 1], v[:, t + 1])

        # If an approximate steady state is reached, quit.
        # We know HRF is at least 10 seconds, so make sure we wait at least 10 seconds until identifying a 'steady state'
        if (t * dt) > 10 and np.sum(np.abs(BOLD[:, t + 1] - BOLD[:, t])) == 0: break

    ## After identifying steady state, re-initialize to run actual simulation
    s[:, 0] = s[:, t + 1]
    f[:, 0] = f[:, t + 1]
    v[:, 0] = v[:, t + 1]
    q[:, 0] = q[:, t + 1]
    BOLD[:, 0] = y(q[:, t + 1], v[:, t + 1])

    for t in range(timepoints - 1):
        # 1st order increments (regular Euler)
        # s_k1 = z[:,t] - (1.0/kappa)*s[:,t] - (1.0/gamma)*(f[:,t] - 1.0)
        s_k1 = z[:, t] - (kappa) * s[:, t] - (gamma) * (f[:, t] - 1.0)
        f_k1 = s[:, t]
        v_k1 = (f[:, t] - v[:, t] ** (1.0 / alpha)) / tau
        q_k1 = (f[:, t] * E(f[:, t]) / rho - (v[:, t] ** (1.0 / alpha)) * q[:, t] / v[:, t]) / tau

        # Compute intermediate values (Euler method)
        s_a = s[:, t] + s_k1 * dt
        f_a = f[:, t] + f_k1 * dt
        v_a = v[:, t] + v_k1 * dt
        q_a = q[:, t] + q_k1 * dt

        # 2nd order increments (RK2 method)
        # s_k2 = z[:,t+1] - (1.0/kappa)*s_a - (1.0/gamma)*(f_a - 1.0)
        s_k2 = z[:, t + 1] - (kappa) * s_a - (gamma) * (f_a - 1.0)
        f_k2 = s_a
        v_k2 = (f_a - v_a ** (1.0 / alpha)) / tau
        q_k2 = (f_a * E(f_a) / rho - (v_a ** (1.0 / alpha)) * q_a / v_a) / tau

        # Compute RK2 increment
        s[:, t + 1] = s[:, t] + (.5 * (s_k1 + s_k2)) * dt
        f[:, t + 1] = f[:, t] + (.5 * (f_k1 + f_k2)) * dt
        v[:, t + 1] = v[:, t] + (.5 * (v_k1 + v_k2)) * dt
        q[:, t + 1] = q[:, t] + (.5 * (q_k1 + q_k2)) * dt

        BOLD[:, t + 1] = y(q[:, t + 1], v[:, t + 1])

    return BOLD, s, f, v, q