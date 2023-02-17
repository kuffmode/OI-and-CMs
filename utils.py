import numpy as np
from numba import njit
import pandas as pd
from typing import Union, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from copy import deepcopy


@njit
def simple_dynamical_system(adjacency_matrix:np.ndarray,
                               input_matrix:np.ndarray,
                               function=Union[any, None],
                               function_kwargs = Optional[None]) -> np.ndarray:

    function_kwargs = function_kwargs if function_kwargs else {}
    X = np.zeros((input_matrix.shape[0], input_matrix.shape[1]))
    if function:
        for timepoint in range(input_matrix.shape[1] - 1):
            X[:, timepoint + 1] = function(adjacency_matrix @ X[:, timepoint] + input_matrix[:,timepoint],**function_kwargs)
    else:
        for timepoint in range(input_matrix.shape[1] - 1):
            X[:, timepoint + 1] = adjacency_matrix @ X[:, timepoint] + input_matrix[:,timepoint]
    return X


def lesion_simple_nodes(complements:Tuple, 
                        adjacency_matrix:np.ndarray, 
                        index:int,
                        input:np.ndarray,
                        model:Callable = simple_dynamical_system,
                        model_kwargs = None) -> np.ndarray:
    model_kwargs = model_kwargs if model_kwargs else {}

    lesioned_connectivity = deepcopy(adjacency_matrix)
    for target in complements:
        lesioned_connectivity[:,target] = 0.0
        lesioned_connectivity[target,:] = 0.0
    
    dynamics = model(lesioned_connectivity,input,**model_kwargs)
    lesioned_signal = dynamics[index]
    return lesioned_signal


def find_density(adjacency_matrix:np.ndarray) -> float:
    return np.where(adjacency_matrix!=0,1,0).sum()/adjacency_matrix.shape[0]**2


def normer(data:Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    return (data - data.min()) / (data.max() - data.min())


def spectral_normalization(target_radius:float,
                           adjacency_matrix:np.ndarray) -> np.ndarray:
    spectral_radius = np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))
    return adjacency_matrix * target_radius/spectral_radius


def threshold(lower_threshold:int,
              adjacency_matrix:np.ndarray) -> np.ndarray:
    adjacency_matrix = pd.DataFrame(adjacency_matrix)
    adjacency_matrix = adjacency_matrix.fillna(0)
    
    lower = np.percentile(adjacency_matrix, lower_threshold)
    upper = np.percentile(adjacency_matrix, 100-lower_threshold)
    adjacency_matrix[(adjacency_matrix < upper) & (adjacency_matrix > lower)] = 0.
    
    return adjacency_matrix


def event_maker(n_units:int, 
                timesteps:int,
                probability:float=1,
                rng:np.random.Generator=np.random.default_rng(seed=2023)) -> np.ndarray:
    
    if probability < 1:
        input = rng.choice([0,1],p=[1-probability,probability],size=(n_units,timesteps))
        input = input.astype(float)
    else:
        input = np.zeros((n_units,timesteps))
        for node in range(n_units):
            event_timepoints = rng.integers(0,timesteps,1)
            input[node,event_timepoints] += 1
    
    return input


def brain_plotter(data:np.ndarray,
                  coordinates:np.ndarray,
                  axis:plt.Axes,
                  view:Tuple[int,int]=(90,180),
                  size:int=20,
                  cmap:any='viridis',
                  scatter_kwargs=Optional[None]) -> plt.Axes:
    scatter_kwargs = scatter_kwargs if scatter_kwargs else {}
    
    axis.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=data,cmap=cmap,s=size,**scatter_kwargs)
    axis.view_init(*view)
    axis.axis('off')
    scaling = np.array([axis.get_xlim(),
                    axis.get_ylim(),
                    axis.get_zlim()])
    axis.set_box_aspect(tuple(scaling[:, 1]/2 - scaling[:, 0]))
    return axis


def make_influence_ratio(difference_matrix:pd.DataFrame,
                         axis:int=0) -> pd.Series:
    positives=(difference_matrix>0).sum(axis)/len(difference_matrix)
    negatives=(difference_matrix<0).sum(axis)/len(difference_matrix)
    return(positives-negatives)


def check_symmetric(adjacency_matrix:np.ndarray, tol:float=1e-8) -> bool:
    return np.all(np.abs(adjacency_matrix-adjacency_matrix.T) < tol)