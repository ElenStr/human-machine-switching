"""
Environment help functions.
"""

import numpy as np
from copy import copy
import math
from environments.env import CELL_TYPES, TRAFFIC_LEVELS
# default width and height
WIDTH = 3

def feature2onehot(value, n_onehot):
    """
    Parameters
    ----------
    value: int 
        Scalar feature value

    n_onehot: int
        Number of possible feature values

    Returns
    -------
    f_v: list
        The onehot represantation of value
    """
    f_v = [0.]*n_onehot
    f_v[n_onehot - value - 1] = 1.
    return f_v

def state2features(state, n_features, real_v=False):
    """
    Parameters
    ----------
    state: list of strings
        Current cell type and traffic factor and cell types of next rows 

    n_features: int
        Number of features required by the network
    
    real_v: bool
        Return features as real values

    Returns
    -------
    features: list of int
        The state feature vector
    """
    

    cell_t = np.array(CELL_TYPES+['wall'])
    traffic_l = np.array(TRAFFIC_LEVELS)
    features = []
    feature = np.argwhere(cell_t == state[0])[0][0]
    # TODO: onehot encoding for features
    if real_v:
        features.append((feature + 1)*0.2)
    else:
        features.extend(feature2onehot(feature, cell_t.size - 1))

    for i in range(1,n_features):
        if (i-1) % (WIDTH+1) == 0:
            state_i =  state[i] if i <= len(state)-1 else 'not-available'
            feature = np.argwhere(traffic_l == state_i)
            # add value for no traffic for last states
            feature = traffic_l.size if not feature.size else feature[0][0]
            # 1 stands for not available
            if real_v: 
                features.append(feature + 2.)
            else:
                features.extend(feature2onehot(feature, traffic_l.size + 1))              


        else:
            state_i =  state[i] if i <= len(state)-1 else 'wall'
            feature = np.argwhere(cell_t == state_i)[0][0]
            if real_v:
                features.append((feature + 1.) * 0.2)
            else:
                features.extend(feature2onehot(feature, cell_t.size)) 


    return features

