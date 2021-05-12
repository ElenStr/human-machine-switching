"""
Environment help functions.
"""

import numpy as np
from copy import copy
import math
from environments.env import CELL_TYPES, TRAFFIC_LEVELS, Environment
# default width and height
WIDTH, HEIGHT = 3, 10
# default max episode length
EP_L = HEIGHT


def state2features(state, env: Environment, real_v=True):
    """
    Parameters
    ----------
    state: list of strings
        Current cell type and traffic factor and cell types of next rows 

    env: Environment
        Current environment

    real_v: bool
        Return features as real values

    Returns
    -------
    features: array of int
        The state feature vector
    """
    cell_t = np.array(CELL_TYPES+['wall'])
    traffic_l = np.array(TRAFFIC_LEVELS)
    features = [np.argwhere(cell_t == state[0])[0][0]]
    # TODO: onehot encoding for features
    if real_v:
                features[-1]+=1
                features[-1]*=.2 
    for i in range(1,len(state)):
        if (i-1) % (WIDTH+1) == 0:
            features.append(np.argwhere(traffic_l == state[i])[0][0])
            # 1 stands for not available
            if real_v: features[-1] = features[-1] + 2. 
        else:
            features.append(np.argwhere(cell_t == state[i])[0][0])
            if real_v:
                features[-1]= (features[-1] + 1.) / 10 *2
    print(features)
    return features

