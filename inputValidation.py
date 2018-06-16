import numpy as np
import sys
import os

numbertypes = [np.float32, np.float64, np.float16, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, int, float]
vectortypes = [np.ndarray, list]

def check_if_number(thing):

    for elem in numbertypes:
        if type(thing) is elem:
            return True
    return False

def check_if_vector(thing, dimension):
    if dimension > 0:
        for t in vectortypes:
            if type(thing) == t:
                for elem in thing:
                    if not check_if_vector(elem, dimension-1):
                        return False
                return True
        return False
    elif dimension == 0:
        return check_if_number(thing)
