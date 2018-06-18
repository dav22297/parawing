import numpy as np
import sys
import os

number_types = [np.float32, np.float64, np.float16, np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64, int,
                float]
vector_types = [np.ndarray, list]


def check_if_number(number):

    if type(number) in number_types:
        return True
    return False


def check_if_vector(vector, dimension):

    if type(vector) in vector_types:
        for component in vector:
            if not check_if_number(component):
                return False
        if len(vector) == dimension:
            return True
    return False

