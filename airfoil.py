import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class Airfoil(object):
    def __init__(self, aoa=0, contour=[], normal_vector=[0,0,1], name=''):
        # check the type of the input
        if type(contour) is not list or type(contour) is not np.ndarray:
            raise TypeError('The contour has to be a tuple of real numbers')
        if type(aoa) is not float or type(aoa) is not np.float:
            raise TypeError('The angle of attack has to be a real number')
        if type(normal_vector) is not list or type(normal_vector) is not np.ndarray or type(normal_vector) is not tuple:
            raise TypeError('The normal vector is a tuple/list of three real numbers')
            if len(normal_vector) != 3:
                raise TypeError('The normal vector has to be of dimension 3')
        if type(name) is not str:
            raise TypeError('The name of the Airfoil has to be a string')
        # assign the input to the variables
        self.aoa = aoa
        self.contour = contour
        self.normal_vector = normal_vector
        self.name = name

    def import_xfoil_airfoil(self, filepath):
        if not os.path.exists(filepath):
            raise ImportError('The filepath does not lead to a valid File')
        try:
            self.contour = np.loadtxt(filepath, dtype=(float, float), comments='#', skiprows=1)
            with open(filepath, r) as f:
                self.name = f.readline()
        except BaseException:
            raise ImportError('The File given cannot be interpreted correctly')


