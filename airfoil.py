import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class Airfoil(object):
    def __init__(self, aoa=0, contour=[], normal_vector=[0,0,1], name='', rotation_angle=0):
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
        if type(rotation_angle) is not float:
            return TypeError('The rotation Angle has to be a real number')

        # assign the input to the variables
        self.aoa = aoa
        self.contour = contour
        self.normal_vector = normal_vector
        self.name = name
        self.rotation_angle = rotation_angle
        self.rotated_to = rotation_angle

    def import_xfoil_airfoil(self, filepath):
        if not os.path.exists(filepath):
            raise ImportError('The filepath does not lead to a valid File')

        try:
            self.contour = np.loadtxt(filepath, dtype=(float, float), comments='#', skiprows=1)
            with open(filepath, r) as f:
                self.name = f.readline()
        except BaseException:
            raise ImportError('The File given cannot be interpreted correctly')

    def set_normal(self, normal_vector):
        types = [np.ndarray, tuple, list]
        for t in types:
            if type(normal_vector) is not t:
                raise TypeError('The normal vector has to be either a list, tuple, or numpy array')
        if len(normal_vector) is not 3:
            raise TypeError('The normal vector has to be three dimensional')
        self.normal_vector = normal_vector

    def set_rotation_angle(self, theta, unit='deg'):
        if type(theta) is not float:
            raise TypeError('The rotation angle has to be a real number')
        if unit == 'deg':
            theta = theta * np.pi/180
        elif unit != 'rad':
            raise ValueError('The unit of the rotation angle has to be either in "rad" or "deg"')
        self.rotation_angle = theta

    def rotate(self, theta):
        if type(theta) is not float:
            raise TypeError('The rotation angle has to be a float')
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.contour = [np.matmul(rotation_matrix, point) for point in self.contour]
        self.rotated_to += theta

    def to3D(self, All_DOF=True):
        if type(All_DOF) is not bool:
            return TypeError('The All_DOF parameter has to be a boolean')
        # determin if the aoa is set correctly
        DeltaRot = self.rotation_angle - self.rotated_to
        if DeltaRot != 0:
            self.rotate(DeltaRot)
        # determin X-rotation angle from the normal vector
        phiV = self.normal_vector * np.array([1,0,1])
        phi = np.dot(phiV, np.array([[1,0,0]]))/np.linalg.norm(phiV)
        # rotate around the X-achsis (the longitudinal axis)
        rotation_matrix = np.array([[1,0,0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        if All_DOF:
            # determin Y-rotation angle from the normal vector
            thetaV = self.normal_vector * np.array([0,1,1])
            theta = np.dot(thetaV, np.array([0,0,1])) / np.linalg.norm(thetaV)
            # apply the rotation to the rotational Matrix
            y_rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            rotation_matrix = np.matmul(y_rotation, rotation_matrix)
        return [np.matmul(rotation_matrix, np.array([point[0], point[1], 0])) for point in self.contour]


