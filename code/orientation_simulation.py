import numpy as np
import random

class orientation_simulation:
    def __init__(self, series, random_changes_amount, degree_change, degree_multiplicator):
        self.series = series
        self.rotated_series = self.series.copy()
        self.degree_change = degree_change
        self.random_changes_amount = random_changes_amount
        self.random_changes = np.ones((1,random_changes_amount))
        self.degree_multiplicator = degree_multiplicator
        self.angles = np.ones(series.shape) 
    
    def create_angles_random_walk(self):
        for r in range(0,3):            
            for i in range(1, self.series_length + 1):
                self.angles[r][i] = self.angles[r][i-1] + self.degree_change * random.uniform(-self.degree_multiplicator, self.degree_multiplicator)
                
    def create_angles_random_occurences(self):
        for i in range(0,self.random_changes_amount):
            self.random_changes[i] = (random.uniform(0, self.series.length), [random.uniform(-180, 180), random.uniform(-180, 180), random.uniform(-180, 180)])
            
    def create_rotation_matrix(degrees):
        radians = [degree * (np.pi / 180) for degree in degrees]            
            
        theta_rad = radians[0]
        rot_x = np.array([[1, 0, 0],
                        [0, np.cos(theta_rad), -np.sin(theta_rad)],
                        [0, np.sin(theta_rad), np.cos(theta_rad)]])

        phi_rad = radians[1]
        rot_y = np.array([[np.cos(phi_rad), 0, np.sin(phi_rad)],
                        [0, 1, 0],
                        [-np.sin(phi_rad), 0, np.cos(phi_rad)]])

        psi_rad = radians[2]
        rot_z = np.array([[np.cos(psi_rad), -np.sin(psi_rad), 0],
                        [np.sin(psi_rad), np.cos(psi_rad), 0],
                        [0, 0, 1]])
        
        rot_matrix = rot_z @ rot_y @ rot_x
        return rot_matrix
        
    def apply_rotation_random_accourences(self):
        for (index, degrees) in self.random_changes:
            rot_matrix = self.create_rotation_matrix(degrees)
            self.rotated_series[:, index:] = rot_matrix @ self.rotated_series[:, index:].T
        
    def apply_rotation_random_walk(self):
        for i in range(0, self.series.length):
            rot_matrix = self.create_rotation_matrix(self.angles[:, i])
            self.rotated_series[:, i] = rot_matrix @ self.rotated_series[:, i].T