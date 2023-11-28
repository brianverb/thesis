import numpy as np
import random

class orientation_simulation:
    def __init__(self, series, random_changes_amount=0, degree_change=0, degree_multiplicator=0):
        self.series = series
        self.rotated_series = self.series.copy()
        self.degree_change = degree_change
        self.random_changes_amount = random_changes_amount
        self.random_changes = []
        self.degree_multiplicator = degree_multiplicator
        self.angles = np.ones(series.shape) 
    
    def reset_angles(self):
        self.angles = np.ones(self.series.shape) 

    def create_uniform_random_rotation(self):
        x_angle = random.randint(-180, 180)
        y_angle = random.randint(-180, 180)
        z_angle = random.randint(-180, 180)

        self.angles =  np.array([np.full(self.series.shape[0], x_angle), np.full(self.series.shape[0], y_angle), np.full(self.series.shape[0], z_angle)])
        
        self.apply_single_rotation(x_angle, y_angle, z_angle)
        return x_angle, y_angle, z_angle, self.rotated_series
        
    
    def create_angles_random_walk(self):
        for r in range(0,2):            
            for i in range(1, len(self.series)):
                self.angles[i,r] = self.angles[i-1,r] + self.degree_change * random.randint(-self.degree_multiplicator, self.degree_multiplicator)
                
    def create_angles_random_occurences(self):
        for i in range(0,self.random_changes_amount):
            self.random_changes.append((random.randint(0, len(self.series)), [random.randint(-180, 180), random.randint(-180, 180), random.randint(-180, 180)]))
        for (index, degrees) in self.random_changes:           
            for i in range(index, len(self.series)):
                self.angles[i] = self.angles[i] + degrees
            
    def create_rotation_matrix(self, degrees):
        radians = [degree * np.pi / 180 for degree in degrees]            
            
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
            self.rotated_series[-index:, :] = (rot_matrix @ self.rotated_series[-index:, :].T ).T
        
    def apply_rotation(self):
        for i in range(0, len(self.series)):
            rot_matrix = self.create_rotation_matrix(self.angles[i, :])
            self.rotated_series[i, :] = (rot_matrix @ self.rotated_series[i, :].T ).T

    def apply_single_rotation(self, x_angle, y_angle, z_angle):
        rot_matrix = self.create_rotation_matrix([x_angle, y_angle, z_angle])
        self.rotated_series = (rot_matrix @ self.rotated_series.T ).T
        print(rot_matrix)