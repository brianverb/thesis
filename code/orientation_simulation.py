import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal

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
    

    def plot_3d_vecs(self, vs):
        if type(vs) == list:
            vs = np.array(vs)
            
        fig = plt.figure(figsize=(10,10))

        self._plot_2d_arrows(vs, 1)
        self._plot_2d_histogram(vs, 2)
        self._plot_3d_points(vs, 3)
        
        plt.tight_layout()

        plt.show()
        
    def _plot_2d_arrows(self, vs, plot_ix):
        # Plot arrows for the first 1000 or so, projected in 2d
        ax = plt.subplot(1, 3, plot_ix)
        ax.set_title('100 vecs plotted in 3D')
        
        ax.set_aspect('equal', adjustable='box')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        [ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.1) for (x,y,z) in vs[:100]]
        
    def _plot_2d_histogram(self, vs, plot_ix):
        # Histogram of angles, also projected in 2d
        ax = plt.subplot(1, 3, plot_ix, projection='polar')
        ax.set_title('Histogram of angles about z axis')
        
        angles = [np.arctan2(y, x) for (x,y,z) in vs]
        ax.hist(angles, bins=100)

    def _plot_3d_points(self, vs, plot_ix):
        # Plot the points on a sphere.
        ax = plt.subplot(1, 3, plot_ix, projection='3d')
        ax.set_title('300 pts shown on a sphere')

        # Plot a sphere
        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        ax.plot_surface(
            x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

        # Plot data
        data = vs[:300]*1.05
        xx, yy, zz = np.hsplit(data, 3) 
        ax.scatter(xx, yy, zz, color="k", s=20)

        # Matplotlib actually sucks - removed set_axis equal for 3d plots... so have to make
        # spheres look roughly spherical by this bs now ;-;
        r = 1.3
        ax.set_xlim([-r,r])
        ax.set_ylim([-r,r])
        ax.set_zlim([-1,1])

    def gram_schmidt(self, A):
        for k in range(0, A.shape[1]):
            # Normalize
            A[:, k] /= np.linalg.norm(A[:, k])
            # Subtract from vecs below
            for j in range(k+1, A.shape[1]):
                r = np.dot(A[:, k], A[:, j])
                A[:, j] -= r*A[:, k]
        return A
        
    # Change the handedness of the matrix. Converts from S(3) to SO(3)
    def make_left(self, A):
        # If mat is right handed...
        if np.cross(A[0], A[1]).dot(A[2]) < 0:
            # Swap first two rows
            A[[0,1]] = A[[1,0]]
        return A

    def get_random_rotation_matrix(self):
        m = normal(size=(3,3))  # Random mat w/ normal distribution components
        m = self.gram_schmidt(m)     # Random rotation mat?
        #m = scipy.linalg.orth(m) # Uses SVD: Not uniform!
        
        m = self.make_left(m)

        self.plot_3d_vecs(m)
        return m