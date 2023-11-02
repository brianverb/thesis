import numpy as np
import random

class orientation_simulation:
    def __init__(self, series, degree):
        self.series = series
        self.series_length = len(series)
        self.degree = degree
        self.angles = np.ones((self.series_length,1)) 
    
    def simulate(self):
        self.angles[0] = self.degree * random.uniform(-180, 180)
        for i in range(1, self.series_length + 1):
            self.angles[i] = self.angles[i-1] + self.degree * random.choice([-1, 0, 1])
            #Todo: Rotate based on the angles