"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import numpy as np
import matplotlib.pyplot as plt

class preprocessor:
    def __init__(self, series, templates, window_size=10, minimum_low_variance_in_order=50, tolerance=0.5):
        self.series = series
        self.templates = templates
        self.window_size = window_size
        self.minimum_low_variance_in_order = minimum_low_variance_in_order
        self.tolerance = tolerance
    
    def process(self):
        self.remove_idle_parts()
        return self.series
    
    def remove_idle_parts(self):
        #self.plot_series("Before preprocessing")
        deviations = self.get_standard_deviation()
        #self.plot_deviations(deviations)
        self.remove_low_deviations(deviations)
        #self.plot_series("After preprocessing")
           
    def get_standard_deviation(self):
        deviations = []
        
        for i in range(0, len(self.series)-self.window_size, self.window_size):
            window = self.series[i:i+self.window_size]
            mean = np.mean(window)
            
            # Calculate the squared differences
            squared_diff = [(x - mean) ** 2 for x in window]

            # Calculate the variance
            variance = np.mean(squared_diff)
            deviations.append((i,i+self.window_size,variance))
            
        return deviations
    
    def remove_low_deviations(self, deviations):
        for i in range(0, len(deviations) - self.minimum_low_variance_in_order + 1):
            grouped_deviations = deviations[i:i+self.minimum_low_variance_in_order]
            
            if all(abs(grouped_deviations[j][2] - grouped_deviations[j+1][2]) <= self.tolerance for j in range(self.minimum_low_variance_in_order-1)):
                self.mask_deviation_parts(grouped_deviations)
               
    def mask_deviation_parts(self, grouped_deviations):
        (start, _, _) = grouped_deviations[0]
        (_, end, _) = grouped_deviations[self.minimum_low_variance_in_order-1]
        
        for i in range(start,end):
            self.series[i] = 0
       
                    
    def plot_series(self, title):
        plt.plot(range(0,len(self.series)), self.series)
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('acc meters')
        plt.title(title)
        plt.show()
        
    def plot_deviations(self, deviations):
        deviations = [tup[2] for tup in deviations]
        plt.plot(range(0,len(deviations)), deviations)
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Deviation')
        plt.title('Deviations of the time series')
        plt.show()
