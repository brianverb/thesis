import kabsch
import numpy as np
from dtaidistance import dtw_ndim

class dtw_windowed:
    def __init__(self, series, templates, scaling=False, max_distance=25):
        self.series = series
        self.series_length = len(series)
        self.templates = templates
        self.matches = []
        self.annotated_series = np.full((self.series_length,1), -1) 
        self.scaling = scaling
        self.max_distance = max_distance
    
    def find_matches(self, k=False, steps=1):
        for t in range(0,len(self.templates)):
            template = self.templates[t]
            template_length = len(template)
            for i in range(0,self.series_length-template_length, steps):
                window = self.series[i:i+template_length]
                if(k):
                    _, R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), self.scaling)
                    window = np.dot(window,R)
                print(window.shape)
                print(template.shape)
                distance = dtw_ndim.distance(window, template)
                self.matches.append((i,i+template_length,distance,t))
            print("Matching done for template: " + str(t+1))
    
    def order_matches(self):
        self.matches = sorted(self.matches, key=lambda x: x[2])
    
    def annotate_series(self):
         for (start, end, distance, label) in self.matches:
            if(distance <=self.max_distance):
                for index in range(start,end+1):
                    if(self.annotated_series[index] == -1):
                        self.annotated_series[index] = label
            else:
                break