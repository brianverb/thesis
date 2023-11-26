import kabsch
import numpy as np
from dtaidistance import dtw_ndim
import matplotlib.pyplot as plt
from itertools import groupby

class dtw_windowed:
    def __init__(self, series, templates, annotation_margin=0, scaling=True, max_distance=25, max_matches=30):
        self.series = series
        self.series_length = len(series)
        self.templates = templates
        self.matches = []
        self.annotated_series = np.full((self.series_length,1), -1) 
        self.scaling = scaling
        self.max_distance = max_distance
        self.annotation_margin = annotation_margin
        self.max_matches = max_matches
        
    def find_matches(self, k=False, steps=1):
        print("Start finding matches.")
        for t in range(0,len(self.templates)):
            template = self.templates[t]
            template_length = len(template)
            for i in range(0,self.series_length-template_length, steps):
                window = self.series[i:i+template_length]
                if(k):
                    _, R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), self.scaling)
                    window = np.dot(window,R)
                    window = np.array(window)
 
                distance = dtw_ndim.distance(window, template, use_c=True)
                self.matches.append((i,i+template_length,distance,t))
            print("Matching done for template: " + str(t+1))
    
    def find_matches_svd(self, steps=1):
        print("Start finding matches.")
        for t in range(0,len(self.templates)):
            template = self.templates[t]
            template_length = len(template)
            principal_components = self.svd(template)
            
            for i in range(0,self.series_length-template_length, steps):
                window = self.series[i:i+template_length]
                result = self.apply_svd(window, principal_components)
                
                distance = dtw_ndim.distance(result, template,use_c=True)
                self.matches.append((i,i+template_length,distance,t))
            print("Matching done for template: " + str(t+1))
            
    def apply_svd(self, window, principal_components):
        return np.dot(window, principal_components.T)
   
    def svd(self, template):
        #mean = np.mean(template, axis=0)
        #template = template - mean
        _, _, VT = np.linalg.svd(template, full_matrices=False)
        print(VT)
        return VT 
        
    def get_distances_by_template_id(self,arr, x):
        return [tup[2] for tup in arr if len(tup) >= 4 and tup[3] == x]
    
    def get_tuples_by_template_id(self, arr, x):
        return [tup for tup in arr if len(tup) >= 4 and tup[3] == x]
    
    def plot_distances(self):     
        for i in range(0, len(self.templates)):
            data = self.get_tuples_by_template_id(self.matches,i)
            plt.plot(range(0,len(data)), data)
            # Add labels and title
            plt.xlabel('Matches')
            plt.ylabel('DTW distance')
            plt.title('DTW distances for all matches')
            plt.show()
    
    def plot_distances_points(self, ground_truths):
        for i in range(0, len(self.templates)):
            plt.close()
            plt.figure(figsize=(10, 6))
            x_cords = []
            distances = []
            distance_matches = self.get_tuples_by_template_id(self.matches,i)
            for (start,end,distance,_) in distance_matches:
                x_cords.append((start+end)/2)
                distances.append(distance)
            
            for (start,end, _) in ground_truths:
                plt.axvline(x=((start+end)/2), color='yellow', linestyle='--')
            plt.axvline(x=((start+end)/2),color='yellow', linestyle='--', label="Ground truth")
            
            plt.scatter(x_cords, distances, color='black', marker='o', label='Points')
            plt.plot(self.series[:,0], label='X_acc', color='green')
            plt.plot(self.series[:,1], label='Y_acc', color='red')
            plt.plot(self.series[:,2], label='Z_acc', color='blue')
            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel('distance and acceleration')
            plt.title('Found matches for template: ' + str(i+1))

            plt.legend()
            plt.show()


        
    def order_matches(self):
        self.ordered_matches = sorted(self.matches, key=lambda x: x[2])
    
    def annotate_series_max_distance(self):
        for (start, end, distance, label) in self.ordered_matches:
            if(distance <=self.max_distance):
                length_of_segment = end-start
                start_margined = start + int(length_of_segment*self.annotation_margin//2)
                end_margined = end - int(length_of_segment*self.annotation_margin//2)

                for index in range(start_margined,end_margined+1):
                    if(self.annotated_series[index] == -1):
                        self.annotated_series[index] = label
            else:
                break
    
    def annotate_series_max_matches(self):
        index = 0
        while index <= self.max_matches:
            (start, end, _, label) = self.ordered_matches[index]
            length_of_segment = end-start
            start_margined = start + int(length_of_segment*self.annotation_margin//2)
            end_margined = end - int(length_of_segment*self.annotation_margin//2)

            for index in range(start_margined,end_margined+1):
                if(self.annotated_series[index] == -1):
                    self.annotated_series[index] = label
            index +=1   
            
    def annotate_series_max_matches_expected_matched_segments(self):
        expected_matched_segments = 0
        for i in range(0, len(self.templates)):
            expected_matched_segments += len(self.templates[i]) * 10
        matched_segments = 0
        index = 0
        print(expected_matched_segments)
        while matched_segments <= expected_matched_segments:
            (start, end, _, label) = self.ordered_matches[index]
            
            length_of_segment = end-start
            start_margined = start + int(length_of_segment*self.annotation_margin//2)
            end_margined = end - int(length_of_segment*self.annotation_margin//2)

            for indice in range(start_margined,end_margined+1):
                if(self.annotated_series[indice] == -1):
                    matched_segments +=1
                    self.annotated_series[indice] = label
            index +=1   

        