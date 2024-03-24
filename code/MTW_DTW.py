"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import kabsch
import numpy as np
from dtaidistance import dtw_ndim
import matplotlib.pyplot as plt
from itertools import groupby

class dtw_windowed:
    def __init__(self, series, templates, scaling=False, max_distance=80, max_matches=30):
        self.series = series
        self.timeseries_length = len(series)
        
        self.templates = templates
        
        self.scaling = scaling
        
        self.max_distance = max_distance
        self.max_matches = max_matches
        
        self.found_matches = []
            
        
        self.match_overlap_allowed = 0.2
        
    def find_matches(self, k=False, steps=1):
        matches = []
        #print("Start finding matches.")
        #distances = np.load("distances.npy", allow_pickle=True)
        #distances = distances.tolist()
        for t in range(0,len(self.templates)):
            template = self.templates[t]
            template_length = len(template)
            for i in range(0,self.timeseries_length-template_length, steps):
                window = self.series[i:i+template_length]
                if(k):
                    _, R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), self.scaling)
                    window = np.dot(R, window.T).T
                    window = np.array(window)
 
                #distance = dtw_ndim.distance(window, template, penalty=10, window=(template_length//10)*2, max_dist=30,use_c=True)
                distance = dtw_ndim.distance(window, template, use_c=True)
                distance *= distance
                distance /= template_length
                
                #distances.append(distance)
                matches.append((i,i+template_length,distance,t))
            #print("Matching done for template: " + str(t+1))
        #distances = np.array(distances)
        #np.save("distances.npy", distances)
        return self.order_matches(matches)
    
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
        
    def order_matches(self, matches):
        self.ordered_matches = sorted(matches, key=lambda x: x[2])
        return self.ordered_matches
    
    
    def remove_all_overlapping_matches(self, start, end, matches):
        new_matches = []
        for (start_m, end_m, distance_m, label_m) in matches:
            overlap_length = max(0, min(end, end_m) - max(start, start_m))
            match_length = end - start
            
            if overlap_length/match_length <= self.match_overlap_allowed:
                new_matches.append((start_m, end_m, distance_m, label_m))

        return new_matches
    
    def find_exercises_max_distance(self, kabsch=False, steps=1):
        matches = self.find_matches(kabsch, steps)
        found_matches = []
        index = 0
        (start, end, distance, label) = matches[index]
        
        while distance <= self.max_distance:
            found_matches.append((start,end,label))
            matches = self.remove_all_overlapping_matches(start, end, matches)
            index +=1
            
            if index >= len(matches)-1:
                print("not enough matches left")
                break
            
            (start, end, distance, label) = matches[index]
        
        self.found_matches = found_matches       
        return found_matches
    
    def find_exercises_max_matches(self, kabsch=False, steps=1):
        matches = self.find_matches(kabsch, steps)
        index = 0
        found_matches = []
        while index < self.max_matches:
            (start, end, _, label) = matches[index]

            found_matches.append((start,end,label))
            matches = self.remove_all_overlapping_matches(start, end, matches)
            index +=1   

            if index >= len(matches)-1:
                print("not enough matches left")
                break
        
        self.found_matches = found_matches
        return found_matches
    
    def calculate_expected_matched_segments(self):
        total = 0
        for i in range(0, len(self.templates)):
            total += len(self.templates[i]) * 10
        return total
        
    def find_exercises_max_matches_expected_matched_segments(self, kabsch=False, steps=1):
        matches = self.find_matches(kabsch, steps)
        expected_matched_segments = self.calculate_expected_matched_segments()
        found_matches = []
        matched_segments = 0
        index = 0
        
        while matched_segments <= expected_matched_segments:
            (start, end, _, label) = matches[index]
            found_matches.append((start,end,label))
            
            length_of_segment = end-start
            matched_segments += length_of_segment
    
            matches = self.remove_all_overlapping_matches(start, end, matches)
            index +=1
            
            if index >= len(matches)-1:
                print("not enough matches left")
                break   
            
        self.found_matches = found_matches
        return found_matches
        