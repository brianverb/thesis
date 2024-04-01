"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

class evaluation:
    def __init__(self, timeseries=None, ground_truth=None, templates=None ,found_exercises=[]):
        self.timeseries = timeseries
        self.timeseries_length = len(timeseries)
        self.annotated_series = np.full((self.timeseries_length,1), -1)
        
        self.ground_truth = ground_truth
        self.ground_truth_series = np.full((self.timeseries_length,1), -1) 
        
        self.templates = templates
                
        self.found_exercises = found_exercises

    
    def annotate_timeseries(self):    
        for (start,end,label) in self.found_exercises:
            for index in range(start,end+1):
                self.annotated_series[index] = label
                
    def annotate_ground_truth(self):
        for (start,end,label) in self.ground_truth:
            for index in range(int(start),int(end+1)):
                self.ground_truth_series[index] = int(label)
    
    def plot_simple_confusion_matrix(self, found_truth):
        xlabels = [1, 2, 3, "MD"]
        ylabels = [1, 2, 3, "FD"]

        # Plot confusion matrix
        sns.set_theme(font_scale=1.2)
        plt.figure(figsize=(8, 6))
        sns.heatmap(found_truth, annot=True, fmt='d', cmap='Blues',
                    xticklabels=xlabels,
                    yticklabels=ylabels)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Ground truth')
        plt.show()        
          
    def remove_overlapping_matches(self, start_m, end_m, old_percentages, overlap_ratio_allowed=0.05):
        new_percentages = []
        length_match = end_m - start_m
        overlap_allowed = length_match * overlap_ratio_allowed

        
        for (start, end, distance, template) in old_percentages:
            
            overlap_length = max(0, min(end, end_m) - max(start, start_m))
    
            if overlap_length < overlap_allowed:
                new_percentages.append((start, end, distance, template))
            
        return new_percentages
    
    def overlap_matrix(self):
        overlap_matrix = np.zeros((len(self.ground_truth), len(self.found_exercises)))
        index_gt=0
        for (start_gt, end_gt,_) in self.ground_truth:
            index_d = 0
            for (start_d, end_d,_) in self.found_exercises:
                overlap_length = max(0, min(end_d, end_gt) - max(start_d, start_gt))
                if(overlap_length > 0):   
                    overlap_matrix[index_gt, index_d] = 1 - (overlap_length / (max(end_d, end_gt) - min(start_d, start_gt)))
                else:
                    overlap_matrix[index_gt, index_d] = 100
                index_d +=1
            index_gt +=1
        return overlap_matrix
     
    def make_square(self, matrix):
        num_rows, num_cols = matrix.shape
        
        if num_rows < num_cols:
            num_dummy_rows = num_cols - num_rows
            dummy_rows = np.full((num_dummy_rows, num_cols), 100)
            matrix = np.concatenate((matrix, dummy_rows), axis=0)
            
        if num_cols < num_rows:
            num_dummy_cols = num_rows - num_cols
            dummy_cols = np.full((num_rows, num_dummy_cols), 100)      
            matrix = np.concatenate((matrix, dummy_cols), axis=1)
        return matrix
            
    def match_overlaps(self, matrix):
        cost_matrix = self.make_square(matrix)
        matched = []
        not_matched = []
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_indices, col_indices):
            #print(cost_matrix[row, col].sum())
            if(cost_matrix[row, col].sum() <100):
                matched.append((row,col))
                #print("Row {} -> Column {}".format(row, col))
            else:
                not_matched.append((row,col))
        return matched, not_matched
    
    def increment_value_in_file(self, file_path, value):
        array = np.load(file_path, allow_pickle=True)
        array = array.tolist()  
        array.append(value)
        array = np.array(array)
        np.save(file_path, array)
        
    def evaluate(self):
        overlap_matrix = self.overlap_matrix()
        matched, non_matched = self.match_overlaps(overlap_matrix)
        confusion_matrix = np.zeros((4,4), int)
        
        for (row, col) in matched:
            (_,_,label_gt) = self.ground_truth[row]
            label_gt = int(label_gt)
            (start,end,label_d) = self.found_exercises[col]
            confusion_matrix[label_gt, label_d] +=1
            '''
            length = (end-start)
            if(label_d == label_gt):
                self.increment_value_in_file("correct.npy",length)
            else:
                self.increment_value_in_file("miss.npy", length)
            '''


        for (row, col) in non_matched:

            if(row <= len(self.ground_truth)-1):
                (start,end,label_gt) = self.ground_truth[row]
                label_gt = int(label_gt)
                confusion_matrix[label_gt, 3] +=1
                #print("missed prediction-> start:{} end:{} label:{}".format(start,end,label_gt))
            if(col <= len(self.found_exercises)-1):
                (start,end,label_d) = self.found_exercises[col]
                confusion_matrix[3, label_d] += 1
                length = (end-start)
                #print("false prediciton-> start:{} end:{} label:{}".format(start,end,label_d))
                #self.increment_value_in_file("false.npy", length)
        
        accuracy = self.calculate_accuracy(conf=confusion_matrix)
        return accuracy, confusion_matrix

    def calculate_accuracy(self, conf): 
        correct = 0
        false = 0
        for i in range (0,conf.shape[0]):
            for j in range(0,conf.shape[1]):
                if(i==j):
                    correct += conf[i,j]
                else:
                    false += conf[i,j]
        
        return correct / (correct + false)
                        
        


        
        

        
    
    