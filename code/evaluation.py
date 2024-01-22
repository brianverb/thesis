import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class evaluation:
    def __init__(self, series, ground_truth, templates, segment_percentage=0.8, exercise_percentage=0.8 ,segmented_indices=[]):
        self.series = series
        self.segmented_indices = segmented_indices
        self.length = len(series)
        self.annotated_series = np.full((self.length,1), -1)
        self.ground_truth = ground_truth
        self.ground_truth_serie = np.full((self.length,1), -1) 
        self.TP = 1
        self.TN = 1
        self.FP = 1
        self.FN = 1
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1score = None
        self.mcc = None
        self.segment_percentage = segment_percentage
        self.exercise_percentage = exercise_percentage
        self.found_truth = []
        self.templates = templates
        
    def calculate_confusion_values(self):
        self.TP = 1
        self.TN = 1
        self.FP = 1
        self.FN = 1
        for i in range(0, self.annotated_series.shape[0]):
            if(self.annotated_series[i] == -1 and self.ground_truth_serie[i] == -1):
                self.TN += 1
            if(self.annotated_series[i] == -1 and self.ground_truth_serie[i] != -1):
                self.FN += 1    
            if(self.annotated_series[i] != -1 and self.ground_truth_serie[i] == -1):
                self.FP += 1
            if(self.annotated_series[i] != -1 and self.ground_truth_serie[i] != -1 and self.annotated_series[i] == self.ground_truth_serie[i]):
                self.TP += 1
                
        print("The TP is: " + str(self.TP))
        print("The TN is: " + str(self.TN))
        print("The FP is: " + str(self.FP))
        print("The FN is: " + str(self.FN))
        
    def calculate_accuracy(self):
        self.accuracy = (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)
        print("The accuracy of the classification is: " + str(self.accuracy))
    
    def calculate_precision(self):
        self.precision = self.TP/(self.TP + self.FP)
        print("The precision of the classification is: " + str(self.precision))

    def calculate_recall(self):
        self.recall = self.TP/(self.TP+self.FN)
        print("The recall of the classification is: " + str(self.recall))

    def calculate_f1score(self):
        self.f1score = (2*(self.precision*self.recall))/(self.precision+self.recall)
        print("The f1score of the classification is: " + str(self.f1score))
        
    def calculate_mcc(self):
        self.mcc = ((self.TP * self.TN)-(self.FP * self.FN)) / math.sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN))
        print("The MCC of the classificaiton is: " + str(self.mcc))

    def calculate_confusion_matrix(self):
        labels = [-1,0, 1, 2]
        conf_matrix = confusion_matrix(self.ground_truth_serie, self.annotated_series, labels=labels)
        # Create a heatmap to visualize the confusion matrix
        '''plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()'''
        print(conf_matrix)
        
    def evaluate(self):
        self.calculate_confusion_values()
        self.calculate_accuracy()
        self.calculate_precision()
        self.calculate_recall()
        self.calculate_f1score()
        self.calculate_mcc()
        self.calculate_confusion_matrix()
        
    def annotate_timeseries(self):    
        for (start,end,label) in self.segmented_indices:
            for index in range(start,end+1):
                self.annotated_series[index] = label
                
    def annotate_ground_truth(self):

        for (start,end,label) in self.ground_truth:
            for index in range(int(start),int(end+1)):
                self.ground_truth_serie[index] = int(label)
                
    def get_segments(self):
        segments = []
        start = 0
        while start < self.length -1: 
            end = start + 1
            label = self.annotated_series[start]
            while self.annotated_series[end] == label:
                if end < self.length-1:
                    end += 1
                else:
                    break
            
            end -= 1
            segments.append((start, end, label))
            
            start = end + 1
        self.segmented_indices = segments
        return segments
    
    def clean_annotations(self):
        segments = self.get_segments()
        new_segments = []
        cleaned_segments = 0
        for index in range(0,len(segments)-2):
            (s1, e1, first_label) = segments[index]
            (s2, e2, second_label) = segments[index+1]
            (s3, e3, third_label) = segments[index+2]
            
            if first_label == third_label and second_label != first_label:
                #print("found a sandwiched segment")
                #print(str(((e1-s1) + (e3-s3))) + " / " + str(((e1-s1) + (e2-s2) + (e3-s3))) + " = " + str(((e1-s1) + (e3-s3)) / ((e1-s1) + (e2-s2) + (e3-s3))))
                if(((e1-s1) + (e2-s2) + (e3-s3)) >0):
                    if (((e1-s1) + (e3-s3)) / ((e1-s1) + (e2-s2) + (e3-s3))) > self.segment_percentage and (e2-s2) < 50:
                        #print("cleaning up segment.")
                        index += 1
                        cleaned_segments += 2
                        new_segments.append((s1,e3, first_label))
                    else:
                        new_segments.append((s1,e1,first_label))
        #print("Amount of removed excess segments: " + str(cleaned_segments))
        self.segmented_indices = new_segments
        self.annotate_timeseries()
        
        if(cleaned_segments > 0):
            self.clean_annotations()
            
    def get_exercises(self):
        self.found_truth = []
        for t in range(0,len(self.templates)):
            template_length = len(self.templates[t])
            i = 0
            while i < len(self.annotated_series)-template_length:
                count =  np.count_nonzero(self.annotated_series[i:i+template_length] == t)
                
                if count/template_length > self.exercise_percentage:
                    self.found_truth.append((i,i+template_length,t))
                    i = i + template_length
                else:  
                    i += 1
        print(self.found_truth)
        return self.found_truth
    
    def exercise_confusion_matrix(self):
        discovered = self.get_exercises()
        self.ground_truth
        conf = np.zeros((4,4))
        
        for(start_d, end_d, label_d) in discovered:
            found_match = False
            for (start_gt, end_gt, label_gt) in self.ground_truth:
                overlap_length = max(0, min(end_d, end_gt) - max(start_d, start_gt))
                length_d = end_d - start_d
                length_gt = end_gt - start_gt
                
                percentage_overlap = (overlap_length / min(length_d, length_gt)) * 100
                if percentage_overlap > self.exercise_percentage:
                    found_match = True
            if not found_match:
                conf[0, int(label_d)+1] += 1
                
        for (start_gt, end_gt, label_gt) in self.ground_truth:
            found_match = False
            for(start_d, end_d, label_d) in discovered:
                overlap_length = max(0, min(end_d, end_gt) - max(start_d, start_gt))
                length_d = end_d - start_d
                length_gt = end_gt - start_gt
                
                percentage_overlap = (overlap_length / min(length_d, length_gt))
                if percentage_overlap > self.exercise_percentage:
                    found_match = True
                    conf[int(label_gt)+1, label_d+1] += 1
            if not found_match:
                conf[int(label_gt)+1, 0] += 1

        return conf

    def exercise_accuracy(self): 
        conf = self.exercise_confusion_matrix()
        correct = 0
        false = 0
        for i in range (0,conf.shape[0]):
            for j in range(0,conf.shape[1]):
                if(i==j):
                    correct += conf[i,j]
                else:
                    false += conf[i,j]
        
        return correct / (correct + false)
    
    def plot_simple_confusion_matrix(self):
        conf = self.exercise_confusion_matrix().astype(int)
        xlabels = ["Missed Predictions", 1, 2, 3]
        ylabels = ["False Predictions", 1, 2, 3]

        # Plot confusion matrix
        sns.set(font_scale=1.2)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=xlabels,
                    yticklabels=ylabels)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def build_percentage_arrays(self):
        distances = []
        
        for t in range(0,len(self.templates)):
            template_length = len(self.templates[t])
            
            window = self.annotated_series[0:template_length]
            distance = window.count(t)
            distances.append(0, template_length, distances, t)
            
            for i in range(0,len(self.annotated_series)-template_length):
                if(self.annotated_series[i] == t):
                    distance -= 1
                if(self.annotated_series[i+template_length] == t):
                    distance += 1
                    distances.append(i, i+template_length, distances, t)
        return distances
    
    def remove_overlapping_matches(self, start_m, end_m, distances, overlap_ratio_allowed=0.05):
        distances = []
        length_match = end_m - start_m
        overlap_allowed = length_match * overlap_ratio_allowed
        
        for (start, end, distance, template) in distances:
            overlap_length = max(0, min(end, end_m) - max(start, start_m))
            
            if overlap_length < overlap_allowed:
                distances.append((start, end, distance, template))
            
        return distances
    
    def matrix_profiling_exercise_amount(self, exercise_amounts=30):
        distances = self.build_percentage_arrays()
        found_exercises = []
        found_exercises_amount = 0
        
        while found_exercises_amount < exercise_amounts:
            (start, end, distance, template)  = max(distances, key=lambda x: x[2])
            found_exercises.append((start, end, distance, template) )
            distances = self.remove_overlapping_matches(start, end, distances)
            
            if len(distances) == 0:
                break
        
    def matrix_profiling_distance_percentage(self, percentage=0.9):
        distances = self.build_percentage_arrays()
        found_exercises = []
        (start, end, distance, template) = max(distances, key=lambda x: x[2])
        
        while distance <= percentage:
            found_exercises.append((start, end, distance, template))
            (start, end, distance, template) = max(distances, key=lambda x: x[2])
            distances = self.remove_overlapping_matches(start, end, distances)
            
            if len(distances) == 0:
                break
            
        
        
        
            
    
    