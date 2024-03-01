import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class evaluation:
    def __init__(self, series, ground_truth, templates, segment_percentage=0, exercise_percentage=0.6 ,segmented_indices=[]):
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
        #print(segments)
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
                    #print("voldoet aan eerste vergelijking")
                    if (((e1-s1) + (e3-s3)) / ((e1-s1) + (e2-s2) + (e3-s3))) > self.segment_percentage and (e2-s2) < 50:
                        test = ((e1-s1) + (e3-s3)) / ((e1-s1) + (e2-s2) + (e3-s3))
                        #print(str(test) + " voldoet aan 2de vergelijking")
                        #print("cleaning up segment.")
                        index += 1
                        cleaned_segments += 2
                        new_segments.append((s1,e3, first_label))
                    else:
                        test = ((e1-s1) + (e3-s3)) / ((e1-s1) + (e2-s2) + (e3-s3))
                        #print(str(test) + " voldoet NIET aan 2de vergelijking")
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
                    self.found_truth.append((i, i+template_length, 0, t))
                    i = i + template_length
                else:  
                    i += 1
        print(self.found_truth)
        return self.found_truth
    
    def exercise_confusion_matrix(self):
        discovered = self.found_truth
        conf = np.zeros((4,4))
        #print(self.ground_truth)
        for(start_d, end_d, label_d) in discovered:
            found_match = False
            for (start_gt, end_gt,label_gt) in self.ground_truth:
                overlap_length = max(0, min(end_d, end_gt) - max(start_d, start_gt))
                length_d = end_d - start_d
                
                percentage_overlap = (overlap_length / length_d) 
                if percentage_overlap > self.exercise_percentage:
                    found_match = True
        
            if not found_match:
                conf[0, int(label_d)+1] += 1
                
        for (start_gt, end_gt, label_gt) in self.ground_truth:
            found_match = False
            for(start_d, end_d, label_d) in discovered:
                overlap_length = max(0, min(end_d, end_gt) - max(start_d, start_gt))
                length_gt = end_gt - start_gt
                
                percentage_overlap = (overlap_length / length_gt) 
                if percentage_overlap > self.exercise_percentage:
                    found_match = True
                    conf[int(label_gt)+1, label_d+1] += 1
            if not found_match:
                conf[int(label_gt)+1, 0] += 1
           

        return conf

    def exercise_accuracy(self, conf): 
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
        percentages = []
  
        for t in range(0,len(self.templates)):
            template_length = len(self.templates[t])
            matching_labels = 0
            
            for i in range(0,template_length):
                if t == self.annotated_series[i]:
                    matching_labels +=1
            distance = matching_labels / template_length
            percentages.append((i, i+template_length, distance, t))
            
            for i in range(1,len(self.annotated_series)-template_length):
                if(self.annotated_series[i-1] == t):
                    matching_labels -= 1
                if(self.annotated_series[i+template_length] == t):
                    matching_labels += 1
                distance = matching_labels / template_length
                percentages.append((i, i+template_length, distance, t))
        
        #self.plot_percentages(percentages)
        return percentages
    
    def plot_percentages(self, percentages):
        for t in range(0,len(self.templates)):
            percentages_of_template_x = list(filter(lambda item: item[3] == t, percentages))
            distances_of_template_x = [item[2] for item in percentages_of_template_x]

            plt.plot(distances_of_template_x, label='Percentages of label: ' + str(t), color='red')
            plt.show()
            
    def plot_found_truth(self):    
        data = np.full(self.annotated_series.shape, 0)
        for (start,end,label) in self.found_truth:
            for index in range(start,end+1):
                data[index] = label
        plt.plot(data, label='found exercises: ' , color='red')
        plt.show()
                
    def remove_overlapping_matches(self, start_m, end_m, old_percentages, overlap_ratio_allowed=0.1):
        new_percentages = []
        length_match = end_m - start_m
        overlap_allowed = length_match * overlap_ratio_allowed

        
        for (start, end, distance, template) in old_percentages:
            
            overlap_length = max(0, min(end, end_m) - max(start, start_m))
    
            if overlap_length < overlap_allowed:
                new_percentages.append((start, end, distance, template))
            
        return new_percentages
    
    def matrix_profiling_exercise_amount(self, exercise_amounts=30):
        percentages = self.build_percentage_arrays()
        found_exercises = []
        found_exercises_amount = 0
        
        while found_exercises_amount < exercise_amounts:
            (start, end, percentage, template)  = max(percentages, key=lambda x: x[2])
            #print("Found best match: " + str(start) + " , " + str(end) + " , " + str(percentage) + " , " + str(template) )
            found_exercises.append((start, end, template) )
            percentages = self.remove_overlapping_matches(start, end, percentages)
            
            print(len(percentages))
            if len(percentages) == 0:
                #print("breaking bcs of no matches left")
                break
            found_exercises_amount +=1
            
        self.found_truth = found_exercises
        #self.plot_found_truth()
        return found_exercises
        
    def matrix_profiling_distance_percentage(self, percentage_threshold=0.8):
        percentages = self.build_percentage_arrays()
        found_exercises = []
        percentage=1
        while percentage >= percentage_threshold:
            (start, end, percentage, template)  = max(percentages, key=lambda x: x[2])
            found_exercises.append((start, end, template) )
            percentages = self.remove_overlapping_matches(start, end, percentages)
            
            if len(percentages) == 0:
                break
            
        self.found_truth = found_exercises
        #self.plot_found_truth()
        #print(found_exercises)
        return found_exercises

    def overlap_matrix(self):
        overlap_matrix = np.zeros((len(self.ground_truth), len(self.found_truth)))
        index_gt=0
        for (start_gt, end_gt,_) in self.ground_truth:
            index_d = 0
            for (start_d, end_d,_) in self.found_truth:
                overlap_length = max(0, min(end_d, end_gt) - max(start_d, start_gt))
                if(overlap_length > 0):   
                    overlap_matrix[index_gt, index_d] = overlap_length / (max(end_d, end_gt) - min(start_d, start_gt))
                index_d +=1
            index_gt +=1
        return overlap_matrix
        
    def choose_matrix_matches(self, matrix):
        truth_matrix = np.zeros(matrix.shape, dtype=bool)
        while not np.all(matrix == 0):
            max_index = np.argmax(matrix)
            max_row, max_col = divmod(max_index, matrix.shape[1])
            truth_matrix[max_row, max_col] = True
            matrix[:, max_col] = 0
            matrix[max_row,:] = 0
        return truth_matrix
    
    def create_confusion_matrix_with_assignmentproblem(self):
        overlap_matrix = self.overlap_matrix()
        matrix = self.choose_matrix_matches(overlap_matrix)
        confusion_matrix = np.zeros((4,4))
        discovered_truth_not_matched = list(range(0, len(self.found_truth)))
        for row_idx, row in enumerate(matrix):
            flag = False
            for col_idx, value in enumerate(row):
                (_,_,label_gt) = self.ground_truth[row_idx]
                label_gt = int(label_gt) +1
                if value:
                    (_,_,label_d) = self.found_truth[col_idx]
                    label_d += 1
                    confusion_matrix[label_gt, label_d] +=1
                    discovered_truth_not_matched.remove(col_idx)
                    flag = True
            if not flag:
                confusion_matrix[label_gt, 0] +=1
            #print("Exercise in gt: " + str(row_idx) + " is: " + str(flag) + "conf: " + str(confusion_matrix))
        
        for false_prediction in discovered_truth_not_matched:
            (_,_,label_d) = self.found_truth[false_prediction]
            confusion_matrix[0, label_d+1] += 1
        
        return confusion_matrix

                    
                        
        


        
        

        
    
    