import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class evaluation:
    def __init__(self, series, ground_truth ,segmented_indices=[]):
        self.series = series
        self.segmented_indices = segmented_indices
        self.length = len(series)
        self.annotated_series = np.full((self.length,1), -1)
        self.ground_truth = ground_truth
        self.ground_truth_serie = np.full((self.length,1), -1) 
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1score = None
        self.mcc = None
        self.segment_percentage = 0.8
        
    def calculate_confusion_values(self):
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
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
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
                print("found a sandwiched segment")
                print(str(((e1-s1) + (e3-s3))) + " / " + str(((e1-s1) + (e2-s2) + (e3-s3))) + " = " + str(((e1-s1) + (e3-s3)) / ((e1-s1) + (e2-s2) + (e3-s3))))
                if (((e1-s1) + (e3-s3)) / ((e1-s1) + (e2-s2) + (e3-s3))) > self.segment_percentage and (e2-s2) < 50:
                    print("cleaning up segment.")
                    index += 1
                    cleaned_segments += 2
                    new_segments.append((s1,e3, first_label))
                else:
                    new_segments.append((s1,e1,first_label))
        print("Amount of removed excess segments: " + str(cleaned_segments))
        self.segmented_indices = new_segments
        self.annotate_timeseries()
        
        if(cleaned_segments > 0):
            
            self.clean_annotations()
        
              
            
            