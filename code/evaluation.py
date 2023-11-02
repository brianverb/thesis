import numpy as np
import math

class evaluation:
    def __init__(self, series, segmented_indices, ground_truth):
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
        
    def calculate_confusion_values(self):
        for i in range(0, self.annotated_series.shape[0]):
            if(self.annotated_series[i] != -1):
                if(self.annotated_series[i] == self.ground_truth_serie[i]):    
                    self.TP += 1
                else:
                    self.TN += 1
            else:
                if(self.ground_truth_serie[i] == -1):
                    self.FP +=1
                else:
                    self.FN += 1
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

    def evaluate(self):
        self.annotate_timeseries()
        self.calculate_confusion_values()
        self.calculate_accuracy()
        self.calculate_precision()
        self.calculate_recall()
        self.calculate_f1score()
        self.calculate_mcc()
        
    def annotate_timeseries(self):    
        for (start,end,label) in self.segmented_indices:
            for index in range(start,end):
                self.annotated_series[index] = label
        for (start,end,label) in self.ground_truth:
            for index in range(int(start),int(end)):
                self.ground_truth_serie[index] = int(label)