import loading as loader
import numpy as np
import kabsch_timeseries as kabsch_time
import DTW as dtw
import evaluation as eval
import pandas as pd
import csv

#set up the data
l = loader.Loading("code\data")
l.load_all_id()
subjects = l.time_series
scaling = False
subject = 0
exercise = 4
sensor = 0

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

#Use the kabsch algorithm to transform the timeseries to optimal rotated series based on the templates
transformed_series = kabsch_time.transform(templates,time_series,scaling)

for i in range(0,3):
    transformed_series[i] = transformed_series[i][:, 1:]
    templates[i] = templates[i][:, 1:]
    
# Save the array to a CSV file
with open('transformed_series_kabsch_numbered.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for plane in transformed_series:
        writer.writerows(plane)

print(transformed_series[0])

#Use DTW to recognize every occurence of an exercise
(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,transformed_series,min_path_length =5,max_iterations=500, max_iterations_bad_match = 25)

#Use the evaluation metrics to calculate the accuracy and confusion matrix
#accuracies = eval(segmented_series, segmented_series_classification_indices, subject, exercise, sensor)
