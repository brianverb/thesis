import loading as loader
import numpy as np
import kabsch_timeseries as kabsch_time
import DTW as dtw
import evaluation as eval
import pandas as pd
import csv

#set up the data
l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series
scaling = False
subject = 2
exercise = 0
sensor = 1

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

#Use the kabsch algorithm to transform the timeseries to optimal rotated series based on the templates
transformed_series = kabsch_time.transform(templates,time_series,scaling)

# Save the array to a CSV file
with open('transformed_series.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for plane in transformed_series:
        writer.writerows(plane)

#Use DTW to recognize every occurence of an exercise
(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,transformed_series,min_path_length =5,max_iterations=1000, max_iterations_bad_match = 50)

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)
MTMM_DTW_EVAL = eval.evaluation(series=time_series, segmented_indices=segmented_series_classification_indices, ground_truth=ground_truth)
MTMM_DTW_EVAL.annotate_timeseries()
MTMM_DTW_EVAL.annotate_ground_truth()
MTMM_DTW_EVAL.evaluate()
