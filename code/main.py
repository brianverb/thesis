import loading as loader
import numpy as np
import kabsch_timeseries as kabsch_time
import DTW as dtw
import evaluation as eval

#set up the data
l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series
scaling = False
subject = 0
exercise = 4
sensor = 0

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

#Use the kabsch algorithm to transform the timeseries to optimal rotated series based on the templates
transformed_series = kabsch_time(templates,time_series,scaling)

#Use DTW to recognize every occurence of an exercise
(segmented_series, segmented_series_classification_indices) = dtw(templates,transformed_series,min_length =5,max_iterations=500, max_iterations_bad_match = 25)

#Use the evaluation metrics to calculate the accuracy and confusion matrix
accuracies = eval(segmented_series, segmented_series_classification_indices, subject, exercise, sensor)
