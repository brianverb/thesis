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

time_series_copies = []
time_series_copies.append(time_series.copy())
time_series_copies.append(time_series.copy())
time_series_copies.append(time_series.copy())
#Use DTW to recognize every occurence of an exercise
(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,time_series_copies,min_path_length =5,max_iterations=500, max_iterations_bad_match = 25)

#Use the evaluation metrics to calculate the accuracy and confusion matrix
#accuracies = eval(segmented_series, segmented_series_classification_indices, subject, exercise, sensor)
