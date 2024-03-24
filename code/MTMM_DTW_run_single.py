"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import loading as loader
import numpy as np
import MTMM_DTW as MTMMM
import evaluation as eval
import matplotlib.pyplot as plt
import orientation_simulation as orsim
import preprocessing as preproces
import os



#set up the data
l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series
scaling = False
subject = 0
exercise = 6
sensor = 1

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

def apply_rotation(time_series, rotation_file):
    simulation = orsim.orientation_simulation()
    rotation_matrix = np.load(rotation_file)

    rotated_series = simulation.apply_rotation(series=time_series, rotation_matrix=rotation_matrix)
    return rotated_series 

rotation_file_path = os.path.join("code/rotations", "rotation_gram_schmidt_5.npy")
time_series = apply_rotation(time_series=time_series, rotation_file=rotation_file_path)

preprocessor = preproces.preprocessor(series=time_series, templates=templates)
time_series = preprocessor.process()

#Use DTW to recognize every occurence of an exercise
found_exercises = MTMMM.find_exercises(templates,time_series=time_series, kabsch=True, max_iterations=1000, max_iterations_bad_match = 500, min_segment_length=0.5)
print(found_exercises)

#Load ground_truth
ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)

#Evaluate
EVAL = eval.evaluation(timeseries=time_series, templates=templates, ground_truth=ground_truth, found_exercises=found_exercises)
acc, conf = EVAL.evaluate()
print(conf)
print(acc)
EVAL.plot_simple_confusion_matrix(conf)

    