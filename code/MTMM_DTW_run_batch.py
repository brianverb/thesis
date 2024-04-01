"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import loading as loader
import numpy as np
import MTMM_DTW as MTMMM
import evaluation as eval
import orientation_simulation as orsim
import preprocessing as prep

def run(subject, exercise, unit, rotation_file, kabsch):
    templates, time_series, ground_truth = load_data(subject, exercise, unit)

    time_series = apply_rotation(time_series, rotation_file)
    
    preprocessor = prep.preprocessor(series=time_series, templates=templates)
    time_series = preprocessor.process()
    if kabsch:
        found_exercises = MTMMM.find_exercises(templates,time_series=time_series, kabsch=kabsch, min_segment_length=0.5, max_distance=1)
    else:
        found_exercises = MTMMM.find_exercises(templates,time_series=time_series, kabsch=kabsch, min_segment_length=0.7, max_distance=0.9)

    accuracy, conf, amount_of_expected_exercises = evaluate_time_series(time_series=time_series, templates=templates, found_exercises=found_exercises, ground_truth=ground_truth)
    
    return accuracy, conf, amount_of_expected_exercises
    
def load_data(subject, exercise, unit):
    l = loader.Loading("code\data")
    l.load_all()
    subjects = l.time_series

    templates, time_series = subjects[subject][exercise][unit]
    ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)

    return templates, time_series, ground_truth

def apply_rotation(time_series, rotation_file):
    simulation = orsim.orientation_simulation()
    rotation_matrix = np.load(rotation_file)

    rotated_series = simulation.apply_rotation(series=time_series, rotation_matrix=rotation_matrix)
    return rotated_series 
     
def evaluate_time_series(time_series, templates, ground_truth, found_exercises): 
    EVAL = eval.evaluation(timeseries=time_series, templates=templates, ground_truth=ground_truth, found_exercises=found_exercises)
    acc, conf = EVAL.evaluate()
    amount_of_gt = len(ground_truth)
    
    return acc, conf, amount_of_gt