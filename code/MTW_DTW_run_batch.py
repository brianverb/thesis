"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import loading as loader
import MTW_DTW as dtw
import evaluation as eval
import orientation_simulation as orsim
import numpy as np
import preprocessing as prep

def run(subject, exercise, unit, rotation_file, kabsch):
    templates, time_series, ground_truth = load_data(subject, exercise, unit)

    preprocessor = prep.preprocessor(series=time_series, templates=templates)
    time_series = preprocessor.process()

    time_series = apply_rotation(time_series, rotation_file)

    found_exercises = mtw_dtw(time_series, templates, kabsch)
    
    accuracy, conf, amount_of_expected_exercises = evaluate_time_series(time_series, templates, ground_truth, found_exercises)
    
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
    
def mtw_dtw(time_series, templates, kabsch):
    #Use DTW to recognize every occurence of an exercise
    DTW = dtw.dtw_windowed(series=time_series, templates=templates, scaling=False, max_distance=17.5, max_matches=30)
    found_exercises = DTW.find_exercises_max_matches(kabsch=kabsch, steps=5)
    
    return found_exercises

def evaluate_time_series(time_series, templates, ground_truth, found_exercises):
    MTW_DTW_EVAL = eval.evaluation(timeseries=time_series, templates=templates, ground_truth=ground_truth, found_exercises=found_exercises)
    acc, conf = MTW_DTW_EVAL.evaluate()
    amount_of_gt = len(ground_truth)
    
    return acc, conf, amount_of_gt