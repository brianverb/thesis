import loading as loader
import MTW_DTW as dtw
import evaluation as eval
import orientation_simulation as orsim
import numpy as np
import preprocessing as prep

def run(subject, exercise, unit, rotation_file, preprocess, kabsch):
    templates, time_series, ground_truth = load_data(subject, exercise, unit)

    if preprocess:
        preprocessor = prep.preprocessor(series=time_series, templates=templates)
        time_series = preprocessor.process()

    time_series = apply_rotation(time_series, rotation_file)

    annotated_series = segment_time_series(time_series, templates, kabsch)
    
    accuracy = evaluate_time_series(time_series, templates, annotated_series, ground_truth)
    
    return accuracy

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
    
def segment_time_series(time_series, templates, kabsch):
    #Use DTW to recognize every occurence of an exercise
    DTW = dtw.dtw_windowed(series=time_series, templates=templates, scaling=False, max_distance=35, max_matches=30,annotation_margin=0)
    DTW.find_matches(k=kabsch, steps=1)
    DTW.order_matches()
    return DTW.annotate_series_max_distance()

def evaluate_time_series(time_series, templates, annotated_series, ground_truth):
    MTMM_DTW_EVAL = eval.evaluation(series=time_series, templates=templates, ground_truth=ground_truth)
    MTMM_DTW_EVAL.annotated_series = annotated_series
    MTMM_DTW_EVAL.annotate_ground_truth()
    MTMM_DTW_EVAL.clean_annotations()

    acc = MTMM_DTW_EVAL.exercise_accuracy()
    
    return acc