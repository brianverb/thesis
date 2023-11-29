import loading as loader
import numpy as np
import kabsch_timeseries as kabsch_time
import MTMM_DTW as dtw
import evaluation as eval
import orientation_simulation as orsim
import preprocessing as prep

def run(subject, exercise, unit, rotation_file, preprocess, kabsch):
    templates, time_series, ground_truth = load_data(subject, exercise, unit)

    
    if preprocess:
        preprocessor = prep.preprocessor(series=time_series, templates=templates)
        time_series = preprocessor.process()

    time_series = apply_rotation(time_series, rotation_file)

    classificaiton_indices = segment_time_series(time_series, templates, kabsch)
    
    accuracy, confusion_matrix = evaluate_time_series(time_series, classificaiton_indices, ground_truth)
    
    return accuracy, confusion_matrix
    
def load_data(subject, exercise, unit):
    l = loader.Loading("code\data")
    l.load_all()
    subjects = l.time_series
    subject = 2
    exercise = 0
    unit = 1

    templates, time_series = subjects[subject][exercise][unit]
    ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)

    return templates, time_series, ground_truth

def apply_rotation(time_series, rotation_file):
    simulation = orsim.orientation_simulation(time_series)
    simulation.angles = np.load(rotation_file)

    simulation.apply_rotation()
    return simulation.rotated_series

def segment_time_series(time_series, templates, kabsch):
    if kabsch:
        #Use the kabsch algorithm to transform the timeseries to optimal rotated series based on the templates
        time_series = kabsch_time.transform(templates,time_series,scaling=False)
        (_, segmented_series_classification_indices) = dtw.segment(templates,time_series=time_series,min_path_length=20,max_iterations=300, max_iterations_bad_match = 30)
    else:
        time_series = [time_series.copy() for _ in range(3)]
        (_, segmented_series_classification_indices) = dtw.segment(templates,time_series=time_series,min_path_length=20,max_iterations=2000, max_iterations_bad_match = 400)
    
    return segmented_series_classification_indices

def evaluate_time_series(time_series, classificaiton_indices, ground_truth):
    MTMM_DTW_EVAL = eval.evaluation(series=time_series[:,0], segmented_indices=classificaiton_indices, ground_truth=ground_truth)
    MTMM_DTW_EVAL.annotate_ground_truth()
    MTMM_DTW_EVAL.annotate_timeseries()
    MTMM_DTW_EVAL.clean_annotations()
    
    conf = MTMM_DTW_EVAL.simple_confusion_matrix()
    acc = MTMM_DTW_EVAL.simple_accuracy()
    
    return acc, conf
    