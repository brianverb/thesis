import loading as loader
import numpy as np
import kabsch_timeseries as kabsch_time
import MTMM_DTW as dtw
import evaluation as eval
import csv
import matplotlib.pyplot as plt
import orientation_simulation as orsim
import preprocessing as preproces
import os

def apply_rotation(time_series, rotation_file):
    simulation = orsim.orientation_simulation()
    rotation_matrix = np.load(rotation_file)

    rotated_series = simulation.apply_rotation(series=time_series, rotation_matrix=rotation_matrix)
    return rotated_series 
    
#set up the data
l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series
scaling = False
subject = 0
exercise = 1
sensor = 4

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

#preprocessor = preproces.preprocessor(series=time_series, templates=templates)
#time_series = preprocessor.process()

rotation_file_path = os.path.join("code/rotations", "no_rotation.npy")
'''
plt.plot(range(0,len(time_series)), time_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Accel')
plt.title('Rotated time-series')
plt.show()
'''
#time_series = apply_rotation(time_series=time_series, rotation_file=rotation_file_path)


#Use the kabsch algorithm to transform the timeseries to optimal rotated series based on the templates
transformed_series = kabsch_time.transform(templates,time_series,scaling)


plt.plot(range(0,len(time_series)), time_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Accel')
plt.title('Rotated time-series')
plt.show()


plt.plot(range(0,len(transformed_series[1])), transformed_series[1])
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Accel')
plt.title('transformed timeseries')
plt.show()

transformed_series = [time_series.copy() for _ in range(3)]

#Use DTW to recognize every occurence of an exercise
#(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,time_series=time_series,min_path_length=20,max_iterations=300, max_iterations_bad_match = 30)
(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,time_series=transformed_series, max_iterations=3000, max_iterations_bad_match = 500)

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)
#MTMM_DTW_EVAL = eval.evaluation(series=time_series[0], segmented_indices=segmented_series_classification_indices, ground_truth=ground_truth)
MTMM_DTW_EVAL = eval.evaluation(series=transformed_series[0], templates=templates, segmented_indices=segmented_series_classification_indices, ground_truth=ground_truth)
MTMM_DTW_EVAL.annotate_ground_truth()
MTMM_DTW_EVAL.annotate_timeseries()

# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(MTMM_DTW_EVAL.annotated_series, label='Annotated series', color='red')
plt.plot(MTMM_DTW_EVAL.ground_truth_serie, label='Ground truth', color='green')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('label')
plt.title('Annotated labels')

plt.legend()
plt.show()

#MTMM_DTW_EVAL.clean_annotations()

# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(MTMM_DTW_EVAL.annotated_series, label='Annotated series', color='red')
plt.plot(MTMM_DTW_EVAL.ground_truth_serie, label='Ground truth', color='green')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('label')
plt.title('Annotated labels')

plt.legend()
plt.show()

MTMM_DTW_EVAL.matrix_profiling_distance_percentage()
conf = MTMM_DTW_EVAL.exercise_confusion_matrix()
acc = MTMM_DTW_EVAL.exercise_accuracy(conf)
print(conf)
print(acc)
    