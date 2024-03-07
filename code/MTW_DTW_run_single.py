import loading as loader
import MTW_DTW as dtw
import evaluation as eval
import matplotlib.pyplot as plt
import orientation_simulation as orsim
import numpy as np
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
subject = 1
exercise = 1
sensor = 3

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

preprocessor = preproces.preprocessor(series=time_series, templates=templates)
time_series = preprocessor.process()

rotation_file_path = os.path.join("code/rotations", "no_rotation.npy")

time_series = apply_rotation(time_series=time_series, rotation_file=rotation_file_path)

'''
plt.plot(range(0,len(time_series)), time_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Accel')
plt.title('Rotated time-series')
plt.show()
'''

#Use DTW to recognize every occurence of an exercise
DTW = dtw.dtw_windowed(series=time_series, templates=templates, scaling=False, max_distance=0.5, max_matches=30,annotation_margin=0)
DTW.find_matches(k=True, steps=1)
DTW.order_matches()
DTW.annotate_series_max_matches_expected_matched_segments()

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)

MTW_DTW_EVAL = eval.evaluation(series=time_series, templates=templates, ground_truth=ground_truth)
MTW_DTW_EVAL.annotated_series = DTW.annotated_series
MTW_DTW_EVAL.annotate_ground_truth()
MTW_DTW_EVAL.evaluate()
MTW_DTW_EVAL.matrix_profiling_distance_percentage()
conf = MTW_DTW_EVAL.exercise_confusion_matrix()
conf = MTW_DTW_EVAL.create_confusion_matrix_with_assignmentproblem()
print(conf)
acc = MTW_DTW_EVAL.exercise_accuracy(conf)
print(acc)
MTW_DTW_EVAL.plot_simple_confusion_matrix(conf)
# Plotting

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(MTW_DTW_EVAL.annotated_series, label='Annotated series', color='red')
plt.plot(MTW_DTW_EVAL.ground_truth_serie, label='Ground truth', color='green')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('label')
plt.title('Annotated labels')

plt.legend()
plt.show()

#DTW.plot_distances_points(ground_truths=MTMM_DTW_EVAL.ground_truth)

