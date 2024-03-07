import loading as loader
import MTW_DTW as dtw
import evaluation as eval
import matplotlib.pyplot as plt
import orientation_simulation as orsim
import numpy as np
import preprocessing as preproces
import os

#set up the data
l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series
subject = 1
exercise = 1
sensor = 3

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

def apply_rotation(time_series, rotation_file):
    simulation = orsim.orientation_simulation()
    rotation_matrix = np.load(rotation_file)

    rotated_series = simulation.apply_rotation(series=time_series, rotation_matrix=rotation_matrix)
    return rotated_series 

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
DTW = dtw.dtw_windowed(series=time_series, templates=templates, scaling=False, max_distance=30, max_matches=30,annotation_margin=0)
found_exercises = DTW.find_exercises_max_distance(kabsch=True, steps=1)

#Load ground_truth
ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)

#Evaluate
MTW_DTW_EVAL = eval.evaluation(timeseries=time_series, templates=templates, ground_truth=ground_truth, found_exercises=found_exercises)
acc, conf = MTW_DTW_EVAL.evaluate()
print(conf)
print(acc)
MTW_DTW_EVAL.plot_simple_confusion_matrix(conf)



