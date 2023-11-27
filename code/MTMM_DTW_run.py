import loading as loader
import numpy as np
import kabsch_timeseries as kabsch_time
import MTMM_DTW as dtw
import evaluation as eval
import csv
import matplotlib.pyplot as plt
import orientation_simulation as orsim
import preprocessing as preproces

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

preprocessor = preproces.preprocessor(series=time_series, templates=templates)
preprocessor.process()

simulation = orsim.orientation_simulation(time_series, 1,1,3)
simulation.angles = np.load('rotation_150_degrees.npy')

simulation.apply_rotation()
time_series = simulation.rotated_series

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


# Save the array to a CSV file
with open('transformed_series.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for plane in transformed_series:
        writer.writerows(plane)

time_series = [time_series.copy() for _ in range(3)]

#Use DTW to recognize every occurence of an exercise
(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,time_series=time_series,min_path_length=20,max_iterations=300, max_iterations_bad_match = 30)
#(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,time_series=transformed_series,min_path_length=20,max_iterations=2000, max_iterations_bad_match = 400)

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)
#MTMM_DTW_EVAL = eval.evaluation(series=time_series[0], segmented_indices=segmented_series_classification_indices, ground_truth=ground_truth)
MTMM_DTW_EVAL = eval.evaluation(series=transformed_series[0], segmented_indices=segmented_series_classification_indices, ground_truth=ground_truth)
MTMM_DTW_EVAL.annotate_ground_truth()
MTMM_DTW_EVAL.annotate_timeseries()
MTMM_DTW_EVAL.evaluate()

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
MTMM_DTW_EVAL.clean_annotations()
MTMM_DTW_EVAL.annotate_timeseries()
MTMM_DTW_EVAL.evaluate()

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

MTMM_DTW_EVAL.plot_simple_confusion_matrix()
