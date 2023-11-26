import loading as loader
import MTW_DTW as dtw
import evaluation as eval
import matplotlib.pyplot as plt
import orientation_simulation as orsim
import numpy as np

#set up the data
l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series
subject = 2
exercise = 0
sensor = 1

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[subject][exercise][sensor]

simulation = orsim.orientation_simulation(time_series, 1,1,3)
simulation.angles = np.load('rotation__degrees.npy')
simulation.apply_rotation()


time_series = simulation.rotated_series

#Use DTW to recognize every occurence of an exercise
DTW = dtw.dtw_windowed(series=time_series, templates=templates, scaling=False, max_distance=50, max_matches=30,annotation_margin=0)
DTW.find_matches(k=False, steps=1)
#DTW.find_matches_svd(steps=10)
#DTW.plot_matches()
DTW.order_matches()
DTW.annotate_series_max_matches_expected_matched_segments()

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)


MTMM_DTW_EVAL = eval.evaluation(series=time_series, ground_truth=ground_truth)
MTMM_DTW_EVAL.annotated_series = DTW.annotated_series
MTMM_DTW_EVAL.annotate_ground_truth()
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

DTW.plot_distances_points(ground_truths=MTMM_DTW_EVAL.ground_truth)


