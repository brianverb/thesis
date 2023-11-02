import loading as loader
import DTW_w as dtw
import evaluation as eval
import matplotlib.pyplot as plt


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

#Use DTW to recognize every occurence of an exercise
DTW = dtw.dtw_windowed(series=time_series, templates=templates, scaling=scaling, max_distance=25, annotation_margin=-0.1)
DTW.find_matches(k=True, steps=10)
DTW.order_matches()
DTW.annotate_series()

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)


MTMM_DTW_EVAL = eval.evaluation(series=time_series, segmented_indices=DTW.annotated_series, ground_truth=ground_truth)
MTMM_DTW_EVAL.annotated_series = DTW.annotated_series
MTMM_DTW_EVAL.annotate_ground_truth()
MTMM_DTW_EVAL.evaluate()

plt.plot(range(0,len(MTMM_DTW_EVAL.annotated_series)), MTMM_DTW_EVAL.annotated_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('label')
plt.title('Annotated labels')
plt.show()