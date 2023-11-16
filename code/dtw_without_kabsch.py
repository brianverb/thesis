import loading as loader
import DTW as dtw
import evaluation as eval


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

time_series_copies = []
time_series_copies.append(time_series.copy())
time_series_copies.append(time_series.copy())
time_series_copies.append(time_series.copy())
#Use DTW to recognize every occurence of an exercise
(segmented_series, segmented_series_classification_indices) = dtw.segment(templates,time_series_copies,min_path_length =5,max_iterations=1000, max_iterations_bad_match = 50, margin=0)

print("Amount of exercises found: " + str(len(segmented_series_classification_indices)))

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)
MTMM_DTW_EVAL = eval.evaluation(series=time_series, segmented_indices=segmented_series_classification_indices, ground_truth=ground_truth)
MTMM_DTW_EVAL.annotate_timeseries()
MTMM_DTW_EVAL.annotate_ground_truth()
MTMM_DTW_EVAL.evaluate()