import loading as loader
import matplotlib.pyplot as plt
from dtaidistance.subsequence.dtw import subsequence_alignment
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw_ndim
import numpy as np

l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[0][4][0]
#time_series = np.abs(time_series)

#for t in range(0,3):
#    templates[t] = np.abs(templates[t])

def get_path_indices_from_array(series, matching_path):
    matching_path_indexes = [row[1] for row in matching_path]
    matching_path = [series[i] for i in matching_path_indexes]
    matching_path = [list(tpl) for tpl in matching_path]
    matching_path = np.array(matching_path)
    return matching_path      
      
max_distance = 25
best_match_distance = 0
paths_long_enough = True
condition = best_match_distance < max_distance 
min_path_length = 5
while condition:
    paths_long_enough = False
    matches = []
    best_match_index = None
    best_match_distance = 25
    for t in range(0,3):
        fig = plt.figure(t)
        query = templates[t]
        series = time_series
        #query = templates[t]
        #series = time_series
        
        sa = subsequence_alignment(query, series,use_c=True)
        match = sa.best_match()
        
        matching_path = get_path_indices_from_array(series=time_series,matching_path=match.path)
        
        distance = dtw_ndim.distance(query[:len(match.path)],matching_path)
        if distance < best_match_distance:
            best_match_distance = distance
            best_match_index = t
            
        matches.append(match)
        dtwvis.plot_warpingpaths(query, series, sa.warping_paths(), match.path, figure=fig,showlegend=True)
        plt.show()
        plt.close('all')
        
    if best_match_index==None:
        print("There is no path found that is close enough")
        break
    print("The matched template of the best match is: " + str(best_match_index+1) + "  The best distance is: " + str(best_match_distance))
    best_match_path = get_path_indices_from_array(series=time_series,matching_path= matches[best_match_index].path)
    print(best_match_path)
    print("Distance is under threshold so will continue: " + str(condition))

    if(condition):
        distinct_path, _, _ = np.unique(best_match_path, axis=0, return_counts=True, return_index=True)
        length_of_best_path = len(distinct_path)
        print("The length of the best path is: " + str(length_of_best_path))
        
        s, e = matches[best_match_index].segment
        print("start of segment to match: " + str(s))
        print("end of segment to match: " + str(e))
        for i in range(s, e+1):
            if(length_of_best_path > min_path_length):  
                print("the path length is: " + str(length_of_best_path) + " so the time series goes *100")
                time_series[i] = (best_match_index+1)*100
            else: 
                print("the path length is: " + str(length_of_best_path) + " so the time series goes *1000")
                time_series[i] = (best_match_index+1)*1000

plt.plot(range(0,len(time_series)), time_series[:,1])
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()
