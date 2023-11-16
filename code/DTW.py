import matplotlib.pyplot as plt
from dtaidistance.subsequence.dtw import subsequence_alignment
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw_ndim
import numpy as np


def get_path_indices_from_array(series, matching_path):
    matching_path_indexes = [row[1] for row in matching_path]
    matching_path = [series[i] for i in matching_path_indexes]
    matching_path = [list(tpl) for tpl in matching_path]
    matching_path = np.array(matching_path)
    return matching_path      

def segment(templates, time_series, min_path_length, max_iterations, max_iterations_bad_match,margin=0, max_distance = 50):
    time_series = [time_series.copy() for _ in range(3)]
    iterations = 0
    iterations_bad_match = 0
    min_path_length = 5
    time_series_segment_indexes = []
    
    while iterations < max_iterations and iterations_bad_match < max_iterations_bad_match:
        print(iterations < max_iterations and iterations_bad_match < max_iterations_bad_match)
        iterations += 1 
        matches = []
        best_match_index = None
        best_match_distance = max_distance
        
        for t in range(0,3):
            fig = plt.figure(t)
            query = templates[t]
            serie = time_series[t]
            sa = subsequence_alignment(query, serie,use_c=True)
            match = sa.best_match()
                      
            if sa.distance < best_match_distance:
                best_match_distance = sa.distance
                best_match_index = t
                
            matches.append(match)
            dtwvis.plot_warpingpaths(query, serie, sa.warping_paths(), match.path, figure=fig,showlegend=True)
            #plt.show()
            #plt.close('all')
            
        if best_match_index == None:
            print("There is no path found that is close enough, we finish early")
            break
        
        print("The matched template of the best match is: " + str(best_match_index+1) + "  The best distance is: " + str(best_match_distance))
        best_match_path = get_path_indices_from_array(series=time_series[best_match_index],matching_path= matches[best_match_index].path)

        distinct_path, _, _ = np.unique(best_match_path, axis=0, return_counts=True, return_index=True)
        length_of_best_path = len(distinct_path)
        print("The length of the best path is: " + str(length_of_best_path))
        
        s, e = matches[best_match_index].segment
        print("start of segment to match: " + str(s))
        print("end of segment to match: " + str(e))
        s = int(s + (e-s)*margin//2)
        e = int(e - (e-s)*margin//2)
        if(length_of_best_path > min_path_length):  
            print("the path length is: " + str(length_of_best_path) + " so the time series goes *100")
            iterations_bad_match = 0
            for i in range(s, e+1):
                for ts in range(0,3):
                    time_series[ts][i] = (best_match_index+1)*100
            time_series_segment_indexes.append((s,e,best_match_index))
                
        else: 
            print("the path length is: " + str(length_of_best_path) + " so the time series goes *1000")
            iterations_bad_match += 1
            print("Bad match counter: " + str(iterations_bad_match))
            for i in range(s, e+1):
                time_series[best_match_index][i] = (best_match_index+1)*1000
                
    return time_series, time_series_segment_indexes

'''
plt.plot(range(0,len(time_series)), time_series[:,1])
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()
'''
