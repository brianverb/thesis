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

def segment(templates, time_series,  max_iterations, max_iterations_bad_match,min_path_length=0.1, margin=-0.1, max_distance=1000):
    iterations = 0
    iterations_bad_match = 0
    time_series_segment_indexes = []
    distances = np.load("distances.npy", allow_pickle=True)
    distances = distances.tolist()
    while iterations < max_iterations and iterations_bad_match < max_iterations_bad_match:
        #print(iterations < max_iterations and iterations_bad_match < max_iterations_bad_match)
        iterations += 1 
        matches = []
        best_match_index = None
        best_match_distance = max_distance
        
        for t in range(0,3):
            fig = plt.figure(t)
            query = templates[t]
            serie = time_series[t]
            sa = subsequence_alignment(query, serie, penalty=10, use_c=True)
            match = sa.best_match()
            distance = sa.distance / len(templates[t])
            if distance < best_match_distance:         
                best_match_index = t
                
            matches.append(match)
            #dtwvis.plot_warpingpaths(query, serie, sa.warping_paths(), match.path, figure=fig,showlegend=True)
            #plt.show()
            
        if best_match_index == None:
            increment_value_in_file("distance")
            #print("There is no path found that is close enough, we finish early")
            break
        
        #print("The matched template of the best match is: " + str(best_match_index+1) + "  The best distance is: " + str(best_match_distance))
        best_match_path = get_path_indices_from_array(series=time_series[best_match_index],matching_path= matches[best_match_index].path)

        distinct_path, _, _ = np.unique(best_match_path, axis=0, return_counts=True, return_index=True)
        length_of_best_path = len(distinct_path)
        #print("The length of the best path is: " + str(length_of_best_path))
        distances.append(best_match_distance)
        
        s, e = matches[best_match_index].segment
        #print("start of segment to match: " + str(s))
        #print("end of segment to match: " + str(e))
        #s = max(int(s + (e-s)*margin//2),0)
        #e = min(int(e - (e-s)*margin//2), len(time_series[0])-1)
        if(length_of_best_path/len(templates[best_match_index]) > min_path_length):  
            #print("the path length is: " + str(length_of_best_path) + " so the time series goes *100")
            iterations_bad_match = 0
            for i in range(s, e+1):
                for ts in range(0,3):
                    time_series[ts][i] = (best_match_index+1)*100
            time_series_segment_indexes.append((s,e,best_match_index))
                
        else: 
            #print("the path length is: " + str(length_of_best_path) + " so the time series goes *1000")
            iterations_bad_match += 1
            #print("Bad match counter: " + str(iterations_bad_match))
            for i in range(s, e+1):
                time_series[best_match_index][i] = (best_match_index+1)*1000
        
        if iterations >= max_iterations:
            increment_value_in_file("iterations")
        if iterations_bad_match >= max_iterations_bad_match:
            increment_value_in_file("bad_iterations")
    #print("max iterations for a bad match counter: " + str(max_iterations_bad_match) + "  bad iteration counter: " + str(iterations_bad_match) + "  iterations: " + str(iterations))
    distances = np.array(distances)
    np.save("distances.npy", distances)
    return time_series, time_series_segment_indexes


def increment_value_in_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the value from the file
        current_value = int(file.read().strip())

    # Increment the value by 1
    new_value = current_value + 1

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the updated value back to the file
        file.write(str(new_value))