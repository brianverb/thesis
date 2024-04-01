"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import matplotlib.pyplot as plt
from dtaidistance.subsequence.dtw import subsequence_alignment
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw_ndim

import numpy as np
import kabsch
import matplotlib.pyplot as plt


def get_path_indices_from_array(series, matching_path):
    matching_path_indexes = [row[1] for row in matching_path]
    matching_path = [series[i] for i in matching_path_indexes]
    matching_path = [list(tpl) for tpl in matching_path]
    matching_path = np.array(matching_path)
    return matching_path      

def prepare_timeseries(templates, time_series, kabsch):
    if(kabsch):
        time_series = transform(templates,time_series, scaling=False)
        '''
        plt.plot(range(0,len(time_series[0])), time_series[0])
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Accel')
        plt.title('Rotated time-series')
        plt.show()
        plt.plot(range(0,len(time_series[1])), time_series[1])
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Accel')
        plt.title('Rotated time-series')
        plt.show()

        plt.plot(range(0,len(time_series[2])), time_series[2])
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Accel')
        plt.title('Rotated time-series')
        plt.show()
        '''

    else:
        time_series = [time_series.copy() for _ in range(3)]
    return time_series

def find_exercises(templates, time_series, kabsch=False, min_segment_length=0.5, margin=0.1, max_distance=10):
    time_series= prepare_timeseries(templates, time_series, kabsch)
    print(max_distance)
    longest_unmatched_sequence = [len(time_series[0]), len(time_series[1]), len(time_series[2])]
    
    found_exercises = []
    #distances = np.load("distances.npy", allow_pickle=True)
    #distances = distances.tolist()
    while True:
        matches = [0,0,0]
        best_match_index = None
        best_match_distance = max_distance
        
        for t in range(0,3):
            query = templates[t]
            serie = time_series[t]
            if longest_unmatched_sequence[t] >= len(query) * min_segment_length:
                if kabsch:
                    sa = subsequence_alignment(query, serie, use_c=True)
                else:
                    sa = subsequence_alignment(query, serie, penalty=10, use_c=True)
                match = sa.best_match()
                distance = sa.distance / len(templates[t])
                #print(distance)
                if distance < best_match_distance:         
                    best_match_distance = distance
                    best_match_index = t
                
                matches[t] = match
            #dtwvis.plot_warpingpaths(query, serie, sa.warping_paths(), match.path, figure=fig,showlegend=True)
            #plt.show()
            
        if best_match_index == None:
            #print("There is no path found that is close enough, we finish early")
            break
        
        #print("The matched template of the best match is: " + str(best_match_index+1) + "  The best distance is: " + str(best_match_distance))
        best_match_path = get_path_indices_from_array(series=time_series[best_match_index],matching_path= matches[best_match_index].path)

        distinct_path, _, _ = np.unique(best_match_path, axis=0, return_counts=True, return_index=True)
        length_of_best_path = len(distinct_path)
        #print("The length of the best path is: " + str(length_of_best_path))
        #distances.append(best_match_distance)
        
        s, e = matches[best_match_index].segment
        #print("start of segment to match: " + str(s))
        #print("end of segment to match: " + str(e))
        s_o = int(s + (e-s)*margin//2)
        e_o = int(e - (e-s)*margin//2)
        if(length_of_best_path/len(templates[best_match_index]) > min_segment_length):  
            #print("the path length is: " + str(length_of_best_path) + " so the time series goes *100")    
            for ts in range(0,3):
                for i in range(s_o, e_o+1):
                    time_series[ts][i] = float('inf')
                    
                longest_unmatched_sequence[ts] = find_longest_unmatched_sequence(time_series[ts])
            found_exercises.append((s,e,best_match_index))
                
        else: 
            #print("the path length is: " + str(length_of_best_path) + " so the time series goes *1000")
            for i in range(s, e+1):
                time_series[best_match_index][i] = float('inf')
            longest_unmatched_sequence[best_match_index] = find_longest_unmatched_sequence(time_series[best_match_index])
            
        
    #distances = np.array(distances)
    #np.save("distances.npy", distances)
    return found_exercises
def find_longest_unmatched_sequence(timeseries):
    max_length = 0
    current_length = 0
    
    for i in range(len(timeseries)):
        if timeseries[i][0] != float('inf'):
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    return max_length
        
def plot_3d_path(data, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)

    # Plot the 3D data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o', label='Data Points')

    indices= data.shape[0]
    # Connect the points based on the indices
    for i in range(0,indices-1):
        ax.plot([data[i, 0], data[i+1, 0]],
                [data[i, 1], data[i+1, 1]],
                [data[i, 2], data[i+1, 2]], c='r')

    ax.set_xlabel('X_acc')
    ax.set_ylabel('Y_acc')
    ax.set_zlabel('Z_acc')

    plt.legend()
    plt.show()
    
def plot_kabsch(template, window, result):
    plot_3d_path(template, "template")
    plot_3d_path(window, "window")
    plot_3d_path(result, "result")
    
def kabsch_transform_first_window(template, template_length, time_series, transformed_timeserie, stride):
    window = time_series[0:template_length]
            
    already_masked = np.any(np.isinf(window))
    if(not already_masked):
        _, ret_R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), False)
        result = np.dot(ret_R, window.T).T
    else:
        result= window
    _, ret_R, _ = kabsch.rigid_transform_3D(np.matrix(template),np.matrix(window), False)
    
    num_rows_to_copy = template.shape[0] // 2 - (stride//2)
    transformed_timeserie[:num_rows_to_copy, :] = result[:num_rows_to_copy, :]
    
    #plot_kabsch(template, window, result)

def kabsch_transform_sliding_window(template, template_length, time_series, time_series_length, transformed_timeserie, stride):
    #Slide the window for all of the middle values
    for w in range(0,time_series_length-template_length+1, stride):
        
        window = time_series[w:w+template_length]
        
        already_masked = np.any(np.isinf(window))
        if(not already_masked):
            _, ret_R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), False)
            result = np.dot(ret_R, window.T).T
        else:
            result= window


        for i in range(0, stride):
            index = w + template_length//2 - stride + i
            transformed_timeserie[index] = result[template_length//2 - (stride//2) + i]
    return result

def kabsch_transform_last_window(template_length, previous_result, transformed_timeserie, stride):
    if(template_length != stride):
        if template_length // 2 == 0:
            transformed_timeserie[-template_length//2+(stride//2):, :] = previous_result[-template_length//2+(stride//2) :]
        else:
            transformed_timeserie[-template_length//2+1+(stride//2):, :] = previous_result[-template_length//2+1+ (stride//2):]

        
def transform(templates, time_series, scaling):
    time_series_length = len(time_series)
    time_series_transformed = [np.full(time_series.shape, -1) ,np.full(time_series.shape, -1) ,np.full(time_series.shape, -1)]
    #For each template convert the timeseries using the kabsch algorithm
    for t in range(0,3):
        template = templates[t]
        template_length = len(template)
        transformed_timeserie = np.empty(time_series.shape)
        #int(template_length * 0.8)
        stride = template_length
        kabsch_transform_first_window(stride=stride, template=template, template_length=template_length,time_series=time_series, transformed_timeserie=transformed_timeserie)
          
        #print("template: " +str(t+1) + " with shape: " + str(template.shape))
        #print("start: " + str(template_length//2) + "  end: " + str(time_series_length-template_length))
        
        result = kabsch_transform_sliding_window(stride=stride, template=template, template_length=template_length, time_series=time_series, time_series_length=time_series_length, transformed_timeserie=transformed_timeserie)
        
        kabsch_transform_last_window(stride=stride, template_length=template_length, previous_result=result ,transformed_timeserie=transformed_timeserie)
        
        time_series_transformed[t] = transformed_timeserie

    return time_series_transformed
        
