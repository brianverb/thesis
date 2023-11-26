import kabsch
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_path(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o', label='Data Points')

    indices= data.shape[0]
    # Connect the points based on the indices
    for i in range(0,indices-1):
        ax.plot([data[i, 0], data[i+1, 0]],
                [data[i, 1], data[i+1, 1]],
                [data[i, 2], data[i+1, 2]], c='r')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.legend()
    plt.show()
    
def plot_kabsch(template, window, result):
    plot_3d_path(template)
    plot_3d_path(window)
    plot_3d_path(result)
    
def kabsch_transform_first_window(template, template_length, time_series, transformed_timeserie, scaling):
    window = time_series[0:template_length]
    _, ret_R, _ = kabsch.rigid_transform_3D(np.matrix(template),np.matrix(window), scaling)
    
    num_rows_to_copy = template.shape[0] // 2
    result = np.dot(window,ret_R)
    transformed_timeserie[:num_rows_to_copy, :] = result[:num_rows_to_copy, :]
    
    #plot_kabsch(template[:20], window[:20], result[:20])

def kabsch_transform_sliding_window(template, template_length, time_series, time_series_length, transformed_timeserie, scaling):
    #Slide the window for all of the middle values
    for w in range(0,time_series_length-template_length+1):
        window = time_series[w:w+template_length]
        _, ret_R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), scaling)
        result = np.dot(window,ret_R)
        index = w + template_length//2
        transformed_timeserie[index] = result[template_length//2]
    return result

#TODO
def kabsch_transform_sliding_window_with_ids(template, template_length, time_series, time_series_length, transformed_timeserie, scaling):
    #Slide the window for all of the middle values
    for w in range(0,time_series_length-template_length+1):
        window = time_series[w:w+template_length]
        _, ret_R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), scaling)
        result = np.dot(window,ret_R)
        index = w + template_length//2
        transformed_timeserie[index] = result[template_length//2]
    return result

def kabsch_transform_last_window(template_length, previous_result, transformed_timeserie):
    if template_length // 2 == 0:
        transformed_timeserie[-template_length//2:, :] = previous_result[-template_length//2:]
    else: 
        transformed_timeserie[-template_length//2+1:, :] = previous_result[-template_length//2+1:]

def kabsch_check_transformed_series(time_series_transformed):
    #Check whetever a value in the array has not been set
    for i in range (0,3):
        contains_nann = np.isnan(time_series_transformed[i]).any()
        contains_none = any(item is None for item in time_series_transformed[i])
        contains_empty = contains_nann or contains_none
        #print("Does timeseries " + str(i) +" have empty values: " +  str(contains_empty))
        #print(time_series_transformed[i].shape)
        
def transform(templates, time_series, scaling):
    time_series_length = len(time_series)
    time_series_transformed = [np.full(time_series.shape, -1) ,np.full(time_series.shape, -1) ,np.full(time_series.shape, -1)]

    #For each template convert the timeseries using the kabsch algorithm
    for t in range(0,3):
        template = templates[t]
        template_length = len(template)
        transformed_timeserie = np.empty(time_series.shape)
        
        kabsch_transform_first_window(template=template, template_length=template_length,time_series=time_series, transformed_timeserie=transformed_timeserie, scaling=scaling )
          
        #print("template: " +str(t+1) + " with shape: " + str(template.shape))
        #print("start: " + str(template_length//2) + "  end: " + str(time_series_length-template_length))
        
        result = kabsch_transform_sliding_window(template=template, template_length=template_length, time_series=time_series, time_series_length=time_series_length, transformed_timeserie=transformed_timeserie, scaling=scaling)
        
        kabsch_transform_last_window(template_length=template_length, previous_result=result ,transformed_timeserie=transformed_timeserie)
        
        time_series_transformed[t] = transformed_timeserie

    kabsch_check_transformed_series(time_series_transformed=time_series_transformed)
    
    #plot_kabsch(template[:20], time_series[time_series_length//2-10:time_series_length//2+10], transformed_timeserie[time_series_length//2-10:time_series_length//2+10])
    
    return time_series_transformed
        
