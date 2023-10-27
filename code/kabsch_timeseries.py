import loading as loader
import kabsch
import numpy as np


#set up the data
l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series
scaling = False

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[0][4][0]
time_series_length = len(time_series)
time_series_transformed = []

#For each template convert the timeseries using the kabsch algorithm
for t in range(0,3):
    template = templates[t]
    template_length = len(template)

    transformed_timeserie = np.empty((5910, 3))  
    #Take the first couple of values at once instead of sliding the window
    window = time_series[0:template_length]

    s, ret_R, ret_t = kabsch.rigid_transform_3D(np.matrix(template),np.matrix(window), scaling)
    num_rows_to_copy = template.shape[0] // 2
    result = np.dot(window,ret_R)
    
    transformed_timeserie[:num_rows_to_copy, :] = result[:num_rows_to_copy, :]

    print("timeseries: " + str(time_series.shape))
    print("template: " + str(template.shape))
    print("start: " + str(num_rows_to_copy) + "  end: " + str(time_series_length-template_length))
    #Slide the window for all of the middle values
    for w in range(num_rows_to_copy,time_series_length- template_length):
        window = time_series[w:w+template_length]
        s, ret_R, ret_t = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), scaling)
        result = np.dot(window,ret_R)
        index =w+num_rows_to_copy
        transformed_timeserie[:index, :] = result[template_length//2]

    #Take the last couple of values at once instead of sliding the window
    window = time_series[time_series_length-template_length:time_series_length]
    s, ret_R, ret_t = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), scaling)
    result = np.dot(window,ret_R)
    transformed_timeserie[-num_rows_to_copy:, :] = result[-template_length//2:]

    time_series_transformed.append(transformed_timeserie)

#Check whetever a value in the array has not been set
for i in range (0,3):
    contains_nann = np.isnan(time_series_transformed[i]).any()
    contains_none = any(item is None for item in time_series_transformed[i])
    contains_empty = contains_nann or contains_none
    print(contains_empty)
    print(time_series_transformed[i].shape)
    
