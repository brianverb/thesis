import loading as loader
import orientation_simulation as orsim
import matplotlib.pyplot as plt
import numpy as np
import os
import kabsch 
from dtaidistance import dtw_ndim

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

plt.plot(range(0,len(time_series)), time_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation = orsim.orientation_simulation()
rot_M = simulation.create_random_rotation_matrix()
print(rot_M)
rotated_series = simulation.apply_rotation(series=time_series,rotation_matrix=rot_M)
file_name = f"rotation_gram_schmidt_x.npy"
file_path = os.path.join("code/old_rotations", file_name)


np.save(file_path, rot_M)

plt.plot(range(0,len(time_series)), rotated_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('rotated_series over time')
plt.show()

def euclidean_distance_3d_timeseries(series_a, series_b):
    total_distance = 0

    for point_a, point_b in zip(series_a, series_b):
        distance = np.linalg.norm(np.array(point_a) - np.array(point_b))
        total_distance += distance

    return total_distance

for i in range(0,len(templates)):
    template = templates[i]
    template_length = len(template)
    
    window = rotated_series[12:12+template_length]
    
    plt.plot(range(0,len(window)), time_series[12:12+template_length])
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('window')
    plt.show()
    
    _, R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(time_series[12:12+template_length]), True)
    result = np.dot(R,time_series[12:12+template_length].T)
    result = np.array(result.T)
    
    ed_distance = euclidean_distance_3d_timeseries(result, template)
    dtw_distance = dtw_ndim.distance(result, template, use_c=True)
    
    plt.plot(range(0,len(window)), result)
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('kabsch on regular timeseries: ' + str(i) + "  euclidian distance: " + str(ed_distance) +  "  dtw distance: " + str(dtw_distance))
    plt.show()
    
    ed_distance = euclidean_distance_3d_timeseries(time_series[12:12+template_length], template)
    dtw_distance = dtw_ndim.distance(time_series[12:12+template_length], template, use_c=True)
    
    plt.plot(range(0,len(window)), template)
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('template: ' + str(i) + "  euclidian distance: " + str(ed_distance) +  "  dtw distance: " + str(dtw_distance))
    plt.show()
    
    ed_distance = euclidean_distance_3d_timeseries(window, template)
    dtw_distance = dtw_ndim.distance(window, template, use_c=True)

    plt.plot(range(0,len(window)), window)
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('rotated window: ' + str(i) + "  euclidian distance: " + str(ed_distance) +  "  dtw distance: " + str(dtw_distance))
    plt.show()
    
    _, R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), True)
    window = np.dot(R,window.T)
    window = np.array(window.T)
    
    ed_distance = euclidean_distance_3d_timeseries(window, template)
    dtw_distance = dtw_ndim.distance(window, template, use_c=True)
    
    plt.plot(range(0,len(window)), window)
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('rotated window kabsch: ' + str(i)  + "  euclidian distance: " + str(ed_distance) +  "  dtw distance: " + str(dtw_distance))
    plt.show()
    
 