"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""
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
rot_M = simulation.create_rotation_matrix_gimbal()
print(rot_M)
rotated_series = simulation.apply_rotation(series=time_series,rotation_matrix=rot_M)
file_name = f"rotation_gimbal_3.npy"
file_path = os.path.join("code/rotations", file_name)


np.save(file_path, rot_M)

plt.plot(range(0,len(time_series)), rotated_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('rotated_series over time')
plt.show()



for i in range(0,len(templates)):
    template = templates[i]
    template_length = len(template)
    
    window = rotated_series[12:12+template_length]
    
    plt.plot(range(0,len(window)), template)
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('template: ' + str(i))
    plt.show()
    
    distance = dtw_ndim.distance(window, template, use_c=True)

    plt.plot(range(0,len(window)), rotated_series[12:12+template_length])
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('rotated series: ' + str(i) + "  distance: " + str(distance))
    plt.show()
    
    _, R, _ = kabsch.rigid_transform_3D(np.matrix(template), np.matrix(window), True)
    window = np.dot(window,R)
    window = np.array(window)
    
    distance = dtw_ndim.distance(window, template, use_c=True)

    plt.plot(range(0,len(window)), window)
    plt.xlabel('Time')
    plt.ylabel('x_acc')
    plt.title('Kabsch rotated serie: ' + str(i) + "  distance: " + str(distance))
    plt.show()
    
 