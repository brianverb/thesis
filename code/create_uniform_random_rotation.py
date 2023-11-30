import loading as loader
import orientation_simulation as orsim
import matplotlib.pyplot as plt
import numpy as np
import os

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

simulation = orsim.orientation_simulation(time_series, random_changes_amount=1, degree_change=0,degree_multiplicator=0)
x_angle, y_angle, z_angle, rotated_series = simulation.create_uniform_random_rotation()
file_name = f"rotation_uniform_angles_{x_angle}_{y_angle}_{z_angle}.npy"
file_path = os.path.join("code/rotations", file_name)

np.save(file_path, [x_angle, y_angle, z_angle])

plt.plot(range(0,len(time_series)), rotated_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('rotated_series over time')
plt.show()
