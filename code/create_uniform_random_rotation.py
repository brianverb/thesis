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

simulation = orsim.orientation_simulation(time_series, random_changes_amount=1, degree_change=0,degree_multiplicator=0)
x_angle, y_angle, z_angle, rotated_series = simulation.create_uniform_random_rotation()
file_name = f"rotation_uniform_{x_angle}_{y_angle}_{z_angle}.npy"
file_path = os.path.join("code/rotations", file_name)

np.save(file_path, rotated_series)

