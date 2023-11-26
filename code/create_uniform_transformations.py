import loading as loader
import orientation_simulation as orsim
import matplotlib.pyplot as plt
import numpy as np

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

plt.plot(range(0,len(time_series)), time_series[:,1])
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation = orsim.orientation_simulation(time_series, random_changes_amount=1, degree_change=0,degree_multiplicator=0)
simulation.create_uniform_rotation(20)
simulation.apply_rotation()
np.save('rotation_20_degrees.npy', simulation.angles)

plt.plot(range(0,len(time_series)), simulation.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_20')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation2 = orsim.orientation_simulation(time_series, random_changes_amount=1, degree_change=0,degree_multiplicator=0)
simulation2.create_uniform_rotation(150)
simulation2.apply_rotation()
np.save('rotation_150_degrees.npy', simulation2.angles)

plt.plot(range(0,len(time_series)), simulation2.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_150')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation3 = orsim.orientation_simulation(time_series, random_changes_amount=1, degree_change=0,degree_multiplicator=0)
simulation3.create_uniform_rotation(200)
simulation3.apply_rotation()
np.save('rotation_200_degrees.npy', simulation3.angles)

plt.plot(range(0,len(time_series)), simulation3.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_200')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation4 = orsim.orientation_simulation(time_series, random_changes_amount=1, degree_change=0,degree_multiplicator=0)
simulation4.create_uniform_rotation(360)
simulation4.apply_rotation()
np.save('rotation_360_degrees.npy', simulation4.angles)

plt.plot(range(0,len(time_series)), simulation4.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_360')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()