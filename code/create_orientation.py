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
simulation.create_angles_random_occurences()
simulation.apply_rotation_random_accourences()
np.save('random_occurences_1.npy', simulation.angles)

plt.plot(range(0,len(time_series)), simulation.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_1')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation4 = orsim.orientation_simulation(time_series, random_changes_amount=3, degree_change=1,degree_multiplicator=1)
simulation4.create_angles_random_occurences()
simulation4.apply_rotation_random_accourences()
np.save('random_occurences_3.npy', simulation4.angles)

plt.plot(range(0,len(time_series)), simulation4.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_1')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()


simulation2 = orsim.orientation_simulation(time_series, random_changes_amount=0, degree_change=0.1,degree_multiplicator=5)
simulation2.create_angles_random_walk()
simulation2.apply_rotation()
np.save('random_walk_0_01_5.npy', simulation2.angles)

plt.plot(range(0,len(time_series)), simulation2.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_2')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation3 = orsim.orientation_simulation(time_series, random_changes_amount=3, degree_change=0.1,degree_multiplicator=5)
simulation3.create_angles_random_occurences()
simulation3.create_angles_random_walk()
simulation3.apply_rotation()
np.save('random_combination_3_01_5.npy', simulation3.angles)

plt.plot(range(0,len(time_series)), simulation3.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_2')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()