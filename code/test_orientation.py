import loading as loader
import DTW as dtw
import evaluation as eval
import orientation_simulation as orsim
import matplotlib.pyplot as plt

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

simulation = orsim.orientation_simulation(time_series, 3,1,3)
simulation.create_angles_random_occurences()
simulation.apply_rotation_random_accourences()

plt.plot(range(0,len(time_series)), simulation.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_1')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

simulation2 = orsim.orientation_simulation(time_series, random_changes_amount=0, degree_change=1,degree_multiplicator=1)
simulation.create_angles_random_walk()
simulation.apply_rotation_random_walk()

plt.plot(range(0,len(time_series)), simulation.rotated_series[:,1])
# Add labels and title
plt.xlabel('Time_rotated_2')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)
