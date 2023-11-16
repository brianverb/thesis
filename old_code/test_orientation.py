import loading as loader
import DTW as dtw
import evaluation as eval
import orientation_simulation as orsim

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


simulation = orsim.orientation_simulation(time_series, 3,1,3)
simulation.create_angles_random_occurences()
print(simulation.random_changes)
print(simulation.random_changes_amount)

simulation.apply_rotation_random_accourences()
print(simulation.rotated_series)

simulation2 = orsim.orientation_simulation(time_series, 3,1,3)
simulation.create_angles_random_walk()
simulation.apply_rotation_random_walk()
print(simulation.rotated_series)


ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=subject,exercise=exercise)
