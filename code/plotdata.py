import loading as loader
import matplotlib.pyplot as plt
import numpy as np

l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[0][1][4]


plt.plot(range(0,len(time_series)), time_series)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Accel')
plt.title('Rotated time-series')
plt.show()

plt.plot(range(0,len(templates[0])), templates[0])
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()




