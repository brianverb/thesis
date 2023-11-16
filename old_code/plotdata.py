import loading as loader
import matplotlib.pyplot as plt
import numpy as np

l = loader.Loading("code\data")
l.load_all()
subjects = l.time_series

# get accelerometer data of first subject performing the second exercises using the second sensor
templates, time_series = subjects[0][1][1]
print(time_series.shape)

time_series = np.abs(time_series)
for t in range(0,3):
    templates[t] = np.abs(templates[t])
data_to_use= templates[0]
#data_to_use = time_series

plt.plot(range(0,len(time_series)), time_series[:,0])
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

plt.plot(range(0,len(templates[0])), templates[0][:,0])
# Add labels and title
plt.xlabel('Time')
plt.ylabel('x_acc')
plt.title('X_accelator over time')
plt.show()

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use the 'w' dimension to represent color
scatter = ax.scatter(data_to_use[:,0],data_to_use[:,1], data_to_use[:,2], c=data_to_use[:,3], cmap='viridis')

# Add a color bar to show the mapping of the 'w' dimension
color_bar = plt.colorbar(scatter)
color_bar.set_label('4th Dimension (w)')

# Add labels and a title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('4D Scatter Plot')

plt.show()




