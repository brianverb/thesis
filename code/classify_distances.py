import numpy as np
import pandas as pd

distances = np.load("distances.npy", allow_pickle=True)

# Convert the ndarray to a pandas DataFrame
df = pd.DataFrame(distances)

# Specify the file path to save the Excel file
file_path = "distance_data.xlsx"

# Save the DataFrame to an Excel file
df.to_excel(file_path, index=False)

# Define the size of each interval
interval_size = 1

# Initialize a dictionary to store counts for each interval
interval_counts = {}

# Iterate over the intervals and initialize counts to zero
for i in range(int(250 / interval_size) + 1):
    interval_counts[i * interval_size] = 0

# Sample list of numbers between 0 and 5

# Count how many values appear for each interval
for num in distances:
    for interval in interval_counts:
        if interval <= num < interval + interval_size:
            interval_counts[interval] += 1

# Print the counts for each interval
for interval, count in interval_counts.items():
    print(f"Interval {interval} - {interval + interval_size}: {count}")
    
df = pd.DataFrame(list(interval_counts.items()), columns=['Interval Start', 'Count'])

# Specify the file path to save the Excel file
file_path = "classes_data.xlsx"

# Save the DataFrame to an Excel file
df.to_excel(file_path, index=False)