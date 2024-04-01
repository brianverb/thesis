"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import numpy as np
import pandas as pd
'''
distances = []
distances = np.save("miss.npy",  distances)

'''
distances = np.load("correct.npy", allow_pickle=True)

# Convert the ndarray to a pandas DataFrame
df = pd.DataFrame(distances)

file_path = "correct.xlsx"

# Save the DataFrame to an Excel file
df.to_excel(file_path, index=False)


