import numpy as np
import os

results_MTMM = np.load("output_MTMM.npy")
results_MTW = np.load("output_MTW.npy")

print("MTMM averages out at: " + str(np.mean(results_MTMM)))
print("MTW averages out at: " + str(np.mean(results_MTW)))