import numpy as np
import os

results_MTMM = np.load("output_MTMM.npy")
results_MTW = np.load("output_MTW.npy")

print("MTMM averages out at: " + str(np.mean(results_MTMM)))
print("MTW averages out at: " + str(np.mean(results_MTW)))


print("MTMM Kabsch Preprocess: " + str(np.mean(results_MTMM[:, :, :, 0, 0])))
print("MTMM  Preprocess: " + str(np.mean(results_MTMM[:, :, :, 0, 1])))
print("MTMM Kabsch : " + str(np.mean(results_MTMM[:, :, :, 1, 0])))
print("MTMM  : " + str(np.mean(results_MTMM[:, :, :, 1, 1])))

print("MTW Kabsch Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 0])))
print("MTW  Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 1])))
print("MTW Kabsch : " + str(np.mean(results_MTW[:, :, :, 1, 0])))
print("MTW  : " + str(np.mean(results_MTW[:, :, :, 1, 1])))