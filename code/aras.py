import numpy as np

def print_results():
    print_results_MTMM()
    print_results_MTW()

def print_results_MTMM():
    results_MTMM = np.load("output_MTMM.npy")

    #print(results_MTMM)
    print("MTMM averages out at: " + str(np.mean(results_MTMM)))
    print("MTMM Kabsch Preprocess: " + str(np.mean(results_MTMM[:, :, :, 0, 0])))
    print("MTMM  Preprocess: " + str(np.mean(results_MTMM[:, :, :, 0, 1])))
    print("MTMM Kabsch : " + str(np.mean(results_MTMM[:, :, :, 1, 0])))
    print("MTMM  : " + str(np.mean(results_MTMM[:, :, :, 1, 1])))
    
def print_results_MTW():
    results_MTW = np.load("output_MTW.npy")
    
    #print(results_MTW)
    print("MTW averages out at: " + str(np.mean(results_MTW)))
    print("MTW Kabsch Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 0])))
    print("MTW  Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 1])))
    print("MTW Kabsch : " + str(np.mean(results_MTW[:, :, :, 1, 0])))
    print("MTW  : " + str(np.mean(results_MTW[:, :, :, 1, 1])))
    
print_results()