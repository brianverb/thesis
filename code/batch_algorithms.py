import MTMM_DTW_run_batch as MTMM
import MTW_DTW_run_batch as MTW
import numpy as np
import os
import pandas as pd
import time

kabsch = [True]
rotations_directory = "code/rotations"
rotations = os.listdir(rotations_directory)
subjects = 5
exercises = 8 
results_MTMM = np.zeros((subjects, exercises, len(rotations), len(kabsch)))
results_MTW = np.zeros((subjects, exercises, len(rotations), len(kabsch)))
confusion_matrix_MTMM = np.zeros((16,16))
confusion_matrix_MTW = np.zeros((16,16))

def run_each_exercise_and_subject():
    start_time = time.time()
    
    for algorithm in range(1,2):
        for subject in range(0, subjects):
            for exercise in range(0, exercises):
                run_each_execution_setting(subject, exercise, 1, algorithm)
        
        # Specify the file path for saving the array
        file_path_MTMM = 'output_MTMM.npy'
        file_path_MTW = 'output_MTW.npy'
        file_path_confusion_MTMM = 'output_confusion_MTMM.npy'
        file_path_confusion_MTW = 'output_confusion_MTW.npy'
        # Save the 5D array to a binary file in NumPy format
        np.save(file_path_MTMM, results_MTMM)
        np.save(file_path_MTW, results_MTW)
        np.save(file_path_confusion_MTMM, confusion_matrix_MTMM)
        np.save(file_path_confusion_MTW, confusion_matrix_MTW)
        
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the result
    print(f"Time elapsed: {elapsed_time} seconds")

                
def run_each_execution_setting(subject, exercise, unit, algorithm):
    rotation_index = 0
    for rotation_file in rotations:
        rotation_file_path = os.path.join(rotations_directory, rotation_file)
        
        kabsch_index = 0
        
        for k in kabsch:
            print("rotation_file: " + str(rotation_file) + "  kabsch: " + str(k) +"   subject: " + str(subject+1) + "  exercise: " + str(exercise+1) + "  algorithm: " + str(algorithm) )
            if(algorithm == 0):
                acc, conf = MTMM.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, kabsch=k)
                print(acc)
                print(conf)
                results_MTMM[subject, exercise, rotation_index, kabsch_index]= acc
                add_confusion_matrix(subject, confusion_matrix_MTMM, conf)
            else:
                acc, conf = MTW.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, kabsch=k)
                print(acc)
                print(conf)
                results_MTW[subject, exercise, rotation_index, kabsch_index]= acc
                add_confusion_matrix(subject, confusion_matrix_MTW, conf)

            kabsch_index += 1
            
            
        rotation_index += 1



def add_confusion_matrix(subject, conf, toadd):
    for i in range (0,3):
        row = i + subject *3
        for j in range (0,3):
            col = j + subject *3
            conf[row,col] += toadd[i+1,j+1]
        conf[row, 15] += toadd[0,i+1]
        conf[15, row] += toadd[i+1,0]    

def print_results():
    print_results_MTW()
    print_results_MTMM()
    

def print_results_MTMM():
    results_MTMM = np.load("output_MTMM.npy")
    confusion_matrix_MTMM = np.load("output_confusion_MTMM.npy")
   
    df = pd.DataFrame(confusion_matrix_MTMM)
    excel_filename = 'output_MTMM.xlsx'
    df.to_excel(excel_filename, index=False, header=False)

    print(confusion_matrix_MTMM)
    print(np.sum(confusion_matrix_MTMM[-1, :]))
    print(np.sum(confusion_matrix_MTMM[:, -1]))
    print(np.sum(confusion_matrix_MTMM[:,:]))
    print("Miss classified: ")
    miss_classified_MMTW=0
    for i in range(15):
        for j in range(15):
            if i != j and j != 15 and i != 15:
                miss_classified_MMTW += confusion_matrix_MTMM[i,j]
    print(miss_classified_MMTW)
    
    #print(results_MTMM)
    print("MTMM averages out at: " + str(np.mean(results_MTMM)))
    #print("MTMM Kabsch Preprocess: " + str(np.mean(results_MTMM[:, :, :, 0, 0])))
    #print("MTMM  Preprocess: " + str(np.mean(results_MTMM[:, :, :, 0, 1])))
    #print("MTMM Kabsch : " + str(np.mean(results_MTMM[:, :, :, 1, 0])))
    #print("MTMM  : " + str(np.mean(results_MTMM[:, :, :, 1, 1])))
    
def print_results_MTW():
    results_MTW = np.load("output_MTW.npy")
    confusion_matrix_MTW = np.load("output_confusion_MTW.npy")

    df = pd.DataFrame(confusion_matrix_MTW)
    excel_filename = 'output_MTW.xlsx'
    df.to_excel(excel_filename, index=False, header=False)
    
    print(confusion_matrix_MTW)
    print(np.sum(confusion_matrix_MTW[-1, :]))
    print(np.sum(confusion_matrix_MTW[:, -1]))
    print(np.sum(confusion_matrix_MTW[:,:]))
    print("Miss classified: ")
    miss_classified_MTW=0
    for i in range(15):
        for j in range(15):
            if i != j and j != 15 and i != 15:
                miss_classified_MTW += confusion_matrix_MTW[i,j]
    print(miss_classified_MTW)
    #print(results_MTW)
    print("MTW averages out at: " + str(np.mean(results_MTW)))
    #print("MTW Kabsch Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 0])))
    #print("MTW  Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 1])))
    #print("MTW Kabsch : " + str(np.mean(results_MTW[:, :, :, 1, 0])))
    #print("MTW  : " + str(np.mean(results_MTW[:, :, :, 1, 1])))

run_each_exercise_and_subject()
print_results()