import MTMM_DTW_run_batch as MTMM
import MTW_DTW_run_batch as MTW
import numpy as np
import os
import pandas as pd
import time

kabsch = [True]
preprocess = [True]
rotations_directory = "code/rotations"
rotations = os.listdir(rotations_directory)
subjects = 5
exercises = 1  
results_MTMM = np.zeros((subjects, exercises, len(rotations), len(kabsch), len(preprocess)))
results_MTW = np.zeros((subjects, exercises, len(rotations), len(kabsch), len(preprocess)))
confusion_matrix_MTMM = np.zeros((15,15))
confusion_matrix_MTW = np.zeros((15,15))

def run_each_exercise_and_subject():
    start_time = time.time()
    
    for algorithm in range(1, 2):
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
        preprocess_index = 0
        rotation_file_path = os.path.join(rotations_directory, rotation_file)
        
        for p in preprocess:
            kabsch_index = 0
            
            for k in kabsch:
                print("rotation_file: " + str(rotation_file) + "  preprocess: " + str(p) + "  kabsch: " + str(k) +"   subject: " + str(subject+1) + "  exercise: " + str(exercise+1) + "  algorithm: " + str(algorithm) )
                if(algorithm == 0):
                    acc, conf = MTMM.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, preprocess=p, kabsch=k)
                    results_MTMM[subject, exercise, rotation_index, preprocess_index, kabsch_index]= acc
                    add_confusion_matrix(subject, confusion_matrix_MTMM, conf)
                else:
                    acc, conf = MTW.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, preprocess=p, kabsch=k)
                    results_MTW[subject, exercise, rotation_index, preprocess_index, kabsch_index]= acc
                    add_confusion_matrix(subject, confusion_matrix_MTW, conf)

                kabsch_index += 1
                
            preprocess_index += 1 
            
        rotation_index += 1



def add_confusion_matrix(subject, conf, toadd):
    for i in range (0,3):
        row = i + subject *3
        for j in range (0,3):
            col = j + subject *3
            if(row==14 and col==14):
                print("subject: " + str(subject))
            conf[row,col] += toadd[i+1,j+1]
        conf[row, 14] += toadd[0,i+1]
        conf[14, row] += toadd[i+1,0]    

def print_results():
    print_results_MTMM()
    print_results_MTW()
    

def print_results_MTMM():
    results_MTMM = np.load("output_MTMM.npy")
    confusion_matrix_MTMM = np.load("output_confusion_MTMM.npy")
   
    df = pd.DataFrame(confusion_matrix_MTMM)
    excel_filename = 'output_MTMM.xlsx'
    df.to_excel(excel_filename, index=False, header=False)

    print(confusion_matrix_MTMM)
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
    #print(results_MTW)
    print("MTW averages out at: " + str(np.mean(results_MTW)))
    #print("MTW Kabsch Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 0])))
    #print("MTW  Preprocess: " + str(np.mean(results_MTW[:, :, :, 0, 1])))
    #print("MTW Kabsch : " + str(np.mean(results_MTW[:, :, :, 1, 0])))
    #print("MTW  : " + str(np.mean(results_MTW[:, :, :, 1, 1])))

run_each_exercise_and_subject()
print_results_MTW()