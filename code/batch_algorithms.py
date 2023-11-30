import MTMM_DTW_run_batch as MTMM
import MTW_DTW_run_batch as MTW
import numpy as np
import os
import pandas as pd

kabsch = [True, False]
preprocess = [True, False]
rotations_directory = "code/rotations"
rotations = os.listdir(rotations_directory)
subjects = 1
exercises = 1
results_MTMM = np.zeros((subjects, exercises, len(rotations), len(kabsch), len(preprocess)))
results_MTW = np.zeros((subjects, exercises, len(rotations), len(kabsch), len(preprocess)))
    
def run_each_exercise_and_subject():
    for subject in range(0, subjects):
        for exercise in range(0, exercises):
            for algorithm in range(0, 2):
                print("subject: " + str(subject) + "  exercise: " + str(exercise) + "  algorithm: " + str(algorithm))
                run_each_execution_setting(subject, exercise, 1, algorithm)
                    
    print(results_MTMM)
    print(results_MTW)           
    # Specify the file path for saving the array
    file_path_MTMM = 'output_MTMM.npy'
    file_path_MTW = 'output_MTW.npy'
    # Save the 5D array to a binary file in NumPy format
    np.save(file_path_MTMM, results_MTMM)
    np.save(file_path_MTW, results_MTW)
                
def run_each_execution_setting(subject, exercise, unit, algorithm):
    rotation_index = 0
    for rotation_file in rotations:
        preprocess_index = 0
        rotation_file_path = os.path.join(rotations_directory, rotation_file)
        
        for p in preprocess:
            kabsch_index = 0
            
            for k in kabsch:
                print("rotation_file: " + str(rotation_file) + "  preprocess: " + str(p) + "  kabsch: " + str(k) +"   subject: " + str(subject) + "  exercise: " + str(exercise) + "  algorithm: " + str(algorithm) )
                if(algorithm == 0):  
                    results_MTMM[subject, exercise, rotation_index, preprocess_index, kabsch_index],_ = MTMM.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, preprocess=p, kabsch=k)
                else:
                    results_MTW[subject, exercise, rotation_index, preprocess_index, kabsch_index],_ = MTW.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, preprocess=p, kabsch=k)
                kabsch_index += 1
                
            preprocess_index += 1 
            
        rotation_index += 1

def calculate_averages():
    return None

run_each_exercise_and_subject()
