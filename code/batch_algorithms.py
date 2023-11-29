import MTMM_DTW_run_batch as MTMM
import MTW_DTW_run_batch as MTW
import numpy as np
import os
import pandas as pd

kabsch = [True, False]
preprocess = [True, False]
rotations_directory = "code/rotations"
rotations = os.listdir(rotations_directory)

def run_each_exercise_and_subject():
    results = np.zeros((2,5,8))
    for subject in range(0,5):
        for exercise in range(0,8):
            for algorithm in range(0,2):
                print("subject: " + str(subject) + "  exercise: " + str(exercise) + "  algorithm: " + str(algorithm))
                results = run_each_execution_setting(subject, exercise, 1, algorithm)
    df = pd.DataFrame(results)
    excel_file_path = 'results.xlsx'
    df.to_excel(excel_file_path, index=False, header=False)  # Set index and header to False to exclude row and column labels

def run_each_execution_setting(subject, exercise, unit, algorithm):
    results = np.zeros((2,5,8,len(rotations),len(kabsch)+len(preprocess)))

    rotation_index = 0
    for rotation_file in rotations:
        preprocess_index = 0
        rotation_file_path = os.path.join(rotations_directory, rotation_file)
        
        for p in preprocess:
            kabsch_index = 0
            
            for k in kabsch:
                print("rotation_file: " + str(rotation_file) + "  preprocess: " + str(p) + "  kabsch: " + str(k))
                if(algorithm == 0):  
                    results[algorithm, subject, exercise, rotation_index, preprocess_index*2+kabsch_index],_ = MTMM.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, preprocess=p, kabsch=k)
                else:
                    results[algorithm, subject, exercise, rotation_index, preprocess_index*2+kabsch_index],_ = MTW.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, preprocess=p, kabsch=k)
                kabsch_index += 1
                
            preprocess_index += 1
            
        rotation_index += 1
        
    return results

run_each_exercise_and_subject()
