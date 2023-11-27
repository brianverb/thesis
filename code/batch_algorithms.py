import MTMM_DTW_run_batch as MTMM
import MTW_DTW_run_batch as MTW
import numpy as np

kabsch = [True, False]
preprocess = [True, False]
rotations = [""]


def run_each_exercise_and_subject():
    results = np.zeros(2,5,8)
    for subject in range(0,4):
        for exercise in range(0,7):
            for algorithm in range(0,1):
                results[algorithm, subject-1, exercise-1] = run_each_execution_setting(subject, exercise, 1, algorithm)

def run_each_execution_setting(subject, exercise, unit, algorithm):
    results = np.zeros((len(rotations,len(kabsch)+len(preprocess))))

    rotation_index = 0
    for rotation_file in rotations:
        preprocess_index = 0
        
        for p in preprocess:
            kabsch_index = 0
            
            for k in kabsch:
                if(algorithm == 0):  
                    results[rotation_index, preprocess_index*2+kabsch_index] = MTMM.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file, preprocess=p, kabsch=k)
                else:
                    results[rotation_index, preprocess_index*2+kabsch_index] = MTW.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file, preprocess=p, kabsch=k)
                kabsch_index += 1
                
            preprocess_index += 1
            
        rotation_index += 1
        
    return results