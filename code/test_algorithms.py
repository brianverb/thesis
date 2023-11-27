import MTMM_DTW_run as MTMM
import MTW_DTW_run as MTW
import numpy as np

kabsch = [True, False]
preprocess = [True, False]
rotations = [""]


def run_each_exercise_and_subject():
    results = np.zeros(5,8)
    for s in range(0,4):
        for e in range(0,7):
            results[s-1,e-1] = run_each_execution_setting(s,e,1)

def run_each_execution_setting(subject, exercise, unit):
    results = np.zeros((len(rotations,len(kabsch)+len(preprocess))))

    rotation_index = 0
    for rotation_file in rotations:
        preprocess_index = 0
        
        for p in preprocess:
            kabsch_index = 0
            
            for k in kabsch:
                results[rotation_index, preprocess_index*2+kabsch_index] = MTMM.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file, preprocess=p, kabsch=k)
                kabsch_index += 1
                
            preprocess_index += 1
            
        rotation_index += 1
        
    return results