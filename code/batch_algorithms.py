"""
:author Brian Verbanck
:copyright: Copyright 2024 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import MTMM_DTW_run_batch as MTMM
import MTW_DTW_run_batch as MTW
import numpy as np
import os
import pandas as pd
import time
import evaluation as eval

kabsch = [True]
rotations_directory = "code/rotations"
rotations = os.listdir(rotations_directory)
subjects = 5
exercises = 8
results_MTMM = np.zeros((subjects, exercises, len(rotations), len(kabsch)), dtype=object)
results_MTW = np.zeros((subjects, exercises, len(rotations), len(kabsch)), dtype=object)
results_conf_MTW = np.zeros((5,4,4), dtype=int)
results_conf_MTMM = np.zeros((5,4,4), dtype=int)
def run_each_exercise_and_subject():
    start_time = time.time()
    
    for algorithm in range(1,2):
        for subject in range(0, subjects):
            for exercise in range(0, exercises):
                run_each_execution_setting(subject, exercise, 1, algorithm)
        
    
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the result
    print(f"Time elapsed: {elapsed_time} seconds")

                
def run_each_execution_setting(subject, exercise, unit, algorithm):
    rotation_index = 0
    if exercise == 1:
        unit = 3
           
    for rotation_file in rotations:
        rotation_file_path = os.path.join(rotations_directory, rotation_file)
         
        kabsch_index = 0
        
        for k in kabsch:
            print("rotation_file: " + str(rotation_file) + "  kabsch: " + str(k) +"   subject: " + str(subject+1) + "  exercise: " + str(exercise+1) + "  algorithm: " + str(algorithm) )
            if(algorithm == 0):
                acc, conf, amount_exercises = MTMM.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, kabsch=k)
                print(acc)
                print(conf)
                add_result_tuple(results_MTMM, results_conf_MTMM, subject, exercise, rotation_index, kabsch_index, conf, acc, amount_exercises)
            else:
                acc, conf, amount_exercises = MTW.run(subject=subject, exercise=exercise, unit=unit, rotation_file=rotation_file_path, kabsch=k)
                print(acc)
                print(conf)
                add_result_tuple(results_MTW,  results_conf_MTW, subject, exercise, rotation_index, kabsch_index, conf, acc, amount_exercises)
            kabsch_index += 1
            
            
        rotation_index += 1

def add_result_tuple(results, results_confs, subject, exercise, rotation_index, kabsch_index, conf, acc, amount_exercises):
    mp = 0
    mc = 0
    fp = 0
    correct = 0
    total = 0
     
    for i in range (0,4):
        for j in range (0,4):
            if j == 3:
                mp += conf[i,j]
            elif i == 3: 
                fp += conf[i,j]
            elif i == j:
                correct += conf[i,j]
            else:
                mc += conf[i,j]
            total += conf[i,j]
            results_confs[subject,i,j] += int(conf[i,j])
    #print(f"mp: {mp}, fp: {fp}, correct: {correct}, mc: {mc}, total: {total}, expected amount: {amount_exercises}")
    # Accuracy, missed prediction rate, false prediction rate, miss classified rate, total found exercises, expected exercises
    results[subject,exercise,rotation_index, kabsch_index] = (correct/total, mp/amount_exercises, fp/(total-mp), mc/(total-mp), total-mp, amount_exercises)

def print_results():
    print_results_MTW()
    #print_results_MTMM()
    

def print_results_MTMM():
    for i in range (0,5):
        confusion_matrix = results_conf_MTMM[i]
        #print(confusion_matrix)
        EVAL = eval.evaluation(confusion_matrix)
        #EVAL.plot_simple_confusion_matrix(confusion_matrix)
    
    print("MTMM:")
    print("accuracy: " + str(np.mean(np.array([x[0] for x in np.ravel(results_MTMM)]))))
    print("missed prediction rate: " + str(np.mean(np.array([x[1] for x in np.ravel(results_MTMM)]))))
    print("false prediciton rate: " + str(np.mean(np.array([x[2] for x in np.ravel(results_MTMM)]))))
    print("miss classified rate: " + str(np.mean(np.array([x[3] for x in np.ravel(results_MTMM)]))))
    print("total exercises found: " + str(np.sum(np.array([x[4] for x in np.ravel(results_MTMM)]))))
    print("exercises expected: " + str(np.sum(np.array([x[5] for x in np.ravel(results_MTMM)]))))
    print("__________________________________________________________________________________________________")
    
def print_results_MTW():
    
    for i in range (0,5):
        confusion_matrix = results_conf_MTW[i]
        confusion_matrix *= 10
        EVAL = eval.evaluation(confusion_matrix)
        EVAL.plot_simple_confusion_matrix(confusion_matrix)
    
    print("MTW:")
    print(results_conf_MTW)
    
    print("accuracy: " + str(np.mean(np.array([x[0] for x in np.ravel(results_MTW)]))))
    print("missed prediction rate: " + str(np.mean(np.array([x[1] for x in np.ravel(results_MTW)]))))
    print("false prediciton rate: " + str(np.mean(np.array([x[2] for x in np.ravel(results_MTW)]))))
    print("miss classified rate: " + str(np.mean(np.array([x[3] for x in np.ravel(results_MTW)]))))
    print("total exercises found: " + str(np.sum(np.array([x[4] for x in np.ravel(results_MTW)]))))
    print("exercises expected: " + str(np.sum(np.array([x[5] for x in np.ravel(results_MTW)]))))
    print("__________________________________________________________________________________________________")

run_each_exercise_and_subject()
print_results()