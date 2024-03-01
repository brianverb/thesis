import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import evaluation as eval
import loading as loader

results = np.empty((5, 8), dtype=object)
results_acc = np.zeros((5, 8))

# Replace 'file.xlsx' with the path to your Excel file
excel_file = 'matches_aras.csv'

# Replace x with the column number (0-indexed) you want to read
column_number = 5

df = pd.read_csv(excel_file)

change = False
result = []
previous_s = 1
previous_e = 1
# Iterate over each row
for index, row in df.iterrows():
    # Access individual elements of the row using column names
    subject = int(row['subject'])
    exercise = int(row['exercise'])
    start = int(row['start'])
    end = int(row['end'])
    label = int(row['classified_execution_type'])
    
    if previous_s != subject or previous_e != exercise:
        results[previous_s-1, previous_e-1] = result
        result = []
        previous_s = subject
        previous_e = exercise
    
    result.append((start,end, label-1))
        
    #print(f"Index: {subject}, Value1: {exercise}, Value2: {start}, Value3: {end}, Value3: {label}")
results[previous_s-1, previous_e-1] = result

for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        l = loader.Loading("code\data")
        l.load_all()
        subjects = l.time_series
        print("subject: "+ str(i) + "  exercise: " + str(j))
        templates, time_series = subjects[i][j][2]
        ground_truth = loader.Loading.get_ground_truth_labels(self=l, subject=i,exercise=j)

        EVALUATION = eval.evaluation(series=time_series, templates=templates, ground_truth=ground_truth)
        EVALUATION.ground_truth = ground_truth
        EVALUATION.found_truth = results[i,j]

        conf = EVALUATION.create_confusion_matrix_with_assignmentproblem()
        print(conf)
        acc = EVALUATION.exercise_accuracy(conf)
        print(acc)
        results_acc[i,j] = acc
        
print("Final acc: ")
print(np.mean(results_acc))