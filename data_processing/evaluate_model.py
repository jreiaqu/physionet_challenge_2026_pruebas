#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d labels.csv -o predictions.csv -s scores.csv
#   python evaluate_model.py -d ../output_data/demographics.csv -o dummy   
#
# where 'labels.csv' is a CSV file containing the labels, 'predictions.csv' is a CSV file containing containing the predictions, and
# 'scores.csv' (optional) is a collection of scores for the predictions.
#
# The Challenge webpage describes the file formats and scoring functions.

import argparse
import numpy as np
import os
import os.path
import pandas as pd
import sys

id_patients = 'BDSPPatientID'
id_labels = 'Cognitive_Impairment'
id_binary_predictions = 'Cognitive_Impairment'
id_probability_predictions = 'Cognitive_Impairment_Probability'

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--labels_folder', type=str, required=True)
    parser.add_argument('-o', '--predictions_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Compute AUC.
def compute_auc(labels, predictions):
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(labels, predictions, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
    auprc = average_precision_score(labels, predictions, average='macro', pos_label=1, sample_weight=None)
    return auroc, auprc

# Compute accuracy.
def compute_accuracy(labels, predictions):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels, predictions, normalize=True, sample_weight=None)
    return accuracy

# Compute F-measure.
def compute_f_measure(labels, predictions):
    from sklearn.metrics import f1_score
    f_measure = f1_score(labels, predictions, pos_label=1, average='binary')
    return f_measure

# Evaluate the models.
def evaluate_model(demographics_file):
    # Load the single CSV containing both ground truth and predictions
    df = pd.read_csv(demographics_file)
    
    # Define exact column names based on your CSV header
    id_patients = 'BidsFolder' # We use this to ensure rows are distinct
    id_labels = 'Cognitive_Impairment'
    id_binary_predictions = 'Cognitive_Impairment_Prediction'
    id_probability_predictions = 'Cognitive_Impairment_Probability'
    
    # Ensure there are no duplicate patient rows (optional safety check)
    df.drop_duplicates(subset=[id_patients], keep='last', inplace=True)

    def standardize_bool(val):
        s = str(val).strip().upper()
        if s in ['TRUE', '1', '1.0', 'T', 'Y', 'YES']: return 1.0
        if s in ['FALSE', '0', '0.0', 'F', 'N', 'NO']: return 0.0
        return np.nan
    
    # Standardize the labels and predictions to be 0/1.
    df[id_labels] = df[id_labels].apply(standardize_bool)
    df[id_binary_predictions] = df[id_binary_predictions].apply(standardize_bool)

    # Clean the dataframe: Drop rows where Ground Truth is NaN
    df = df.dropna(subset=[id_labels])
    
    # Fill missing predictions with 0 to penalize the model instead of crashing
    df[id_binary_predictions] = df[id_binary_predictions].fillna(0.0)
    df[id_probability_predictions] = df[id_probability_predictions].fillna(0.0)

    # Extract NumPy arrays for Sklearn
    labels = df[id_labels].values
    binary_predictions = df[id_binary_predictions].values
    probability_predictions = df[id_probability_predictions].values

    # Evaluate the predictions.
    auroc, auprc = compute_auc(labels, probability_predictions)
    accuracy = compute_accuracy(labels, binary_predictions)
    f_measure = compute_f_measure(labels, binary_predictions)

    return auroc, auprc, accuracy, f_measure

# Run the code.
def run(args):
    # MODIFICADO: Ahora solo le pasamos un archivo (puedes usar args.labels_folder como la ruta a tu demographics.csv modificado)
    auroc, auprc, accuracy, f_measure = evaluate_model(args.labels_folder)

    output_string = \
        f'AUROC: {auroc:.3f}\n' \
        f'AUPRC: {auprc:.3f}\n' + \
        f'Accuracy: {accuracy:.3f}\n' \
        f'F-measure: {f_measure:.3f}\n'

    if args.score_file:
        with open(args.score_file, 'w') as f:
            f.write(output_string)
    else:
        print(output_string)
        
if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
