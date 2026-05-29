#!/usr/bin/env python
"""
train_model.py
==============
Punto de entrada CLI para entrenar un modelo desde cero.
Llama a models_and_training.train_model() con los argumentos de línea de comandos.

Uso: python train_model.py -d <carpeta_datos> -m <carpeta_modelo> [-v]
  -d  carpeta con datos de entrenamiento (EDF + demographics_total.csv)
  -m  carpeta donde se guardará el checkpoint del modelo
  -v  modo verboso
"""

import argparse
import sys

from data_utils import *
from models_and_training import train_model

# Parse arguments.
def get_parser():
    description = 'Train the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

# Run the code.
def run(args):
    train_model(args.data_folder, args.model_folder, args.verbose) ### Teams: Implement this function!!!

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))