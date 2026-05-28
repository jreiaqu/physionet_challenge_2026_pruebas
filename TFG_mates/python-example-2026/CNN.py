 #!/usr/bin/env

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import shutil
import glob
from sklearn.metrics import roc_auc_score

from helper_code import *

import math
from typing import Tuple, Optional, Dict
################################################################################
# Path & Constant Configuration (Added for Robustness)
################################################################################

# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

N_CHANNELS = 9

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, model_type, verbose, csv_path, n_blocks=2, initial_filters=16, kernel_size=32, pool_size=3, stride=1, padding=2, adaptive_pool=250, dense_units=256, dropout_rate=0.3, max_pool=0, lr=0.0005, bs=16, ne=500, **kwargs): # ** empaqueta diccionario!!! por estar en def
                                                                                                                                                                                                                                                                                # kwargs junta todas las entradas no especificadas (en diccionario), por eso tiene que empaquetarse
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    device = torch.device("cpu")
    
    print(device)                                                                                                                                                                                                                                                                   # la clave es poner **cualquier_cosa (hace la función de empaquetado, usar kwargs es convención)
    t_num = kwargs.get('current_trial', '?') # dict.get('clave', 'valor_por_defecto')
    s_num = kwargs.get('current_split', '?')

    patient_metadata_list = find_patients(csv_path)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    lstm_hidden = kwargs.get('lstm_hidden', 64)
    lstm_layers = kwargs.get('lstm_layers', 1)
    d_model = kwargs.get('d_model', 64)
    bidirectional = kwargs.get('bidirectional', True)
    nhead = kwargs.get('nhead', True)
    num_layers = kwargs.get('num_layers', True)
    dim_feedforward = kwargs.get('dim_feedforward', True)

    if model_type == 0:
        if verbose: print("Instanciando: CNN Pura")
        model = SimpleCNN1D(
            n_channels=N_CHANNELS,
            n_blocks=n_blocks, 
            initial_filters=initial_filters, 
            kernel_size=kernel_size, 
            pool_size=pool_size, 
            stride=stride, 
            padding=padding, 
            adaptive_pool=adaptive_pool, 
            dense_units=dense_units, 
            dropout_rate=dropout_rate, 
            max_pool=max_pool, 
            lr=lr, 
            bs=bs, 
            ne=ne
        )
    elif model_type == 1:
        if verbose: print("Instanciando: CNN + LSTM")
        model = CNNLSTM1D(
            n_channels=N_CHANNELS,
            n_blocks=n_blocks, 
            initial_filters=initial_filters, 
            kernel_size=kernel_size, 
            pool_size=pool_size, 
            stride=stride, 
            padding=padding, 
            adaptive_pool=adaptive_pool, 
            lstm_hidden=lstm_hidden, 
            lstm_layers=lstm_layers, 
            bidirectional=bidirectional,
            dense_units=dense_units, 
            dropout_rate=dropout_rate, 
            max_pool=max_pool, 
            lr=lr, 
            bs=bs, 
            ne=ne
        )
    elif model_type == 2:
        if verbose: print("Instanciando: LSTM")
        model = LSTM(
            n_channels=N_CHANNELS,
            lstm_hidden=lstm_hidden, 
            lstm_layers=lstm_layers, 
            bidirectional=bidirectional,
            dense_units=dense_units, 
            dropout_rate=dropout_rate, 
            lr=lr, 
            bs=bs, 
            ne=ne
        )
        # Dentro de tu if/elif de train_model
    elif model_type == 3:
        if verbose: print("Instanciando: ResNet1D")
        model = ResNet1D(
            n_channels=N_CHANNELS, initial_filters=initial_filters, n_blocks=n_blocks, 
            kernel_size=kernel_size, padding=padding, # ESTO ES NUEVO
            dense_units=dense_units, dropout_rate=dropout_rate, adaptive_pool=adaptive_pool,
            lr=lr, bs=bs, ne=ne
        )
    elif model_type == 4:
        if verbose: print("Instanciando: Time-Series Transformer")
        model = TimeSeriesTransformer(n_channels=N_CHANNELS, d_model=d_model, nhead=nhead, 
                 num_layers=num_layers, dim_feedforward=dim_feedforward, dropout_rate=dropout_rate,
                 lr=lr, bs=bs, ne=ne)
    else:
        raise ValueError("Modelo no soportado.")
        
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=1e-4)

    num_pos = 0
    num_neg = 0
    for record in patient_metadata_list:
        p_id = record[HEADERS['bids_folder']]
        label = float(load_diagnoses(csv_path, p_id))
        if label == 1.0:
            num_pos += 1
        else:
            num_neg += 1
            
    peso_calc = (num_neg / num_pos) if num_pos > 0 else 1.0
    peso_tensor = torch.tensor([peso_calc], dtype=torch.float32).to(device)
    
    if verbose: 
        print(f"Balance de clases (Train): {num_pos} Pos | {num_neg} Neg")
        print(f"Aplicando pos_weight dinámico: {peso_calc:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=peso_tensor)
    
    batch_size = model.bs 
    num_epochs = model.ne 

    batches_per_epoch = math.ceil(num_records / batch_size)
    total_steps = num_epochs * batches_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=model.lr,               # Pico máximo del learning rate
        total_steps=total_steps,       # Total de veces que se hará optimizer.step()
        pct_start=0.1,                 # 10% de warmup (subida)
        anneal_strategy='cos'          # Enfriamiento en coseno
    )

    best_val_auc = 0.0
    best_epoch_num = 1
    patience = 500
    epochs_without_improvement = 0
    recent_val_aucs = []

    for epoch in range(num_epochs):
            
        batch_data = []
        batch_labels = []
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0
        desc_str = f"Trial {t_num} | Split {s_num} | Epoch {epoch+1}/{num_epochs}"
        pbar = tqdm(range(num_records), desc=desc_str, unit="patient", disable=not verbose, leave=False)
    
        for i in pbar:
            try:
                record = patient_metadata_list[i]
                patient_id = record[HEADERS['bids_folder']]
                site_id    = record[HEADERS['site_id']]
                session_id = record[HEADERS['session_id']]
                
                if verbose:
                    pbar.set_postfix({"patient": patient_id})

                algorithmic_annotations_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
                if not os.path.exists(algorithmic_annotations_file):
                    if verbose: tqdm.write(f"Missing file: {patient_id}. Skipping...")
                    continue # directamente siguiente iter
                algorithmic_annotations, algorithmic_fs = load_signal_data(algorithmic_annotations_file)

                channel_order = [
                    'caisr_prob_no-ar', 'caisr_prob_arous', 'limb_caisr', 
                    'resp_caisr', 'caisr_prob_n3', 'caisr_prob_n2', 
                    'caisr_prob_n1', 'caisr_prob_r', 'caisr_prob_w'
                ]

                if not all(ch in algorithmic_annotations for ch in channel_order):
                    if verbose: tqdm.write(f"Incomplete channels in {patient_id}. Skipping...")
                    continue

                algorithmic_annotations['limb_caisr'] = algorithmic_annotations['limb_caisr'] / 2.0
                
                resp_signal = algorithmic_annotations['resp_caisr']
                resp_mapped = np.zeros_like(resp_signal, dtype=float)
                resp_mapped[(resp_signal == 4) | (resp_signal == 5)] = 0.5
                resp_mapped[(resp_signal == 1) | (resp_signal == 2) | (resp_signal == 3)] = 1.0
                algorithmic_annotations['resp_caisr'] = resp_mapped

                algorithmic_annotations, _ = match_sampling_rate(algorithmic_annotations, algorithmic_fs,"30s")

                signal_matrix = np.array([algorithmic_annotations[ch] for ch in channel_order])

                label = load_diagnoses(csv_path, patient_id)

                # Convertimos las etiquetas a float porque BCEWithLogitsLoss lo requiere
                batch_data.append(torch.tensor(signal_matrix, dtype=torch.float32))
                batch_labels.append(torch.tensor([float(label)], dtype=torch.float32))
                
                del signal_matrix 
                del algorithmic_annotations
                
                if len(batch_data) == batch_size or i == num_records - 1:
                    # (Canales, Datos)
                    max_len = max([tensor.shape[1] for tensor in batch_data])

                    padded_data = []
                    for tensor in batch_data:
                        pad_amount = max_len - tensor.shape[1]
                        
                        # F.pad rellena empezando por la última dimensión (el tiempo en nuestro caso).
                        # El formato es (padding_izquierdo, padding_derecho).
                        padded_tensor = F.pad(tensor, (0, pad_amount), mode="constant", value=0.0)
                        padded_data.append(padded_tensor)

                    # tensor_batch tendrá forma: (Batch, Canales, Tiempo)
                    tensor_batch = torch.stack(padded_data).to(device) 
                    tensor_labels = torch.stack(batch_labels).to(device)

                    optimizer.zero_grad()
                    predictions = model(tensor_batch)
                    loss = criterion(predictions, tensor_labels)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    
                    scheduler.step()

                    running_train_loss += loss.item()
                    num_train_batches += 1

                    del tensor_batch
                    del tensor_labels
                    batch_data.clear()
                    batch_labels.clear()
                    padded_data.clear()
                    # break # para ver si la red aprende con un solo batch 

            except Exception as e:
                # If an error occurs (e.g., a record is corrupted), log it and move to the next
                tqdm.write(f"  !!! Error processing record {i+1} ({patient_id}): {e}")
                continue

        pbar.close()

        model.eval() 
        
        val_csv_path = kwargs.get('val_csv_path') 
        val_data_folder = kwargs.get('val_data_folder')
        
        if val_csv_path and val_data_folder:
            y_true_val = []
            y_probs_val = []
            
            running_val_loss = 0.0
            num_val_records_processed = 0
            
            val_metadata_list = find_patients(val_csv_path)
            
            for val_record in val_metadata_list:
                val_patient_id = val_record[HEADERS['bids_folder']]
                val_label = float(load_diagnoses(val_csv_path, val_patient_id))
                
                logits_val, _, prob_val = run_model(model, val_record, val_data_folder, verbose=False)
                
                if logits_val is not None:
                    y_true_val.append(val_label)
                    y_probs_val.append(prob_val)
                    
                    label_tensor_val = torch.tensor([[val_label]], dtype=torch.float32).to(device)
                    
                    val_loss = criterion(logits_val, label_tensor_val)
                    running_val_loss += val_loss.item()
                    
                    num_val_records_processed += 1
                else:
                    continue

            try: 
                if len(y_true_val) > 0:
                    current_val_auc = roc_auc_score(y_true_val, y_probs_val)
                else:
                    current_val_auc = 0.5
            except ValueError: 
                current_val_auc = 0.5
                
            epoch_train_loss = running_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
            epoch_val_loss = running_val_loss / num_val_records_processed if num_val_records_processed > 0 else float('inf')

            if verbose:
                print(f"   -> Época {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {current_val_auc:.4f}")                
            
            # --- LÓGICA DE EARLY STOPPING ---

            # 1. Actualizar la cola de las últimas 3 épocas
            recent_val_aucs.append(current_val_auc)
            if len(recent_val_aucs) > 3:
                recent_val_aucs.pop(0)
                
            # 2. Calcular la media móvil
            smoothed_val_auc = sum(recent_val_aucs) / len(recent_val_aucs)

            if epoch >=15 and smoothed_val_auc > best_val_auc:
                best_val_auc = smoothed_val_auc
                best_epoch_num = epoch + 1
                epochs_without_improvement = 0
                save_model(model_folder, model)
                if verbose:
                    print(f"      ¡Nuevo récord SÓLIDO (Media 3 épocas: {smoothed_val_auc:.4f})! Modelo guardado.")
            elif epoch >= 15:
                epochs_without_improvement += 1
                if verbose:
                    print(f"      No mejora. Media actual: {smoothed_val_auc:.4f}. Paciencia: {epochs_without_improvement}/{patience}")

                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early Stopping activado en la época {epoch+1}. La media del Val AUC no mejora desde hace {patience} épocas.")
                    break
            else:
                # Épocas de 0 a 9: La red está calentando (pesos aleatorios)
                if verbose:
                    print(f"      (Fase de Warm-up. Media actual: {smoothed_val_auc:.4f}. No se evalúan récords aún)")                    
                 
                    
        else:
            # --- NUEVO: Qué hacer si NO hay validación ---
            epoch_train_loss = running_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
            
            if verbose:
                print(f"   -> Época {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | (Sin validación)")
                
            best_epoch_num = epoch + 1
            save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()
    
    return best_epoch_num
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.pth')
    checkpoint = torch.load(model_filename, map_location=torch.device('cpu'), weights_only=False)
    hp = checkpoint['hyperparams']
    m_type = hp.get('model_type', 0) # si no lo tiene, por defecto 0 (CNN)
    
    if m_type == 0:
        if verbose: print("Cargando arquitectura: CNN Pura")
        model = SimpleCNN1D(
            n_channels=hp['n_channels'], n_blocks=hp['n_blocks'], initial_filters=hp['initial_filters'], 
            kernel_size=hp['kernel_size'], pool_size=hp['pool_size'], stride=hp['stride'], 
            padding=hp.get('padding', 2), adaptive_pool=hp['adaptive_pool'], 
            dense_units=hp['dense_units'], dropout_rate=hp['dropout_rate'], 
            max_pool=hp.get('max_pool', 0), lr=hp.get('lr', 0.0001), 
            bs=hp.get('bs', 32), ne=hp.get('ne', 30)
        )
    elif m_type == 1:
        if verbose: print("Cargando arquitectura: CNN + LSTM")
        model = CNNLSTM1D(
            n_channels=hp['n_channels'], n_blocks=hp['n_blocks'], initial_filters=hp['initial_filters'], 
            kernel_size=hp['kernel_size'], pool_size=hp['pool_size'], stride=hp['stride'], 
            padding=hp.get('padding', 2), adaptive_pool=hp['adaptive_pool'], 
            lstm_hidden=hp.get('lstm_hidden', 128), lstm_layers=hp.get('lstm_layers', 1),
            bidirectional=hp.get('bidirectional', True),
            dense_units=hp['dense_units'], dropout_rate=hp['dropout_rate'], 
            max_pool=hp.get('max_pool', 0), lr=hp.get('lr', 0.0001), 
            bs=hp.get('bs', 32), ne=hp.get('ne', 30)
        )
    elif m_type == 2:
        if verbose: print("Cargando arquitectura: LSTM")
        model = LSTM(
            n_channels=hp.get('n_channels'),
            lstm_hidden=hp.get('lstm_hidden', 128), lstm_layers=hp.get('lstm_layers', 1),
            bidirectional=hp.get('bidirectional', True),
            dense_units=hp['dense_units'], dropout_rate=hp['dropout_rate'],
            lr=hp.get('lr', 0.0001), 
            bs=hp.get('bs', 32), ne=hp.get('ne', 30)
        )
    elif m_type == 3:
        if verbose: print("Cargando arquitectura: ResNet1D")
        model = ResNet1D(
            n_channels=hp.get('n_channels', 9),
            initial_filters=hp.get('initial_filters', 32), 
            n_blocks=hp.get('n_blocks', 3), 
            kernel_size=hp.get('kernel_size', 3), # ESTO ES NUEVO
            padding=hp.get('padding', 1),         # ESTO ES NUEVO
            dense_units=hp.get('dense_units', 128), 
            dropout_rate=hp.get('dropout_rate', 0.5), 
            adaptive_pool=hp.get('adaptive_pool', 1),
            lr=hp.get('lr', 0.0001), 
            bs=hp.get('bs', 32), 
            ne=hp.get('ne', 60)
        )
    elif m_type == 4:
        if verbose: print("Cargando arquitectura: Time-Series Transformer")
        model = TimeSeriesTransformer(
            n_channels=hp.get('n_channels', 9),
            d_model=hp.get('d_model', 64), 
            nhead=hp.get('nhead', 4), 
            num_layers=hp.get('num_layers', 3), 
            dim_feedforward=hp.get('dim_feedforward', 128), 
            dropout_rate=hp.get('dropout_rate', 0.3),
            lr=hp.get('lr', 0.0001), 
            bs=hp.get('bs', 32), 
            ne=hp.get('ne', 60)
        )
    else:
        raise ValueError(f"Tipo de modelo {m_type} no soportado o corrupto.")
        
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(model, record, data_folder, verbose):
    try:    
        # Extract identifiers from the record dictionary
        patient_id = record[HEADERS['bids_folder']]
        site_id    = record[HEADERS['site_id']]
        session_id = record[HEADERS['session_id']]

        channel_order = [
                'caisr_prob_no-ar', 'caisr_prob_arous', 'limb_caisr', 
                'resp_caisr', 'caisr_prob_n3', 'caisr_prob_n2', 
                'caisr_prob_n1', 'caisr_prob_r', 'caisr_prob_w'
        ]

        algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
        if os.path.exists(algo_file):
            algo_data, algo_data_fs = load_signal_data(algo_file)
            # añadir verbose de ref_ch
            ref_ch = list(algo_data.keys())[0] if len(algo_data) > 0 else None
            
            if ref_ch:
                ref_signal = algo_data[ref_ch]
                ref_fs = algo_data_fs[ref_ch]
                
                if 'limb_caisr' in algo_data:
                    algo_data['limb_caisr'] = algo_data['limb_caisr'] / 2.0
                else: # inferencia: hay que sacar predicción sí o sí, mejor poner 0 y predecir con el resto
                    if verbose: print(f"Warning: 'limb_caisr' not found in {patient_id}.")
                    algo_data['limb_caisr'] = np.zeros_like(ref_signal, dtype=float)
                    algo_data_fs['limb_caisr'] = ref_fs
                
                if 'resp_caisr' in algo_data:
                    resp_signal = np.array(algo_data['resp_caisr'])
                    resp_mapped = np.zeros_like(resp_signal, dtype=float)
                    resp_mapped[(resp_signal == 4) | (resp_signal == 5)] = 0.5
                    resp_mapped[(resp_signal == 1) | (resp_signal == 2) | (resp_signal == 3)] = 1.0
                    algo_data['resp_caisr'] = resp_mapped
                else:
                    if verbose: print(f"Warning: 'resp_caisr' not found in {patient_id}.")
                    algo_data['resp_caisr'] = np.zeros_like(ref_signal, dtype=float)
                    algo_data_fs['resp_caisr'] = ref_fs
                    
                for ch in channel_order:
                    if ch not in algo_data:
                        if verbose: print(f"Warning: '{ch}' not found in {patient_id}.")
                        algo_data[ch] = np.zeros_like(ref_signal, dtype=float)
                        algo_data_fs[ch] = ref_fs

                algorithmic_annotations, _ = match_sampling_rate(algo_data, algo_data_fs,"30s")
                
                signal_matrix = np.array([algorithmic_annotations[ch] for ch in channel_order])
                signal_matrix_tensor = torch.tensor(signal_matrix, dtype=torch.float32)
                
            else:
                if verbose: print(f"Warning: File exists but is empty for {patient_id}.")
                return None, False, 0.0
        else:
            if verbose: print(f"Warning: Missing file for {patient_id}.")
            return None, False, 0.0
        
        model.eval()
        device = next(model.parameters()).device
        input_tensor = signal_matrix_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probability_output = torch.sigmoid(logits).item()

        binary_output = bool(probability_output >= 0.5) 

    except Exception as e:
        if verbose: print(f"Unexpected inference error for {record.get('bids_folder', 'Unknown')}: {e}")
        logits = None
        binary_output = False
        probability_output = 0.0

    return logits, binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

class SimpleCNN1D(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, n_blocks=2, initial_filters=32, kernel_size=5, pool_size=2, stride=1, padding=2, adaptive_pool=50, dense_units=128, dropout_rate=0.5, max_pool=1, lr=0.0001, bs=16, ne=60):
        super(SimpleCNN1D, self).__init__()

        self.model_type = 0
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.adaptive_pool = adaptive_pool
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.max_pool = max_pool
        self.lr = lr
        self.bs = bs
        self.ne = ne

        layers = []
        in_ch = n_channels
        out_ch = initial_filters

        for i in range(n_blocks):
            layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
            
            in_ch = out_ch
            out_ch = out_ch * 2 # decisión de diseño, a mayor profundidad, más características
                                # hacerlo hiper?
        self.conv_blocks = nn.Sequential(*layers) # * desempaqueta la lista 
        
        if max_pool==1:
            self.adaptive_pool_layer = nn.AdaptiveMaxPool1d(adaptive_pool)
        else: 
            self.adaptive_pool_layer = nn.AdaptiveAvgPool1d(adaptive_pool)

        flatten_size = in_ch * adaptive_pool
        
        self.fc1 = nn.Linear(in_features=flatten_size, out_features=dense_units)
        self.relu_fc = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=dense_units, out_features=1)
      
    def forward(self, x):
        # x entra con forma: (Batch, Canales, Tiempo) -> (B, N_CHANNELS, X)
        x = self.conv_blocks(x)
        x = self.adaptive_pool_layer(x)
        
        x = x.view(x.size(0), -1) # fija tamaño batch y multiplica (reduciendo) dimensiones canal*tiempo, x ptos canal 1, x ptos canal 2...
        
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNLSTM1D(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, n_blocks=2, initial_filters=32, kernel_size=5, pool_size=2, stride=1, padding=2, adaptive_pool=50, lstm_hidden=128, lstm_layers=1, bidirectional=True, dense_units=128, dropout_rate=0.5, max_pool=1, lr=0.0001, bs=16, ne=60):
        super(CNNLSTM1D, self).__init__()

        self.model_type = 1
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.adaptive_pool = adaptive_pool
        # --- NUEVOS PARÁMETROS LSTM ---
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        # ------------------------------
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.max_pool = max_pool
        self.lr = lr
        self.bs = bs
        self.ne = ne

        layers = []
        in_ch = n_channels
        out_ch = initial_filters

        for i in range(n_blocks):
            layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
            
            in_ch = out_ch
            out_ch = out_ch * 2 
            
        self.conv_blocks = nn.Sequential(*layers) 
        
        if max_pool==1:
            self.adaptive_pool_layer = nn.AdaptiveMaxPool1d(adaptive_pool)
        else: 
            self.adaptive_pool_layer = nn.AdaptiveAvgPool1d(adaptive_pool)
        
        self.lstm = nn.LSTM(
            input_size=in_ch, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        lstm_out_features = lstm_hidden * 2 if bidirectional else lstm_hidden # Al ser bidireccional, la salida oculta se multiplica por 2
        
        self.fc1 = nn.Linear(in_features=lstm_out_features, out_features=dense_units)
        self.relu_fc = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=dense_units, out_features=1)
      
    def forward(self, x):
        # x: (Batch, Canales, Tiempo)
        x = self.conv_blocks(x)
        
        # Descomentar si 2 Hz, aplicar lógica para elegir según el caso?
        # x = self.adaptive_pool_layer(x)
        
        # Reordenar para la LSTM: (Batch, Tiempo, Canales)
        x = x.permute(0, 2, 1) 
        
        lstm_out, _ = self.lstm(x)
        
        # REVISAR PROMEDIO, si hacerlo dense_units poo sentido
        # Agrupar la secuencia entera promediándola sobre el tiempo (dimensión 1)
        # Pasamos de (Batch, Tiempo, Features) -> (Batch, Features)
        # Lógica para tunear si mean o max?
        x, _ = torch.max(lstm_out, dim=1) 
        # x= torch.mean(lstm_out, dim=1) 
        
        # Clasificador
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class LSTM(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, lstm_hidden=128, lstm_layers=1, bidirectional=True, dense_units=128, dropout_rate=0.5, lr=0.0001, bs=16, ne=30):
        super(LSTM, self).__init__()

        self.model_type = 2
        self.n_channels = n_channels
        # --- NUEVOS PARÁMETROS LSTM ---
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        # ------------------------------
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.bs = bs
        self.ne = ne
                
        self.lstm = nn.LSTM(
            input_size=n_channels, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        lstm_out_features = lstm_hidden * 2 if bidirectional else lstm_hidden # Al ser bidireccional, la salida oculta se multiplica por 2
        
        self.fc1 = nn.Linear(in_features=lstm_out_features, out_features=dense_units)
        self.relu_fc = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=dense_units, out_features=1)
      
    def forward(self, x):
        # x: (Batch, Canales, Tiempo)
        # Reordenar para la LSTM: (Batch, Tiempo, Canales)
        x = x.permute(0, 2, 1) 
        
        lstm_out, _ = self.lstm(x)
        
        # REVISAR PROMEDIO, si hacerlo dense_units poo sentido
        # Agrupar la secuencia entera promediándola sobre el tiempo (dimensión 1)
        # Pasamos de (Batch, Tiempo, Features) -> (Batch, Features)
        # x = torch.mean(lstm_out, dim=1)
        x, _ = torch.max(lstm_out, dim=1)    

        # Clasificador
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResidualBlock1D(nn.Module):
    """
    Bloque Residual 1D. Incluye dos convoluciones, BatchNorm y la conexión skip.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2
        
        # Primera convolución del bloque
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Segunda convolución del bloque
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Conexión Skip (Identity mapping)
        self.shortcut = nn.Sequential()
        # Si cambiamos dimensiones espaciales (stride) o de canales, ajustamos la identidad
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Suma elemento a elemento: F(x) + x
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    Arquitectura ResNet adaptada a series temporales 1D.
    """
    def __init__(self, n_channels: int = 9, initial_filters: int = 32, n_blocks: int = 3, 
                 kernel_size: int = 3, padding: int = 1, # Añadidos para Optuna
                 dense_units: int = 128, dropout_rate: float = 0.5, adaptive_pool: int = 1,
                 lr: float = 0.0001, bs: int = 16, ne: int = 60):
        super(ResNet1D, self).__init__()
        
        # --- CRÍTICO: Guardar hiperparámetros como atributos para save_model() ---
        self.model_type = 3
        self.n_channels = n_channels
        self.initial_filters = initial_filters
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.padding = padding
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.adaptive_pool = adaptive_pool
        self.lr = lr
        self.bs = bs
        self.ne = ne
        
        # Capa de entrada inicial
        self.conv1 = nn.Conv1d(n_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Construcción dinámica de bloques residuales
        layers = []
        in_ch = initial_filters
        for i in range(n_blocks):
            out_ch = in_ch * 2 if i > 0 else in_ch
            stride = 2 if i > 0 else 1
            # Pasamos el kernel_size dinámico al bloque
            layers.append(ResidualBlock1D(in_ch, out_ch, stride=stride, kernel_size=kernel_size))
            in_ch = out_ch
            
        self.res_blocks = nn.Sequential(*layers)
        self.adaptive_pool_layer = nn.AdaptiveAvgPool1d(adaptive_pool)
        
        flatten_size = in_ch * adaptive_pool
        self.fc1 = nn.Linear(flatten_size, dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.res_blocks(x)
        x = self.adaptive_pool_layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class PositionalEncoding1D(nn.Module):
    """
    Inyecta información sobre la posición relativa o absoluta de los tokens en la secuencia.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding1D, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Añadimos dimensiones de Batch para que cuadre con (Batch, Time, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # Buffer para que no sea entrenable y se guarde en state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderLayerWithAttention(nn.Module):
    """
    Capa customizada para poder extraer los pesos de atención matemáticos.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1):
        super(TransformerEncoderLayerWithAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # self_attn devuelve (salida, pesos_atencion)
        src2, attn_weights = self.self_attn(src, src, src, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class TimeSeriesTransformer(nn.Module):
    """
    Clasificador global basado en atención para señales de sueño.
    """
    def __init__(self, n_channels: int = 9, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 3, dim_feedforward: int = 128, dropout_rate: float = 0.3,
                 lr: float = 0.0001, bs: int = 16, ne: int = 60):
        super(TimeSeriesTransformer, self).__init__()
        
        # --- CRÍTICO: Guardar hiperparámetros como atributos para save_model() ---
        self.model_type = 4
        self.n_channels = n_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.bs = bs
        self.ne = ne
        
        self.input_projection = nn.Linear(n_channels, d_model)
        self.pos_encoder = PositionalEncoding1D(d_model)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(d_model, nhead, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        x = self.input_projection(x)
        import math
        x = x * math.sqrt(self.d_model) # escalado por raíz de d_model
        x = self.pos_encoder(x)
        
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            if return_attention:
                attn_maps.append(attn)
        
        x = x.mean(dim=1) 
        out = self.fc_out(x)
        
        if return_attention:
            return out, attn_maps
        return out
        
'''def match_sampling_rate(algorithmic_annotations, algorithmic_fs):
    max_fs = max(algorithmic_fs.values())
    
    for label, fs in algorithmic_fs.items():
        if fs < max_fs:
            repeats = int(np.round(max_fs / fs))
            algorithmic_annotations[label] = np.repeat(algorithmic_annotations[label], repeats)
            algorithmic_fs[label] = max_fs

    return algorithmic_annotations, algorithmic_fs'''

def match_sampling_rate(algorithmic_annotations, algorithmic_fs, target="max"):
    """
    Iguala las frecuencias de muestreo de todos los canales.
    - target="max": Sube la frecuencia de todos los canales a la máxima encontrada (Interpolación por repetición).
    - target="30s": Baja la frecuencia a 1 muestra cada 30 segundos (Media por ventana).
    """
    
    if target == "max":
        max_fs = max(algorithmic_fs.values())
        for label, fs in algorithmic_fs.items():
            if fs < max_fs:
                repeats = int(np.round(max_fs / fs))
                algorithmic_annotations[label] = np.repeat(algorithmic_annotations[label], repeats)
                algorithmic_fs[label] = max_fs
                
    elif target == "30s":
        window_sec = 30
        
        # Primero, calculamos cuántas épocas (ventanas de 30s) va a tener el canal más corto 
        # para asegurar que todos los canales acaben midiendo exactamente lo mismo.
        min_epochs = float('inf')
        for label, fs in algorithmic_fs.items():
            samples_per_window = int(np.round(fs * window_sec))
            num_windows = len(algorithmic_annotations[label]) // samples_per_window
            if num_windows < min_epochs:
                min_epochs = num_windows
                
        # Ahora aplicamos la reducción
        for label, fs in algorithmic_fs.items():
            samples_per_window = int(np.round(fs * window_sec))
            signal = algorithmic_annotations[label]
            
            # Recortamos la señal exactamente a 'min_epochs' para que todos los canales cuadren
            signal_trimmed = signal[:min_epochs * samples_per_window]
            
            if samples_per_window > 1:
                # Magia de Numpy: Convertimos array 1D en matriz 2D de (Epocas, Muestras_por_Epoca)
                # y hacemos la media en el eje 1 (colapsamos las muestras, dejamos las épocas)
                signal_30s = signal_trimmed.reshape(min_epochs, samples_per_window).mean(axis=1)
            else:
                # Si el canal ya era de 1 muestra por época (como las fases del sueño), 
                # no hay que hacer media, solo recortar por si sobraba algo al final.
                signal_30s = signal_trimmed
                
            algorithmic_annotations[label] = signal_30s
            # La nueva frecuencia es de 1 muestra cada 30 segundos (1/30 Hz)
            algorithmic_fs[label] = 1.0 / window_sec 
            
    else:
        raise ValueError("El target debe ser 'max' o '30s'")

    return algorithmic_annotations, algorithmic_fs


# Save your trained model.
def save_model(model_folder, model):
    os.makedirs(model_folder, exist_ok=True)
    filename = os.path.join(model_folder, 'model.pth')
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'hyperparams': {
            'model_type': getattr(model, 'model_type', 0),
            'n_channels': getattr(model, 'n_channels', 9),
            'n_blocks': getattr(model, 'n_blocks', 2),
            'initial_filters': getattr(model, 'initial_filters', 16),
            'kernel_size': getattr(model, 'kernel_size', 32), # Pon 5 si estás ya con las épocas de 30s
            'pool_size': getattr(model, 'pool_size', 3),
            'stride': getattr(model, 'stride', 1),
            'padding': getattr(model, 'padding', 2),
            'adaptive_pool': getattr(model, 'adaptive_pool', 250),
            'lstm_hidden': getattr(model, 'lstm_hidden', 64),
            'lstm_layers': getattr(model, 'lstm_layers', 1),
            'bidirectional': getattr(model, 'bidirectional', True),
            'dense_units': getattr(model, 'dense_units', 256),
            'dropout_rate': getattr(model, 'dropout_rate', 0.3),
            'max_pool': getattr(model, 'max_pool', 0),
            'd_model': getattr(model, 'd_model', 64),
            'nhead': getattr(model, 'nhead', 4),
            'num_layers': getattr(model, 'num_layers', 3),
            'dim_feedforward': getattr(model, 'dim_feedforward', 128),
            'lr': getattr(model, 'lr', 0.0001),
            'bs': getattr(model, 'bs', 32),
            'ne': getattr(model, 'ne', 30)
        }
    }
    torch.save(checkpoint, filename)

def create_stratified_splits(base_data_folder="../data", 
                             output_folder="../cv_splits/split_0",
                             train_ratio=0.7, 
                             val_ratio=0.15, 
                             test_ratio=0.15, 
                             target_prevalence=0.10, 
                             target_col="Cognitive_Impairment",  
                             patient_col="BidsFolder", 
                             site_col="SiteID", 
                             session_col="SessionID",
                             seed=42):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Los ratios deben sumar 1.0"
    
    print(f"Iniciando split (Prevalencia Val/Test: {target_prevalence*100}%)...")
    
    demo_path = os.path.join(base_data_folder, "demographics_total.csv")
    df_raw = pd.read_csv(demo_path)
    
    print("Verificando integridad de los archivos .edf originales...")
    valid_rows = []
    for _, row in df_raw.iterrows():
        patient_id = row[patient_col]
        site_id = row[site_col]
        session_id = row[session_col]
        
        search_pattern = os.path.join(base_data_folder, "algorithmic_annotations", site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
        matching_files = glob.glob(search_pattern)
        
        if matching_files and os.path.getsize(matching_files[0]) > 1024:
            valid_rows.append(row)
            
    df = pd.DataFrame(valid_rows)
    print(f"Pacientes válidos tras el filtro: {len(df)} de {len(df_raw)}")

    df_pos = df[df[target_col] == True].sample(frac=1, random_state=seed).reset_index(drop=True)
    df_neg = df[df[target_col] == False].sample(frac=1, random_state=seed).reset_index(drop=True)
    
    total_samples = len(df)
    total_val = int(total_samples * val_ratio)
    total_test = int(total_samples * test_ratio)
    
    n_pos_val = int(total_val * target_prevalence)
    n_neg_val = total_val - n_pos_val
    
    n_pos_test = int(total_test * target_prevalence)
    n_neg_test = total_test - n_pos_test
    
    if (n_pos_val + n_pos_test) > len(df_pos):
        raise ValueError(f"No hay suficientes casos positivos ({len(df_pos)}) para Val y Test.")
    if (n_neg_val + n_neg_test) > len(df_neg):
        raise ValueError(f"No hay suficientes casos negativos ({len(df_neg)}) para Val y Test.")

    val_df = pd.concat([df_pos.iloc[:n_pos_val], df_neg.iloc[:n_neg_val]])
    test_df = pd.concat([df_pos.iloc[n_pos_val:n_pos_val+n_pos_test], df_neg.iloc[n_neg_val:n_neg_val+n_neg_test]])
    
    train_pos_remaining = df_pos.iloc[n_pos_val+n_pos_test:]
    train_neg_remaining = df_neg.iloc[n_neg_val+n_neg_test:]
    
    train_df = pd.concat([train_pos_remaining, train_neg_remaining])

    train_prevalence = len(train_pos_remaining) / len(train_df) if len(train_df) > 0 else 0
    
    print(f"   -> Train final: {len(train_df)} pacientes (Prevalencia: {train_prevalence*100:.1f}% positivos)")
    print(f"   -> Val final: {len(val_df)} pacientes ({target_prevalence*100}% positivos)")
    print(f"   -> Test final: {len(test_df)} pacientes ({target_prevalence*100}% positivos)")
    
    # Mezclar las filas (shuffle), necesario porque si no [pos, pos, ..., pos, neg, ..., neg]
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    splits = {
        "training_data": train_df,
        "val_data": val_df,
        "test_data": test_df
    }
       
    for split_name, split_df in splits.items():
        dest_folder = os.path.join(output_folder, split_name)
        os.makedirs(dest_folder, exist_ok=True)
        
        print(f"Generando estructura en {dest_folder}...")
        
        split_df.to_csv(os.path.join(dest_folder, f"demographics_{os.path.basename(output_folder)}_{split_name}.csv"), index=False)
        
        for _, row in split_df.iterrows():
            patient_id = row[patient_col]
            site_id = row[site_col]
            session_id = row[session_col]

            search_pattern = os.path.join(base_data_folder, "algorithmic_annotations", site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
            matching_files = glob.glob(search_pattern) #Busca archivos que se llamen así y los guarda en una lista
            
            if matching_files:
                src_file = matching_files[0]
                
                dest_site_folder = os.path.join(dest_folder, "algorithmic_annotations", site_id)
                os.makedirs(dest_site_folder, exist_ok=True)
                
                shutil.copy2(src_file, os.path.join(dest_site_folder, os.path.basename(src_file)))
            else:
                print(f"Aviso: No se encontró el .edf para el paciente {patient_id} en {site_id}")

    print("Split estratificado completado con éxito.")
