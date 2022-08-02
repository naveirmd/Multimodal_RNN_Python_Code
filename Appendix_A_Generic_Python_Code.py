import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import ray
from tensorflow import keras
from keras import callbacks
from keras import backend as BK
from keras.preprocessing import sequence
from keras.layers import Input, Dense, GRU, Masking
from keras.layers.merge import concatenate
from keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator

# Change the following to the directory where the data are stored on your device:
data_directory = 'C:/Users/matthew_naveiras/DocumeNTS_1/Machine_Learning_Data/'

# Functions used for preprocessing:
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
def remove_nonoverlap_ID(data_object, ID_list, overlap_ID_list):
    nonoverlap_indices = []
    for id in range(len(ID_list)):
        if ID_list[id] not in overlap_ID_list:
            nonoverlap_indices.append(id)
    data_object = np.delete(data_object, nonoverlap_indices, 0)
    return data_object

### Loading data into Python ###
grouping_variable = pd.read_csv(data_directory + 'grouping_variable.csv',sep=',') # Distinguishes groups of people
NTS_1 = pd.read_csv(data_directory + 'non_time_series_data_1.csv',sep=',') # 1st non-time-series data modality
NTS_2 = pd.read_csv(data_directory + 'non_time_series_data_1.csv',sep=',') # 2nd non-time-series data modality
TS_1 = pd.read_csv(data_directory + 'time_series_data_1.csv',sep=',') # 1st time-series data modality
TS_2 = pd.read_csv(data_directory + 'time_series_data_2.csv',sep=',') # 2nd time-series data modality
TS_3 = pd.read_csv(data_directory + 'time_series_data_3.csv',sep=',') # 3rd time-series data modality
outcome = pd.read_csv(data_directory + 'outcome_data.csv',sep=',') # Outcome variable (to be predicted)

os.chdir(data_directory)
outcome_ID_training_validation = np.loadtxt('outcome_ID_training_validation.txt', dtype=int)
outcome_ID_Testing = np.loadtxt('outcome_ID_Testing.txt', dtype=int)
training_validation_indices = np.where(np.in1d(outcome_ID, outcome_ID_training_validation))[0]
Testing_indices = np.where(np.in1d(outcome_ID, outcome_ID_Testing))[0]

# Crossvalidating specific models:
#hp_list = pd.read_csv(data_directory + 'Models_to_Crossvalidate.csv', sep=',')
#hp_list = hp_list.to_numpy()
#number_models = len(hp_list)

### Data Preprocessing ###
TS_1 = TS_1.to_numpy()
TS_1 = np.split(TS_1, np.where(np.diff(TS_1[:,0]))[0]+1) # Splits TS_1 into separate objects for each person

TS_2 = TS_2.to_numpy()
TS_2 = np.split(TS_2, np.where(np.diff(TS_2[:,0]))[0]+1)

TS_3 = TS_3.to_numpy()
TS_3 = np.split(TS_3,\ np.where(np.diff(TS_3[:,0]))[0]+1)

process_modalities = [TS_1, TS_2, TS_3]

# Non-process data:
NTS_1 = NTS_1.to_numpy()
NTS_1 = NTS_1[NTS_1[:,0].argsort()]
NTS_1_ID = NTS_1[:,0]
NTS_1 = np.delete(NTS_1, 0, 1)

NTS_2 = NTS_2.to_numpy()
NTS_2 = NTS_2[NTS_2[:,0].argsort()]
NTS_2_ID = NTS_2[:,0]
NTS_2 = np.delete(NTS_2, 0, 1)
        
outcome = outcome.to_numpy()
outcome = outcome[outcome[:,0].argsort()]
outcome_ID = outcome[:,0]
outcome = np.delete(outcome, 0, 1)
max_score = 12 # Maximum score for the outcome variable

all_grouping_IDs = grouping_variable['RID']
grouping_variable_1 = grouping_variable['Variable']
grouping_variable_ID = np.array(all_grouping_IDs[grouping_variable_1 == 'A']) # For example, only use sample of persons with grouping_variable_1 of "A"

# Determines the maximum sequence length across all modalities for padding/masking:
modality_max_seqlen = []
for m in range(len(process_modalities)):
    modality = process_modalities[m]
    modality_seqlen = []
    for i in range(np.shape(modality)[0]):
        modality_seqlen.append(np.shape(modality[i][np.where(~np.isnan(np.sum(modality[i], axis=1)))])[0])
    modality_max_seqlen.append(max(modality_seqlen))
max_seqlen = max(modality_max_seqlen)

# Determines the overlapping IDs for all modalities (persons with data for all files):
NTS_1_ID_set = set(NTS_1_ID)
NTS_2_ID_set = set(NTS_2_ID)
outcome_ID_set = set(outcome_ID)
grouping_variable_ID_set = set(grouping_variable_ID)
overlap_ID = NTS_1_ID_set.intersection(NTS_2_ID_set)
overlap_ID = overlap_ID.intersection(outcome_ID_set)
overlap_ID = overlap_ID.intersection(grouping_variable_ID_set)
modalities_ID = []
for m in range(len(process_modalities)):
    modality = process_modalities[m]
    modality_ID = []
    for i in range(np.shape(modality)[0]):
        modality_ID.append(modality[i][0][0].astype(int))
    modalities_ID.append(modality_ID)
    modality_ID_set = set(modalities_ID[m])
    overlap_ID = overlap_ID.intersection(modality_ID_set)
overlap_ID = list(overlap_ID)

# Only uses persons with data in every file:
NTS_1 = remove_nonoverlap_ID(NTS_1, NTS_1_ID, overlap_ID)
NTS_1_ID = remove_nonoverlap_ID(NTS_1_ID, NTS_1_ID, overlap_ID)
NTS_1_training_validation, NTS_1_testing = NTS_1[training_validation_indices,:], NTS_1[Testing_indices,:]
NTS_2 = remove_nonoverlap_ID(NTS_2, NTS_2_ID, overlap_ID)
NTS_2_ID = remove_nonoverlap_ID(NTS_2_ID, NTS_2_ID, overlap_ID)
NTS_2_training_validation, NTS_2_testing = NTS_2[training_validation_indices,:], NTS_2[Testing_indices,:]
outcome = remove_nonoverlap_ID(outcome, outcome_ID, overlap_ID)
outcome_ID = remove_nonoverlap_ID(outcome_ID, outcome_ID, overlap_ID)
outcome_training_validation, outcome_testing = outcome[training_validation_indices,:], outcome[Testing_indices,:]

# Removes IDs from, transforms, and pads features:
modalities_X = []
for m in range(len(process_modalities)):
    modality = process_modalities[m]
    modality_nonmissing = []
    for i in range(np.shape(modality)[0]):
        modality[i] = np.delete(modality[i], 0, 1)
        modality_nonmissing.append(modality[i][np.where(~np.isnan(np.sum(modality[i], axis=1)))])

    modality_ID = modalities_ID[m]
    modality = remove_nonoverlap_ID(modality_nonmissing, modality_ID, overlap_ID)
    modalities_ID[m] = remove_nonoverlap_ID(modality_ID, modality_ID, overlap_ID)
    
    modality_flat = np.concatenate(modality).ravel()
    modality_min = np.amin(modality_flat)
    modality_max = np.amax(modality_flat)
    modality_sd = np.std(modality_flat)
    modality = modality/(modality_max - modality_min) # Normalization (comment out either this or the next line)
    #modality = modality/modality_sd # Standardization (comment our either this or the previous line)
    modality = sequence.pad_sequences(modality, maxlen = max_seqlen, value = -999) # Sequence padding
    modalities_X.append(modality)

# Names each modality:
TS_2_X = modalities_X[0]
TS_2_ID = modalities_ID[0]
TS_2_X_training_validation, TS_2_X_testing = TS_2_X[training_validation_indices,:,:], TS_2_X[Testing_indices,:,:]

TS_1_X = modalities_X[1]
TS_1_ID = modalities_ID[1]
TS_1_X_training_validation, TS_1_X_testing = TS_1_X[training_validation_indices,:,:], TS_1_X[Testing_indices,:,:]

TS_3_X = modalities_X[2]
TS_3_ID = modalities_ID[2]
TS_3_X_training_validation, TS_3_X_testing =  TS_3_X[training_validation_indices,:,:],  
TS_3_X[Testing_indices,:,:]

def total_score_activation(x): # Activation function for scores ranging from 0 to 12:
    tanh_x = BK.tanh(x) # tanh(x) returns values from (-1,1)
    tanh_x_scaled = (tanh_x + 1)/2 # Transforms from (-1,1) -> (0,1)
    output = max_score*tanh_x_scaled # Transforms from (0,1) -> (0, max_score) = (0, 12)
    return output

def coeff_determination(y_true, y_pred): # Defines function for R^2 metric.
    SS_res =  BK.sum(BK.square( y_true-y_pred )) 
    SS_tot = BK.sum(BK.square( y_true - BK.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + BK.epsilon()) )

### Creating and compiling RNNs ###
def process_model(GRU_sizes, num_hidden_layers, hidden_layer_props):
    TS_2_mask_layer = TS_1_mask_layer = TS_3_mask_layer = Masking(mask_value = -999) # Removes the sequence padding

    NTS_1_input_1 = Input(shape=(6,),name='NTS_1_input') # Phase 1
    NTS_2_input_1 = Input(shape=(1,),name='NTS_2_input')
    TS_1_input_1 = Input(shape=(1698,8), name='TS_1_input')
    TS_2_input_1 = Input(shape=(1698,4), name='TS_2_input')
    TS_3_input_1 = Input(shape=(1698,5), name='TS_3_input')
    
    TS_1_input_1_masked = TS_1_mask_layer(TS_1_input_1)
    TS_2_input_1_masked = TS_2_mask_layer(TS_2_input_1)
    TS_3_input_1_masked = TS_3_mask_layer(TS_3_input_1)
    
    TS_1_input_2 = GRU(GRU_sizes[1], name='TS_1_GRU')(TS_1_input_1_masked) # Phase 2
    TS_2_input_2 = GRU(GRU_sizes[0], name='TS_2_GRU')(TS_2_input_1_masked)
    TS_3_input_2 = GRU(GRU_sizes[2], name='TS_3_GRU')(TS_3_input_1_masked)

    concatenation_layer_1 = concatenate([TS_2_input_2, TS_1_input_2, TS_3_input_2, \
    NTS_1_input_1, NTS_2_input_1], axis=-1) # Phase 3
  
    hidden_layer_size_1 = round(hidden_layer_props[0]*(sum(GRU_sizes) + 7)) # 7 = 1 NTS_2 + 6 reader characteristics
    hidden_layer_1 = Dense(hidden_layer_size_1, name='Hidden_Layer_1')(concatenation_layer_1) # Phase 4
  
    hidden_layer_size_2 = round(hidden_layer_props[1]*hidden_layer_size_1)
    hidden_layer_2 = Dense(hidden_layer_size_2, name='Hidden_Layer_2')(hidden_layer_1)
  
    hidden_layer_size_3 = round(hidden_layer_props[2]*hidden_layer_size_2)
    hidden_layer_3 = Dense(hidden_layer_size_3, name='Hidden_Layer_3')(hidden_layer_2)
    
    # Only uses as many hidden layers as necessary (for example, ignoring hidden_layer_3 if there's only 2 hidden layers):
    if num_hidden_layers == 1:
        hidden_layer_out = hidden_layer_1
    elif num_hidden_layers == 2:
        hidden_layer_out = hidden_layer_2
    else:
        hidden_layer_out = hidden_layer_3
        
    output_layer = Dense(1, activation=total_score_activation, name='output_layer')(hidden_layer_out) # Phase 5
    model = Model(inputs=[TS_2_input_1, TS_1_input_1, TS_3_input_1, NTS_1_input_1, NTS_2_input_1], \ 
    outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', coeff_determination])
    return model

### Tuning hyperparameters ###
hyperparameter_space_1 = {
"GRU_size_1": tune.choice([8,16,32,48,64,80,96,112,128]),
    "GRU_size_2": tune.choice([8,16,32,48,64,80,96,112,128]),
    "GRU_size_3": tune.choice([8,16,32,48,64,80,96,112,128]), 
    "hidden_layer_1_proportion":  tune.uniform(0.1, 0.9),
    "batch_size": tune.choice(list(range(1,24)))
}

hyperparameter_space_2 = {
    "GRU_size_1": tune.choice([8,16,32,48,64,80,96,112,128]),
    "GRU_size_2": tune.choice([8,16,32,48,64,80,96,112,128]),
    "GRU_size_3": tune.choice([8,16,32,48,64,80,96,112,128]), 
    "hidden_layer_1_proportion":  tune.uniform(0.1, 0.9),
    "hidden_layer_2_proportion":  tune.uniform(0.1, 0.9),
    "batch_size": tune.choice(list(range(1,24)))
}

hyperparameter_space_3 = {
    "GRU_size_1": tune.choice([8,16,32,48,64,80,96,112,128]),
    "GRU_size_2": tune.choice([8,16,32,48,64,80,96,112,128]),
    "GRU_size_3": tune.choice([8,16,32,48,64,80,96,112,128]), 
    "hidden_layer_1_proportion":  tune.uniform(0.1, 0.9),
    "hidden_layer_2_proportion":  tune.uniform(0.1, 0.9),
    "hidden_layer_3_proportion":  tune.uniform(0.1, 0.9),
    "batch_size": tune.choice(list(range(1,24)))
}

# Callbacks used by Tune (sending necessary information to Tune during training):
class TuneReporterCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()
    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(keras_info=logs, loss=logs.get("loss"))

def tune_process_model_holdout(con):
    num_hidden_layers = con['number_hidden_layers']
    if num_hidden_layers == 1:
        hidden_layer_props = [con["hidden_layer_1_proportion"],1,1]
    elif num_hidden_layers == 2:
        hidden_layer_props = [con["hidden_layer_1_proportion"],con["hidden_layer_2_proportion"],1]
    else:
        hidden_layer_props = [con["hidden_layer_1_proportion"],con["hidden_layer_2_proportion"],con["hidden_layer_3_proportion"]]

    index_list = list(range(144))
    random.shuffle(index_list)
    folds_indices_list = list(split(index_list, number_folds))
    ind_train = folds_indices_list[1:(number_folds+1)]
    ind_train = [item for sublist in ind_train for item in sublist]
    ind_val = folds_indices_list[0]
    ind_list = [ind_train, ind_val]

    NTS_1_train, NTS_1_val = \ 
    NTS_1_training_validation[ind_list[0]], NTS_1_training_validation[ind_list[1]]
    NTS_2_train, NTS_2_val = NTS_2_training_validation[ind_list[0]], NTS_2_training_validation[ind_list[1]]
    outcome_train, outcome_val = outcome_training_validation[ind_list[0]], outcome_training_validation[ind_list[1]]

    TS_1_X_train, TS_1_X_val = TS_1_X_training_validation[ind_list[0]], TS_1_X_training_validation[ind_list[1]]
    TS_2_X_train, TS_2_X_val = TS_2_X_training_validation[ind_list[0]], TS_2_X_training_validation[ind_list[1]]
    TS_3_X_train, TS_3_X_val = \ 
    TS_3_X_training_validation[ind_list[0]], TS_3_X_training_validation[ind_list[1]]
    
    model = process_model(GRU_sizes = [con["GRU_size_1"],con["GRU_size_2"],con["GRU_size_3"]],
                          num_hidden_layers = num_hidden_layers,
                          hidden_layer_props = hidden_layer_props)
    
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 100, restore_best_weights = True)
    checkpoint_callback = ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, save_freq='epoch')
    model_callbacks = [earlystopping, checkpoint_callback, TuneReporterCallback()]
    model.fit([TS_2_X_train, TS_1_X_train, TS_3_X_train, NTS_1_train, NTS_2_train], outcome_train, \
    epochs=number_epochs, batch_size=con['batch_size'], validation_data = ([TS_2_X_val, TS_1_X_val, TS_3_X_val, \ 
    NTS_1_val, NTS_2_val], outcome_val), verbose=1, shuffle=True, callbacks = model_callbacks) # Changed from verbose=2.  

def tune_process_model_crossvalidation(con):
    model_number = con["model_number"]
    fold_number = con["fold_number"]
    hp_model = hp_list[model_number]
    
    if number_HL==1:
        hp = {
            "GRU_size_1": int(hp_model[0]),
            "GRU_size_2": int(hp_model[1]),
            "GRU_size_3": int(hp_model[2]), 
            "hidden_layer_1_proportion":  hp_model[4],
            "batch_size": int(hp_model[3]),
            "patience": int(hp_model[5]),
            "seed_value": int(hp_model[6]),
            "number_hidden_layers": number_HL
        }
    elif number_HL==2:
        hp = {
            "GRU_size_1": int(hp_model[0]),
            "GRU_size_2": int(hp_model[1]),
            "GRU_size_3": int(hp_model[2]), 
            "hidden_layer_1_proportion":  hp_model[4],
            "hidden_layer_2_proportion":  hp_model[5],
            "batch_size": int(hp_model[3]),
            "patience": int(hp_model[6]),
            "seed_value": int(hp_model[7]),
            "number_hidden_layers": number_HL
        }
    else:
         hp = {
            "GRU_size_1": int(hp_model[0]),
            "GRU_size_2": int(hp_model[1]),
            "GRU_size_3": int(hp_model[2]), 
            "hidden_layer_1_proportion":  hp_model[4],
            "hidden_layer_2_proportion":  hp_model[5],
            "hidden_layer_3_proportion":  hp_model[6],
            "batch_size": int(hp_model[3]),
            "patience": int(hp_model[7]),
            "seed_value": int(hp_model[8]),
            "number_hidden_layers": number_HL
        }
         
    num_hidden_layers = hp['number_hidden_layers']
    if num_hidden_layers == 1:
        hidden_layer_props = [hp["hidden_layer_1_proportion"],1,1]
    elif num_hidden_layers == 2:
        hidden_layer_props = [hp["hidden_layer_1_proportion"],hp["hidden_layer_2_proportion"],1]
    else:
        hidden_layer_props = [hp["hidden_layer_1_proportion"],hp["hidden_layer_2_proportion"],hp["hidden_layer_3_proportion"]]
         
    n = 144
    index_list = list(range(n))
    random.seed(hp['seed_value'])
    random.shuffle(index_list)
    folds_indices_list = list(split(index_list, number_folds))
    ind_train = folds_indices_list[:fold_number] + folds_indices_list[fold_number+1:]
    ind_train = [item for sublist in ind_train for item in sublist]
    ind_val = folds_indices_list[fold_number]
    ind_list = [ind_train, ind_val]

    NTS_1_train, NTS_1_val = \ 
    NTS_1_training_validation[ind_list[0]], NTS_1_training_validation[ind_list[1]]
    NTS_2_train, NTS_2_val = NTS_2_training_validation[ind_list[0]], NTS_2_training_validation[ind_list[1]]
    outcome_train, outcome_val = outcome_training_validation[ind_list[0]], outcome_training_validation[ind_list[1]]

    TS_1_X_train, TS_1_X_val = TS_1_X_training_validation[ind_list[0]], TS_1_X_training_validation[ind_list[1]]
    TS_2_X_train, TS_2_X_val = TS_2_X_training_validation[ind_list[0]], TS_2_X_training_validation[ind_list[1]]
    TS_3_X_train, TS_3_X_val = \ 
    TS_3_X_training_validation[ind_list[0]], TS_3_X_training_validation[ind_list[1]]
        
    seed_value = hp['seed_value']
    model = process_model(GRU_sizes = [hp["GRU_size_1"],hp["GRU_size_2"],hp["GRU_size_3"]],
                          num_hidden_layers = num_hidden_layers,
                          hidden_layer_props = hidden_layer_props,
                          seed_value = seed_value)
    
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = hp["patience"], restore_best_weights = True)
    checkpoint_callback = ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, save_freq='epoch')
    model_callbacks = [earlystopping, checkpoint_callback, TuneReporterCallback()]  
    model.fit([TS_2_X_train, TS_1_X_train, TS_3_X_train, NTS_1_train, NTS_2_train], outcome_train, \ 
    epochs=number_epochs, batch_size=hp['batch_size'], validation_data = ([TS_2_X_val, TS_1_X_val, \ 
    TS_3_X_val, NTS_1_val, NTS_2_val], outcome_val), verbose=1, shuffle=True, callbacks = model_callbacks) 

ray.shutdown()
ray.init(log_to_driver=False)
analysis = tune.run(
    tune_process_model_holdout, # Hold-out validation (hyperparameter search)
    #tune_process_model_crossvalidation, # Cross-validation (validating models)
    metric="keras_info/val_mean_squared_error",
    mode="min",
    num_samples=200, # Hold-out validation
    #num_samples=1, # Cross-validation
    verbose=3, 
    search_alg=BasicVariantGenerator(),
    config=hyperparameter_space_1, # Hold-out validation
    #config={"model_number": tune.grid_search(list(range(number_models))),"fold_number": tune.grid_search(list(range(5)))}, # Cross-validation
    name="Machine_Learning_Models")