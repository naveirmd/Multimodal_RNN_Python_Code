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
data_directory = 'C:/Users/matthew_naveiras/Documents/Machine_Learning_Data/'

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
condition_variable = pd.read_csv(data_directory + 'Condition_Variable.csv',sep=',')
eye_tracking = pd.read_csv(data_directory + 'eye_tracking.csv',sep=',')
emotion = pd.read_csv(data_directory + 'emotion.csv',sep=',')
digital_behaviors_process = pd.read_csv(data_directory + 'digital_behaviors_process.csv',sep=',')
reader_characteristics = pd.read_csv(data_directory + 'reader_characteristics.csv',sep=',')
pretest = pd.read_csv(data_directory + 'Pretest.csv',sep=',')
posttest = pd.read_csv(data_directory + 'Posttest.csv',sep=',')

os.chdir(data_directory)
posttest_ID_training_validation = np.loadtxt('posttest_ID_training_validation.txt', dtype=int)
posttest_ID_testing = np.loadtxt('posttest_ID_testing.txt', dtype=int)
training_validation_indices = np.where(np.in1d(posttest_ID, posttest_ID_training_validation))[0]
testing_indices = np.where(np.in1d(posttest_ID, posttest_ID_testing))[0]

# Crossvalidating specific models:
#hp_list = pd.read_csv(data_directory + 'Models_to_Crossvalidate.csv', sep=',')
#hp_list = hp_list.to_numpy()
#number_models = len(hp_list)

### Data Preprocessing ###
eye_tracking = eye_tracking.to_numpy()
eye_tracking = np.split(eye_tracking, np.where(np.diff(eye_tracking[:,0]))[0]+1) # Splits eye_tracking into separate objects for each person

emotion = emotion.to_numpy()
emotion = np.split(emotion, np.where(np.diff(emotion[:,0]))[0]+1)

digital_behaviors_process = digital_behaviors_process.to_numpy()
digital_behaviors_process = np.split(digital_behaviors_process,\ np.where(np.diff(digital_behaviors_process[:,0]))[0]+1)

process_modalities = [eye_tracking, emotion, digital_behaviors_process]

# Non-process data:
reader_characteristics = reader_characteristics.to_numpy()
reader_characteristics = reader_characteristics[reader_characteristics[:,0].argsort()]
reader_characteristics_ID = reader_characteristics[:,0]
reader_characteristics = np.delete(reader_characteristics, 0, 1)

pretest = pretest.to_numpy()
pretest = pretest[pretest[:,0].argsort()]
pretest_ID = pretest[:,0]
pretest = np.delete(pretest, 0, 1)
        
posttest = posttest.to_numpy()
posttest = posttest[posttest[:,0].argsort()]
posttest_ID = posttest[:,0]
posttest = np.delete(posttest, 0, 1)
max_score = 12 # Maximum score for the outcome variable

all_condition_IDs = condition_variable['RID']
condition_AB = condition_variable['Condition']
condition_ID = np.array(all_condition_IDs[condition_AB == 'A']) # Only using condition A

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
reader_characteristics_ID_set = set(reader_characteristics_ID)
pretest_ID_set = set(pretest_ID)
posttest_ID_set = set(posttest_ID)
condition_ID_set = set(condition_ID)
overlap_ID = reader_characteristics_ID_set.intersection(pretest_ID_set)
overlap_ID = overlap_ID.intersection(posttest_ID_set)
overlap_ID = overlap_ID.intersection(condition_ID_set)
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
reader_characteristics = remove_nonoverlap_ID(reader_characteristics, reader_characteristics_ID, overlap_ID)
reader_characteristics_ID = remove_nonoverlap_ID(reader_characteristics_ID, reader_characteristics_ID, overlap_ID)
reader_characteristics_training_validation, reader_characteristics_testing = reader_characteristics[training_validation_indices,:], reader_characteristics[testing_indices,:]
pretest = remove_nonoverlap_ID(pretest, pretest_ID, overlap_ID)
pretest_ID = remove_nonoverlap_ID(pretest_ID, pretest_ID, overlap_ID)
pretest_training_validation, pretest_testing = pretest[training_validation_indices,:], pretest[testing_indices,:]
posttest = remove_nonoverlap_ID(posttest, posttest_ID, overlap_ID)
posttest_ID = remove_nonoverlap_ID(posttest_ID, posttest_ID, overlap_ID)
posttest_training_validation, posttest_testing = posttest[training_validation_indices,:], posttest[testing_indices,:]

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
emotion_X = modalities_X[0]
emotion_ID = modalities_ID[0]
emotion_X_training_validation, emotion_X_testing = emotion_X[training_validation_indices,:,:], emotion_X[testing_indices,:,:]

eye_tracking_X = modalities_X[1]
eye_tracking_ID = modalities_ID[1]
eye_tracking_X_training_validation, eye_tracking_X_testing = eye_tracking_X[training_validation_indices,:,:], eye_tracking_X[testing_indices,:,:]

digital_behaviors_process_X = modalities_X[2]
digital_behaviors_process_ID = modalities_ID[2]
digital_behaviors_process_X_training_validation, digital_behaviors_process_X_testing =  digital_behaviors_process_X[training_validation_indices,:,:],  
digital_behaviors_process_X[testing_indices,:,:]

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
    emotion_mask_layer = eye_tracking_mask_layer = digital_behaviors_process_mask_layer = Masking(mask_value = -999) # Removes the sequence padding

    reader_characteristics_input_1 = Input(shape=(6,),name='reader_characteristics_input') # Phase 1
    pretest_input_1 = Input(shape=(1,),name='pretest_input')
    eye_tracking_input_1 = Input(shape=(1698,8), name='eye_tracking_input')
    emotion_input_1 = Input(shape=(1698,4), name='emotion_input')
    digital_behaviors_process_input_1 = Input(shape=(1698,5), name='digital_behaviors_process_input')
    
    eye_tracking_input_1_masked = eye_tracking_mask_layer(eye_tracking_input_1)
    emotion_input_1_masked = emotion_mask_layer(emotion_input_1)
    digital_behaviors_process_input_1_masked = digital_behaviors_process_mask_layer(digital_behaviors_process_input_1)
    
    eye_tracking_input_2 = GRU(GRU_sizes[1], name='eye_tracking_GRU')(eye_tracking_input_1_masked) # Phase 2
    emotion_input_2 = GRU(GRU_sizes[0], name='emotion_GRU')(emotion_input_1_masked)
    digital_behaviors_process_input_2 = GRU(GRU_sizes[2], name='digital_behaviors_process_GRU')(digital_behaviors_process_input_1_masked)

    concatenation_layer_1 = concatenate([emotion_input_2, eye_tracking_input_2, digital_behaviors_process_input_2, \
    reader_characteristics_input_1, pretest_input_1], axis=-1) # Phase 3
  
    hidden_layer_size_1 = round(hidden_layer_props[0]*(sum(GRU_sizes) + 7)) # 7 = 1 pretest + 6 reader characteristics
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
    model = Model(inputs=[emotion_input_1, eye_tracking_input_1, digital_behaviors_process_input_1, reader_characteristics_input_1, pretest_input_1], \ 
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

    reader_characteristics_train, reader_characteristics_val = \ 
    reader_characteristics_training_validation[ind_list[0]], reader_characteristics_training_validation[ind_list[1]]
    pretest_train, pretest_val = pretest_training_validation[ind_list[0]], pretest_training_validation[ind_list[1]]
    posttest_train, posttest_val = posttest_training_validation[ind_list[0]], posttest_training_validation[ind_list[1]]

    eye_tracking_X_train, eye_tracking_X_val = eye_tracking_X_training_validation[ind_list[0]], eye_tracking_X_training_validation[ind_list[1]]
    emotion_X_train, emotion_X_val = emotion_X_training_validation[ind_list[0]], emotion_X_training_validation[ind_list[1]]
    digital_behaviors_process_X_train, digital_behaviors_process_X_val = \ 
    digital_behaviors_process_X_training_validation[ind_list[0]], digital_behaviors_process_X_training_validation[ind_list[1]]
    
    model = process_model(GRU_sizes = [con["GRU_size_1"],con["GRU_size_2"],con["GRU_size_3"]],
                          num_hidden_layers = num_hidden_layers,
                          hidden_layer_props = hidden_layer_props)
    
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 100, restore_best_weights = True)
    checkpoint_callback = ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, save_freq='epoch')
    model_callbacks = [earlystopping, checkpoint_callback, TuneReporterCallback()]
    model.fit([emotion_X_train, eye_tracking_X_train, digital_behaviors_process_X_train, reader_characteristics_train, pretest_train], posttest_train, \
    epochs=number_epochs, batch_size=con['batch_size'], validation_data = ([emotion_X_val, eye_tracking_X_val, digital_behaviors_process_X_val, \ 
    reader_characteristics_val, pretest_val], posttest_val), verbose=1, shuffle=True, callbacks = model_callbacks) # Changed from verbose=2.  

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

    reader_characteristics_train, reader_characteristics_val = \ 
    reader_characteristics_training_validation[ind_list[0]], reader_characteristics_training_validation[ind_list[1]]
    pretest_train, pretest_val = pretest_training_validation[ind_list[0]], pretest_training_validation[ind_list[1]]
    posttest_train, posttest_val = posttest_training_validation[ind_list[0]], posttest_training_validation[ind_list[1]]

    eye_tracking_X_train, eye_tracking_X_val = eye_tracking_X_training_validation[ind_list[0]], eye_tracking_X_training_validation[ind_list[1]]
    emotion_X_train, emotion_X_val = emotion_X_training_validation[ind_list[0]], emotion_X_training_validation[ind_list[1]]
    digital_behaviors_process_X_train, digital_behaviors_process_X_val = \ 
    digital_behaviors_process_X_training_validation[ind_list[0]], digital_behaviors_process_X_training_validation[ind_list[1]]
        
    seed_value = hp['seed_value']
    model = process_model(GRU_sizes = [hp["GRU_size_1"],hp["GRU_size_2"],hp["GRU_size_3"]],
                          num_hidden_layers = num_hidden_layers,
                          hidden_layer_props = hidden_layer_props,
                          seed_value = seed_value)
    
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = hp["patience"], restore_best_weights = True)
    checkpoint_callback = ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=True, save_freq='epoch')
    model_callbacks = [earlystopping, checkpoint_callback, TuneReporterCallback()]  
    model.fit([emotion_X_train, eye_tracking_X_train, digital_behaviors_process_X_train, reader_characteristics_train, pretest_train], posttest_train, \ 
    epochs=number_epochs, batch_size=hp['batch_size'], validation_data = ([emotion_X_val, eye_tracking_X_val, \ 
    digital_behaviors_process_X_val, reader_characteristics_val, pretest_val], posttest_val), verbose=1, shuffle=True, callbacks = model_callbacks) 

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