import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
from zipfile import ZipFile
import random
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_Attention

class EngineDataModelTraining:
    def __init__(self, config_path, file_path):
        self.file_path = file_path
        self.config_path = config_path
        self.model = None
        self.current_epoch = 0
        self.total_epochs = 75

        self.delimiter = ' '
        self.columns = [
            'unit_number',
            'time_cycles',
            'operational_setting_1',
            'operational_setting_2',
            'operational_setting_3'
        ] + [f'sensor_measurement_{i}' for i in range(1, 27)]  # Adjusted to 21 sensor measurements

    def moving_average_filter(self, data, window_size):
        return data.rolling(window=window_size).mean().fillna(method='bfill')
    
    def _split_data(self, data_list, target_list, split_ratio=0.8):
        num_total = len(data_list)
        num_train = int(num_total * split_ratio)
        
        train_indices = random.sample(range(num_total), num_train)
        x_train = [data_list[i] for i in train_indices]
        y_train = [target_list[i] for i in train_indices]
        
        x_val = [data_list[i] for i in range(num_total) if i not in train_indices]
        y_val = [target_list[i] for i in range(num_total) if i not in train_indices]
        
        return x_train, y_train, x_val, y_val

    def load_and_process_data(self):
        df_list = []
        df = pd.read_csv(self.file_path, delimiter=self.delimiter, header=None, names=self.columns)
        df.dropna(axis=1, inplace=True)
        df.set_index('time_cycles', inplace=True)
        df.reset_index(inplace=True)

        Rearly = 125
        df_max_cycle = df.groupby('unit_number')['time_cycles'].max().reset_index()
        df_max_cycle = df_max_cycle.rename(columns={'time_cycles': 'max_cycles'})
        df = df.merge(df_max_cycle, on='unit_number', how='left')
        df['RUL'] = df['max_cycles'] - df['time_cycles']
        condition = df['time_cycles'] >= (df['max_cycles'] - Rearly)
        df.loc[condition, 'RUL_label'] = df['max_cycles'] - df['time_cycles']
        df.loc[~condition, 'RUL_label'] = Rearly
        df.drop(['max_cycles'], axis=1, inplace=True) 

        sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
        for sensor in sensor_columns:
            df[sensor] = self.moving_average_filter(df[sensor], window_size = 5)

        for col in sensor_columns:
            delta_col = f"delta_{col}"
            df[delta_col] = df.groupby('unit_number')[col].diff().fillna(0)

        op_setting_columns = [col for col in df.columns if col.startswith('operational_setting')]
        all_columns = sensor_columns + op_setting_columns
        for col in all_columns:
            max_vals = df.groupby('unit_number')[col].max()
            min_vals = df.groupby('unit_number')[col].min()
            df[col] = df.groupby('unit_number', group_keys=True)[col].apply(lambda x: (x - min_vals[x.name]) / (max_vals[x.name] - min_vals[x.name])).reset_index(drop=True)
        
        df.set_index('time_cycles', inplace=True) 

        for unit_number in df['unit_number'].unique():
            df_unit = df[df['unit_number'] == unit_number]
            if 'time_cycles' in df_unit.columns:
                        df_unit.set_index('time_cycles', inplace=True)
            df.fillna(0, inplace=True)
            df_list.append(df_unit)

        split_ratio = 0.8
        input_seq_len = 15
        output_seq_len = 1

        features_input_encoder = ['operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21','delta_sensor_measurement_2','delta_sensor_measurement_3','delta_sensor_measurement_4','delta_sensor_measurement_7','delta_sensor_measurement_8','delta_sensor_measurement_9','delta_sensor_measurement_11','delta_sensor_measurement_12','delta_sensor_measurement_13','delta_sensor_measurement_14','delta_sensor_measurement_15','delta_sensor_measurement_17','delta_sensor_measurement_20','delta_sensor_measurement_21']
        features_input_decoder = ['operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21','delta_sensor_measurement_2','delta_sensor_measurement_3','delta_sensor_measurement_4','delta_sensor_measurement_7','delta_sensor_measurement_8','delta_sensor_measurement_9','delta_sensor_measurement_11','delta_sensor_measurement_12','delta_sensor_measurement_13','delta_sensor_measurement_14','delta_sensor_measurement_15','delta_sensor_measurement_17','delta_sensor_measurement_20','delta_sensor_measurement_21']
        feature_target = ['RUL_label']
        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []
        X_list = []
        y_list = []

        for i in range(len(df_list)):
            preprocessor = Preprocessor(df_list[i])
            preprocessor.process(
                features_input_encoder,
                features_input_decoder,
                feature_target,
                input_seq_len,
                output_seq_len,
                model_type="tcn_tcn",
                time_col=df_list[i].index,
                split_ratio=split_ratio,
                split_date=None,
                temporal_encoding_modes=None,
                autoregressive=False
            )
            X, y = preprocessor.train_split()
            X_list.append(X)
            y_list.append(y)

        
        X_train_list, y_train_list, X_val_list, y_val_list = self._split_data(X_list, y_list, split_ratio)
        encoder_super_sequence_train = tf.convert_to_tensor(np.concatenate([elem[0] for elem in X_train_list], axis=0), dtype=tf.float64)
        decoder_super_sequence_train = tf.convert_to_tensor(np.concatenate([elem[1] for elem in X_train_list], axis=0), dtype=tf.float64)
        self.x_train = [encoder_super_sequence_train, decoder_super_sequence_train]
        encoder_super_sequence_val =  tf.convert_to_tensor(np.concatenate([elem[0] for elem in X_val_list], axis=0), dtype=tf.float64)
        decoder_super_sequence_val =  tf.convert_to_tensor(np.concatenate([elem[1] for elem in X_val_list], axis=0), dtype=tf.float64)
        self.x_val = [encoder_super_sequence_val, decoder_super_sequence_val]
        self.y_train = tf.convert_to_tensor(np.concatenate(y_train_list, axis=0),  dtype=tf.float64)
        self.y_val = tf.convert_to_tensor(np.concatenate(y_val_list, axis=0),  dtype=tf.float64)
        
    def training(self):
        self.model = TCN_TCN_Attention()
        self.model.build(
            num_layers_tcn = None,
            num_filters = 16,
            kernel_size = 8,
            dilation_base = 2,
            dropout_rate = 0.2,
            key_size = 6,
            value_size = 6,
            num_attention_heads = 2,
            neurons_output = [480],
            activation = "relu",
            kernel_initializer = "he_normal",
            batch_norm_tcn = False,
            layer_norm_tcn = True,
            autoregressive=False,
            padding_encoder='causal',
            padding_decoder='causal',
            current_epoch = self.current_epoch,
            total_epochs = self.total_epochs,)
        
        self.model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.02,decay=1e-3) , run_eagerly = True)
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(self.x_train,
                        self.y_train,
                        (self.x_val, self.y_val),
                        epochs=self.total_epochs,
                        batch_size=64,
                        callbacks = [cb_early_stopping]
                        )
        self.model.save_model(self.config_path)

    def parameter_check(self):
        self.model = TCN_TCN_Attention()
        self.model.parameter_search(self.x_train,
                                    self.y_train,
                                    self.x_val,
                                    self.y_val,
                                    batch_size = 24,
                                    results_path = "./search_final_RUL",
                                    patience=3,
                                    max_trials = 150,
                                    executions_per_trial = 1,
                                    num_filters = [12, 16],
                                    neurons_output = [16],
                                    kernel_size = [4,6,8],
                                    dilation_base = [2],
                                    dropout_rate = [0.3],
                                    key_value_size = [6, 8],
                                    num_attention_heads = [1, 2],
                                    activation = ["relu"],
                                    kernel_initializer = ["he_normal"],
                                    batch_norm_tcn = [False],
                                    layer_norm_tcn = [True],
                                    padding_encoder = ['same', 'causal'],
                                    padding_decoder = ['causal']
                                )
        

config_path = "C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\EngineHealthPred\\ModelWeights\\RULFD001"
input_file = 'C:\\Users\\ssen\\Documents\\FlexKI\\EngineHealthPred\\ChallengeData\\Challenge_Data\\CMAPSSData\\train_FD001.txt'

trainer = EngineDataModelTraining(config_path, input_file)
trainer.load_and_process_data()
trainer.training()