import zipfile
import io
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_Attention
import random

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

class CustomCallback(Callback):
    def __init__(self, model, current_epoch, total_epochs):
        super(CustomCallback, self).__init__()
        self.model = model
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        print(f"Current Epoch: {self.current_epoch}")

    def on_epoch_end(self, epoch, logs=None):
        pass

class ModelTrainer:
    def __init__(self, config_path, input_file, test_file):
        self.config_path = config_path
        self.input_file = input_file
        self.test_file = test_file
        self.processor = None
        self.preprocess = None
        self.model = None
        self.current_epoch = 0
        self.total_epochs = 75

    # def load_data(self, input_file):
    #     with zipfile.ZipFile(input_file, 'r') as zip_ref:
    #         file_list = zip_ref.namelist()
    #         self.processed_df = pd.DataFrame()
    #         for file_name in file_list:
    #             with zip_ref.open(file_name) as file:
    #                 data = pd.read_csv(io.TextIOWrapper(file))
    #                 self.processed_df = self.processed_df.append(data)

    def combine_data(self):
        with zipfile.ZipFile(self.input_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            self.processor_list = []
            self.preprocess_list = []
            self.df_list = []
            for file_name in file_list:
                with zip_ref.open(file_name) as file:
                    df = pd.read_csv(io.TextIOWrapper(file))
                    if 'time_cycles' in df.columns:
                        df.set_index('time_cycles', inplace=True)
                    df.fillna(0, inplace=True)
                    self.df_list.append(df)

    def combine_data_test(self):
        with zipfile.ZipFile(self.test_file, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            # self.processor_list = []
            # self.preprocess_list = []
            self.df_list_test = []
            for file_name in file_list:
                with zip_ref.open(file_name) as file:
                    df = pd.read_csv(io.TextIOWrapper(file))
                    if 'time_cycles' in df.columns:
                        df.set_index('time_cycles', inplace=True)
                    df.fillna(0, inplace=True)
                    self.df_list_test.append(df)
    
    # def _split_data(self, data_list, split_ratio=0.8):
    #     num_total = len(data_list)
    #     num_train = int(num_total * split_ratio)
    #     return data_list[:num_train], data_list[num_train:]
                    
    def _split_data(self, data_list, target_list, split_ratio=0.8):
        num_total = len(data_list)
        num_train = int(num_total * split_ratio)
        
        train_indices = random.sample(range(num_total), num_train)
        x_train = [data_list[i] for i in train_indices]
        y_train = [target_list[i] for i in train_indices]
        
        x_val = [data_list[i] for i in range(num_total) if i not in train_indices]
        y_val = [target_list[i] for i in range(num_total) if i not in train_indices]
        
        return x_train, y_train, x_val, y_val


    def test_train_dataset(self):
        split_ratio = 0.8
        input_seq_len = 15
        output_seq_len = 1
       
        features_input_encoder = ['operational_setting_1','operational_setting_2','operational_setting_3', 'sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21']  #,'delta_sensor_measurement_2','delta_sensor_measurement_3','delta_sensor_measurement_4','delta_sensor_measurement_7','delta_sensor_measurement_8','delta_sensor_measurement_9','delta_sensor_measurement_11','delta_sensor_measurement_12','delta_sensor_measurement_13','delta_sensor_measurement_14','delta_sensor_measurement_15','delta_sensor_measurement_17','delta_sensor_measurement_20','delta_sensor_measurement_21']
        
        features_input_decoder = ['operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21']  #,'delta_sensor_measurement_2','delta_sensor_measurement_3','delta_sensor_measurement_4','delta_sensor_measurement_7','delta_sensor_measurement_8','delta_sensor_measurement_9','delta_sensor_measurement_11','delta_sensor_measurement_12','delta_sensor_measurement_13','delta_sensor_measurement_14','delta_sensor_measurement_15','delta_sensor_measurement_17','delta_sensor_measurement_20','delta_sensor_measurement_21']
        
        # features_input_encoder = ['sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21']       
        
        # features_input_encoder = ['operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_1','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_5','sensor_measurement_6','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_10','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_16','sensor_measurement_17','sensor_measurement_18','sensor_measurement_19','sensor_measurement_20','sensor_measurement_21']       #'delta_sensor_measurement_1','delta_sensor_measurement_2','delta_sensor_measurement_3','delta_sensor_measurement_4','delta_sensor_measurement_5','delta_sensor_measurement_6','delta_sensor_measurement_7','delta_sensor_measurement_8','delta_sensor_measurement_9','delta_sensor_measurement_11','delta_sensor_measurement_12','delta_sensor_measurement_13','delta_sensor_measurement_14','delta_sensor_measurement_15','delta_sensor_measurement_17','delta_sensor_measurement_18','delta_sensor_measurement_19','delta_sensor_measurement_20','delta_sensor_measurement_21']
        
        # features_input_decoder = ['sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21']
        
        # features_input_decoder = ['operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_1','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_5','sensor_measurement_6','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_10','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_16','sensor_measurement_17','sensor_measurement_18','sensor_measurement_19','sensor_measurement_20','sensor_measurement_21']    #'delta_sensor_measurement_1','delta_sensor_measurement_2','delta_sensor_measurement_3','delta_sensor_measurement_4','delta_sensor_measurement_5','delta_sensor_measurement_6','delta_sensor_measurement_7','delta_sensor_measurement_8','delta_sensor_measurement_9','delta_sensor_measurement_11','delta_sensor_measurement_12','delta_sensor_measurement_13','delta_sensor_measurement_14','delta_sensor_measurement_15','delta_sensor_measurement_17','delta_sensor_measurement_18','delta_sensor_measurement_19','delta_sensor_measurement_20','delta_sensor_measurement_21','operational_setting_1_rolling_mean', 'operational_setting_1_rolling_variance', 'operational_setting_2_rolling_mean', 'operational_setting_2_rolling_variance', 'operational_setting_3_rolling_mean', 'operational_setting_3_rolling_variance']
       

        feature_target = ['RUL_label']
        feature_target_test = ['RUL']

        self.preprocessor_loaded_list = []
        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []
        # X_list = []
        # y_list = []

        for i in range(len(self.df_list)):
            preprocessor = Preprocessor(self.df_list[i])
            # print(self.df_list[i].index)
            preprocessor.process(
                features_input_encoder,
                features_input_decoder,
                feature_target,
                input_seq_len,
                output_seq_len,
                model_type="tcn_tcn",
                time_col=self.df_list[i].index,
                split_ratio=split_ratio,
                split_date=None,
                temporal_encoding_modes=None,
                autoregressive=False
            )
            self.preprocessor_loaded_list.append(preprocessor)
            X, y = preprocessor.train_split()
            X_train_list.append(X)
            y_train_list.append(y)

        for i in range(len(self.df_list_test)):
            preprocessor = Preprocessor(self.df_list_test[i])
            # print(self.df_list[i].index)
            preprocessor.process(
                features_input_encoder,
                features_input_decoder,
                feature_target_test,
                input_seq_len,
                output_seq_len,
                model_type="tcn_tcn",
                time_col=self.df_list_test[i].index,
                split_ratio=split_ratio,
                split_date=None,
                temporal_encoding_modes=None,
                autoregressive=False
            )
            self.preprocessor_loaded_list.append(preprocessor)
            X, y = preprocessor.train_split()
            X_val_list.append(X)
            y_val_list.append(y)
        
        # X_train_list, X_val_list = self._split_data(X_list, split_ratio)
        # y_train_list, y_val_list = self._split_data(y_list, split_ratio)
        # X_train_list, y_train_list, X_val_list, y_val_list = self._split_data(X_list, y_list, split_ratio)
        
        encoder_super_sequence_train = tf.convert_to_tensor(np.concatenate([elem[0] for elem in X_train_list], axis=0), dtype=tf.float64)
        decoder_super_sequence_train = tf.convert_to_tensor(np.concatenate([elem[1] for elem in X_train_list], axis=0), dtype=tf.float64)
        self.x_train = [encoder_super_sequence_train, decoder_super_sequence_train]
        encoder_super_sequence_val =  tf.convert_to_tensor(np.concatenate([elem[0] for elem in X_val_list], axis=0), dtype=tf.float64)
        decoder_super_sequence_val =  tf.convert_to_tensor(np.concatenate([elem[1] for elem in X_val_list], axis=0), dtype=tf.float64)
        self.x_val = [encoder_super_sequence_val, decoder_super_sequence_val]
        self.y_train = tf.convert_to_tensor(np.concatenate(y_train_list, axis=0),  dtype=tf.float64)
        self.y_val = tf.convert_to_tensor(np.concatenate(y_val_list, axis=0),  dtype=tf.float64)
    
    def train(self):
        
        self.model = TCN_TCN_Attention()
        self.model.build(
            num_layers_tcn = None,
            num_filters = 16,
            kernel_size = 3,
            dilation_base = 2,
            dropout_rate = 0.2,
            key_size = 6,
            value_size = 6,
            num_attention_heads = 1,
            neurons_output = [480],
            activation = "relu",
            kernel_initializer = "he_normal",
            batch_norm_tcn = True,
            layer_norm_tcn = False,
            autoregressive=False,
            padding_encoder='causal',
            padding_decoder='causal',
            current_epoch = self.current_epoch,
            total_epochs = self.total_epochs,)

        self.model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.02) , run_eagerly = True)      #decay=1e-3
        # self.model.compile(optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.001))
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        custom_callback = CustomCallback(self.model, self.current_epoch, self.total_epochs)
        self.model.fit(self.x_train,
                        self.y_train,
                        (self.x_val, self.y_val),
                        epochs=self.total_epochs,
                        batch_size=512,
                        callbacks = [custom_callback,cb_early_stopping]
                        )
        # self.model.summary()
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
        


config_path = "C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\EngineHealthPred\\ModelWeights\\RULall"
input_file = "C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_final_all_1.zip"
test_file = "C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_final_test_all_1.zip"
trainer = ModelTrainer(config_path,input_file, test_file)
trainer.combine_data()
trainer.combine_data_test()
trainer.test_train_dataset()
# print('x')
trainer.train()