import os
import zipfile
import pandas as pd
from tcn_sequence_models.data_processing.preprocessor import Preprocessor

class DataProcessor:
    def __init__(self, zip_file_path, save_folder):
        self.zip_file_path = zip_file_path
        self.save_folder = save_folder

    def process_data_from_zip(self):
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_file_names = zip_ref.namelist()
            for file_name in zip_file_names:
                if file_name.endswith('.csv'):
                    csv_file_name = os.path.splitext(os.path.basename(file_name))[0]
                    folder_path = os.path.join(self.save_folder, csv_file_name)
                    os.makedirs(folder_path, exist_ok=True)
                    csv_file_path = zip_ref.extract(file_name, folder_path)
                    self.process_and_save(csv_file_path, folder_path)

    def process_and_save(self, csv_file_path, save_folder):
        df = pd.read_csv(csv_file_path)
        
        time_col = 'time_cycles'
        # df[time_col] = pd.to_datetime(df[time_col])
        if 'time_cycles' in df.columns:
            df.set_index('time_cycles', inplace=True)
        df.fillna(0, inplace=True)
        split_ratio = 0.8
        input_seq_len = 40
        output_seq_len = 1
        # features_input_encoder = ['time_cycles','operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_1','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_5','sensor_measurement_6','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_10','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_16','sensor_measurement_17','sensor_measurement_18','sensor_measurement_19','sensor_measurement_20','sensor_measurement_21']
        features_input_encoder = ['sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21']
        features_input_decoder = ['sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_17','sensor_measurement_20','sensor_measurement_21']
        # features_input_decoder = ['time_cycles','operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_1','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_5','sensor_measurement_6','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_10','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_16','sensor_measurement_17','sensor_measurement_18','sensor_measurement_19','sensor_measurement_20','sensor_measurement_21']
        
        

        # features_input_decoder = ['time_cycles','operational_setting_1','operational_setting_2','operational_setting_3','sensor_measurement_1','sensor_measurement_2','sensor_measurement_3','sensor_measurement_4','sensor_measurement_5','sensor_measurement_6','sensor_measurement_7','sensor_measurement_8','sensor_measurement_9','sensor_measurement_10','sensor_measurement_11','sensor_measurement_12','sensor_measurement_13','sensor_measurement_14','sensor_measurement_15','sensor_measurement_16','sensor_measurement_17','sensor_measurement_18','sensor_measurement_19','sensor_measurement_20','sensor_measurement_21']

        feature_target = ['RUL']
        preprocessor = Preprocessor(df)
        preprocessor.process(
            features_input_encoder,
            features_input_decoder,
            feature_target,
            input_seq_len,
            output_seq_len,
            model_type="tcn_tcn",
            time_col=df.index,
            split_ratio=split_ratio,
            split_date=None,
            temporal_encoding_modes=None,
            autoregressive=True
        )
        
        save_path = save_folder
        preprocessor.save_preprocessor_config(save_path)

# Example usage
zip_file_path = "C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_final_1_3.zip"
save_folder = "C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\Test_preprocessor"
processor = DataProcessor(zip_file_path, save_folder)
processor.process_data_from_zip()
