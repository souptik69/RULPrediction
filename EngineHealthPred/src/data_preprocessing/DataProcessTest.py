import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
from zipfile import ZipFile

class EngineDataTestProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.delimiter = ' '
        self.columns = [
            'unit_number',
            'time_cycles',
            'operational_setting_1',
            'operational_setting_2',
            'operational_setting_3'
        ] + [f'sensor_measurement_{i}' for i in range(1, 27)]  # Adjusted to 21 sensor measurements

    def load_data(self):
        df = pd.read_csv(self.file_path, delimiter=self.delimiter, header=None, names=self.columns)
        return df

    def drop_nan_columns(self, df):
        df.dropna(axis=1, inplace=True)
        return df
    
    def calculate_rul(self, df, Rearly =125):
        # Calculate the maximum time_cycles for each unit_number
        df_max_cycle = df.groupby('unit_number')['time_cycles'].max().reset_index()
        df_max_cycle = df_max_cycle.rename(columns={'time_cycles': 'max_cycles'})
        df = df.merge(df_max_cycle, on='unit_number', how='left')
        # if df['time_cycles'] >= df['max_cycles'] - Rearly:
        #     df['RUL'] = df['max_cycles'] - df['time_cycles']
        # else:
        #     df['RUL'] = Rearly
        df['RUL'] = df['max_cycles'] - df['time_cycles']
        condition = df['time_cycles'] >= (df['max_cycles'] - Rearly)
        df.loc[condition, 'RUL_label'] = df['max_cycles'] - df['time_cycles']
        df.loc[~condition, 'RUL_label'] = Rearly
        df.drop(['max_cycles'], axis=1, inplace=True)

        return df
    
    def calculate_rul_from_last_timestep(self,df, last_timestep_rul_file, Rearly = 125):
        with open(last_timestep_rul_file, 'r') as file:
            last_timestep_ruls = [float(line.strip()) for line in file.readlines()]

        if len(last_timestep_ruls) != len(df['unit_number'].unique()):
            raise ValueError("Number of RUL values does not match the number of engine units.")
        for unit_number, rul_value in zip(df['unit_number'].unique(), last_timestep_ruls):
            condition = df['unit_number'] == unit_number
            max_cycle = df.loc[condition, 'time_cycles'].max()
            df.loc[condition, 'RUL'] = rul_value + (max_cycle - df.loc[condition, 'time_cycles'])
            condition = df['RUL'] > Rearly
            df.loc[condition, 'RUL'] = Rearly


        return df
        
    
    def moving_average_filter(self, data, window_size):
        return data.rolling(window=window_size).mean().fillna(method='bfill')

    def apply_moving_average_filter(self, df, window_size=5):
        sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
        for sensor in sensor_columns:
            df[sensor] = self.moving_average_filter(df[sensor], window_size)
        return df

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def apply_low_pass_filter(self, df):
        fs = 1.0  # Sampling frequency
        cutoff = 0.1  # Cutoff frequency
        sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
        for sensor in sensor_columns:
            df[sensor] = self.butter_lowpass_filter(df[sensor].values, cutoff, fs)
        return df

    def generate_delta_columns(self, df):
        sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
        for col in sensor_columns:
            delta_col = f"delta_{col}"
            df[delta_col] = df.groupby('unit_number')[col].diff().fillna(0)
        return df

    # def normalize_data(self, df):
    #     sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
    #     delta_columns = [f"{col}" for col in sensor_columns]
    #     for col in sensor_columns + delta_columns:
    #         df[col] = df.groupby('unit_number')[col].transform(lambda x: (x - x.mean()) / x.std())
    #     return df
    def normalize_data(self, df):
        sensor_columns = [col for col in df.columns if 'sensor_measurement' in col]
        op_setting_columns = [col for col in df.columns if col.startswith('operational_setting')]
        all_columns = sensor_columns + op_setting_columns
        # for col in sensor_columns:
        #     max_val = df[col].max()
        #     min_val = df[col].min()
        #     df[col] = (df[col] - min_val) / (max_val - min_val)

        # for col in sensor_columns:
        #     max_vals = df.groupby('unit_number')[col].max()
        #     min_vals = df.groupby('unit_number')[col].min()
        #     df[col] = df.groupby('unit_number')[col].apply(lambda x: (x - min_vals[x.name]) / (max_vals[x.name] - min_vals[x.name]))
        
        for col in all_columns:
            max_vals = df.groupby('unit_number')[col].max()
            min_vals = df.groupby('unit_number')[col].min()
            df[col] = df.groupby('unit_number')[col].apply(lambda x: (x - min_vals[x.name]) / (max_vals[x.name] - min_vals[x.name]))

        return df

    def calculate_rolling_statistics(self, df):
        window_size = 10  # Window size for rolling statistics
        for i in [1, 2, 3]:
            df[f'operational_setting_{i}_rolling_mean'] = df.groupby('unit_number')[f'operational_setting_{i}'].transform(lambda x: x.rolling(window=window_size).mean())
            df[f'operational_setting_{i}_rolling_variance'] = df.groupby('unit_number')[f'operational_setting_{i}'].transform(lambda x: x.rolling(window=window_size).var())
        return df

    def compute_rul_transformations(self, df):
        df['log_RUL'] = np.log1p(df['RUL'])
        df['sqrt_RUL'] = np.sqrt(df['RUL'])
        df['exp_RUL'] = np.exp(df['RUL'] / df['RUL'].max()) - 1
        df['inverse_RUL'] = 1 / (df['RUL'] + 1)
        return df

    def segregate_and_zip(self, df):
        zip_filename = "processed_data_test_FD003_all.zip"
        with ZipFile(zip_filename, 'w') as zipf:
            for unit_number in df['unit_number'].unique():
                csv_filename = f"unit_{unit_number}_3_processed.csv"
                df_unit = df[df['unit_number'] == unit_number]
                df_unit.to_csv(csv_filename, index=False)
                zipf.write(csv_filename)
                os.remove(csv_filename)  # Clean up the CSV file after adding to zip

        return zip_filename

    def process_data(self, last_timestep_rul_file):
        df = self.load_data()
        df = self.drop_nan_columns(df)
        # df = self.calculate_rul(df)
        # df = self.apply_low_pass_filter(df)
        df = self.apply_moving_average_filter(df, window_size=5)
        df = self.generate_delta_columns(df)
        df = self.normalize_data(df)
        # df = self.calculate_rolling_statistics(df)
        # df = self.compute_rul_transformations(df)
        df = self.calculate_rul_from_last_timestep(df,last_timestep_rul_file, Rearly= 125)
        zip_file = self.segregate_and_zip(df)
        return zip_file
    
# Example usage
processor = EngineDataTestProcessor(file_path='C:\\Users\\ssen\\Documents\\FlexKI\\EngineHealthPred\\ChallengeData\\Challenge_Data\\CMAPSSData\\test_FD003.txt')
last_timestep_rul_file = 'C:\\Users\\ssen\\Documents\\FlexKI\\EngineHealthPred\\ChallengeData\\Challenge_Data\\CMAPSSData\\RUL_FD003.txt'
zip_file = processor.process_data(last_timestep_rul_file)