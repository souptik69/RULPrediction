from tcn_sequence_models.data_processing.preprocessor import Preprocessor
from tcn_sequence_models.models import TCN_TCN_Attention
from tcn_sequence_models.utils.scaling import inverse_scale_sequences
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = TCN_TCN_Attention()
df = pd.read_csv("C:\\Users\\ssen\\Documents\\FlexKI\\EngineHealthPred\\ChallengeData\\processed_data_final\\unit_56_processed.csv")
preprocessor = Preprocessor(df)
preprocessor.load_preprocessor_config(load_path="C:\\Users\\ssen\\Documents\\FlexKI\\EngineHealthPred\\ChallengeData\\Preprocess_configs\\unit_56_processed")
preprocessor.process_from_config_inference()
config_path = 'C:\\Users\\ssen\\Documents\\FlexKI\\search_final_RUL\\untitled_project\\trial_051'
model.load_model(config_path, preprocessor.X[:3], is_training_data=False)
y_pred = model.predict(preprocessor.X[:3])
# print('y_pred',y_pred.shape)
y_true= preprocessor.y
# print('y_true',y_true.shape)
y_true_original = np.expm1(y_true)
# print('y_true_original',y_true_original.shape)
y_pred_original = np.expm1(y_pred)
# print('y_pred_original',y_pred_original.shape)





# Plotting
y_true_flat = y_true_original.reshape(-1)
y_pred_flat = y_pred_original.reshape(-1)

# Plotting
plt.figure(figsize=(15, 8))

# Plot true values
plt.plot(y_true_flat, label='True RUL Values')

# Plot predicted values
plt.plot(y_pred_flat, label='Predicted RUL Values')

plt.title('True vs Predicted Values for Engine 56')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

