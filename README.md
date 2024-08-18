# RUL Prediction

## Task

- Using a custom developed TCN based model architecture to predict Remaining Useful Life (RUL) of turbofan engines from CMAPSS dataset 

![Custom Model Architecture](Images/Arch.png)

![TCN Architecture](Images/TCN.png)

![Improved Self Attention](Images/ISA.png)

![Dilations](Images/Dilations.png)

## Data

-[CMAPSS Data](https://data.nasa.gov/Aeorspace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)

-[Challenge Data](https://data.nasa.gov/download/nk8v-ckry/application%2Fzip)

## Data Citations

- CMAPSS -  A. Saxena and K. Goebel (2008). “Turbofan Engine Degradation Simulation Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

- Challenge - A. Saxena and K. Goebel (2008). “PHM08 Challenge Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA


## Environment

- python 3.10.8
- pandas
- numpy
- scipy
- tensorflow
- scikit-learn


## WorkFlow

- [rul_prediction/EngineHealthPred/src/data_preprocessing/DataProcess.py] -- Preprocesses the raw training text files of the Challenge dataset
- [rul_prediction/EngineHealthPred/src/data_preprocessing/DataProcessTest.py] -- Preprocesses the raw test text files of the Challenge dataset
- [rul_prediction/EngineHealthPred/src/CombineZips.py] -- Combines individual units of engines and formats them in a format of individual csv files for each individual engine unit , to a single zip file for every engine type (4 types)
- [rul_prediction/EngineHealthPred/src/AttentionTCN.py] -- Contains all the functions to process the training data in machine inter-pretable format ,train the final model with processed data and to do a Bayesian Hyper-Parameter Check with processed training data. Uses helper functions for preprocessing and model trainingg.
- [src/tcn_sequence_models/utils] - Contains various helper functions which are used in preprocessing abd traing / validating different models
- [src/tcn_sequence_models/data_processing] - Contains the various training/testing data processing helper programs which are used to sequence the data using sliding window to generate sequences and to do necessary processing steps like normalization, scaling etc. Also has optional functions to apply seasonal temporal encoding, one-hot encoding etc. to data.
- [src/tcn_sequence_models/tf_models] - Contains diffrent model architecture folders. Each model folder uses a TCN based architecture and has their own encoder and decoder class, and a program which ties together these classes as a feed forward layer class. The tcn.py contains the Temporal Convolutional Network Block. We have used the TCN_TCN_Attention model for our experiments based on results.
- [src/tcn_sequence_models/models.py] - Ties together all the mdoels as different classes and also provides model training/testing and hyper-parameter checking functions.
- [rul_prediction/EngineHealthPred/src/Test.ipynb] - Contains the various tests conducted.
- [rul_prediction/EngineHealthPred/Models] - Contains saved models .
