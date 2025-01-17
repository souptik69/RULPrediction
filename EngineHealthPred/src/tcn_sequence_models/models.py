import json
import os
from abc import ABC
from abc import abstractmethod
from tcn_sequence_models.tf_models.tf_TCN_GRU import tcn_gru_attention_model
from tcn_sequence_models.tf_models.tf_TCN_TCN_attention import tcn_tcn_attention
from tcn_sequence_models.tf_models.tf_TCN_TCN_attention import tcn_tcn_attention_transfer
from tcn_sequence_models.tf_models.tf_TCN_TCN_normal import tcn_tcn_normal
from tcn_sequence_models.utils.scaling import inverse_scale_sequences
from typing import List
from typing import Optional
import tensorflow as tf
import numpy as np
from keras_tuner import BayesianOptimization
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa


class BaseModel(ABC):
    def __init__(self):
        """BaseModel class from which the other models inherit"""
        self.model = None
        pass
    
    def rmse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


        
    def compute_score(y_pred, y_true):
        dn = y_pred - y_true
        score_negative = np.sum(np.exp(-dn[dn < 0] / 13) - 1)
        score_nonnegative = np.sum(np.exp(dn[dn >= 0] / 10) - 1)
        return score_negative + score_nonnegative


    def compile(
        self, loss='mean_squared_error', metrics=[compute_score,rmse,tfa.metrics.r_square.RSquare()], optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-3), **kwargs
    ):
        self.model.compile(loss=loss, metrics=metrics, optimizer=optimizer, **kwargs)

    def summary(self):
        self.model.summary()

    def trainable(self):
        for layer in self.model.layers:
            layer.trainable = True

    def fit(
        self,
        X_train,
        y_train,
        validation_data=None,
        epochs=50,
        batch_size=64,
        callbacks=EarlyStopping(patience=5, restore_best_weights=True),
        **kwargs
    ):
        """Fitting function. Same as for TensorFlow model class

        :param X_train: input training data
        :param y_train: target training data
        :param validation_data: tuple with the validation data (X_val, y_val)
        :param epochs: number of epochs
        :param batch_size: batch size
        :param callbacks: callbacks
        :param kwargs: additional parameters
        :return:
        """
        self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )

    def predict(self, X, scaler=None):
        """Make predictions

        :param X: X values
        :param scaler: scaler. If given, scales the output sequences back to the
        original values
        :return: the predicted sequences
        """
        preds = []
        pred = self.model.predict(X)
        if scaler:
            pred = inverse_scale_sequences(pred, scaler)
        preds.append(pred)
        pred = np.mean(np.array(preds), axis=0)
        return pred

    # def add(self,layer = tf.keras.layers.Dense(64,activation='relu')):
    #     self.model.add(layer)


    def eval(self, X, y, scaler_y=None, metric=metrics.mean_squared_error):
        """Evaluate the score of the model

        :param X: X values
        :param y: Ground truths of target
        :param scaler_y: scaler with which y-values were scaled. If None, data is not
        scaled before computing the score
        :param metric: metric from sklearn library
        :return:
        """
        X = X.copy()
        y = y.copy()
        scores = []
        preds = self.predict(X, scaler_y)
        # mse = metrics.mean_squared_error(y,preds)
        if scaler_y is not None:
            y = inverse_scale_sequences(y, scaler_y)
        for i, pred in enumerate(preds):
            scores.append(metric(pred, y[i]))

        mean_score = np.mean(scores)
        return mean_score

    @abstractmethod
    def build(self, **kwargs):
        """Initialize the models with their parameters

        :param kwargs: parameters used for building
        :return:
        """
        pass

    @abstractmethod
    def parameter_search(self, **kwargs):
        """Automized parameter search using KerasTuner and Bayessian optimization

        :param kwargs: parameters needed for the parameter search
        :return:
        """
        pass

    @abstractmethod
    def save_model(self, save_path):
        """Save the model configuration and weights

        :param save_path: the path to the directory where the model files shall be saved
        :return:
        """
        pass

    # @abstractmethod
    # def add(self, layer):
    #     """Adds a layer to the existing model
    #     :return:
    #     """
    #     pass

    def load_model(self, load_path, X, is_training_data: bool):
        """Load model configuration and weights

        :param load_path: the path to the directory from where the model shall be loaded
        :param X: Input values for the model. Needed to call the model to build it.
        :param is_training_data: whether X is training data or inference data
        :return:
        """
        X_init = []
        for el in X:
            X_init.append(el[:1])
        config_file_dir = os.path.join(load_path, "model_config.json")
        # config_file_dir = os.path.join(load_path, "trial.json")
        config_dict = json.load(open(config_file_dir))
        self.build(**config_dict)
        # Call once to set weights later
        self.model(X_init, training=is_training_data)
        load_dir = os.path.join(load_path, "model_weights.h5")
        # model_weights = tf.keras.models.load_model(load_dir)
        self.model.load_weights(load_dir)

    def load_model_Transfer(self, load_path, X, is_training_data: bool):
        X_init = []
        for el in X:
            X_init.append(el[:1])
        config_file_dir = os.path.join(load_path, "model_config.json")
        config_dict = json.load(open(config_file_dir))
        self.build(**config_dict)
        self.model(X_init, training=is_training_data)
        load_dir = os.path.join(load_path, "model_weights.h5")
        self.model.load_weights(load_dir, by_name=True, skip_mismatch=True)
        # self.model.summary()
        # return self.model
        
        
class TCN_TCN_attention_transfer(BaseModel):
    def __init__(self):
        """Model with TCN as encoder and TCN as decoder. Refer to the implementation
        of the model for further information.

        """
        super().__init__()
        self.model = None
        self.num_filters = None
        self.kernel_size = None
        self.dilation_base = None
        self.dropout_rate = None
        self.key_size = None
        self.value_size = None
        self.num_attention_heads = None
        self.neurons_output = None
        self.num_layers_tcn = None
        self.activation = None
        self.kernel_initializer = None
        self.batch_norm_tcn = None
        self.layer_norm_tcn = None
        self.padding_encoder = None
        self.autoregressive = False
        self.transfer = True

    def build(
        self,
        num_filters: int = 12,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout_rate: float = 0.1,
        key_size: int = 4,
        value_size: int = 4,
        num_attention_heads: int = 1,
        neurons_output: List[int] = None,
        num_layers_tcn: int = None,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        batch_norm_tcn: bool = False,
        layer_norm_tcn: bool = True,
        padding_encoder: str = "same",
        padding_decoder: str = "causal",
        autoregressive: bool = False,
        transfer: bool = True,
    ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.neurons_output = neurons_output if neurons_output is not None else [16]
        self.num_layers_tcn = num_layers_tcn
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm_tcn = batch_norm_tcn
        self.layer_norm_tcn = layer_norm_tcn
        self.padding_encoder = padding_encoder
        self.padding_decoder = padding_decoder
        self.autoregressive = autoregressive
        self.transfer = transfer

        self.model = tcn_tcn_attention_transfer.TCN_TCN_attention_transfer(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout_rate=dropout_rate,
            key_size=key_size,
            value_size=value_size,
            num_attention_heads=num_attention_heads,
            neurons_output=neurons_output,
            num_layers_tcn=self.num_layers_tcn,
            activation=activation,
            kernel_initializer=kernel_initializer,
            batch_norm_tcn=batch_norm_tcn,
            layer_norm_tcn=layer_norm_tcn,
            padding_encoder=padding_encoder,
            padding_decoder=padding_decoder,
            autoregressive=self.autoregressive,
            transfer=transfer,
        )
        

    def parameter_search(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        results_path: str,
        batch_size: int = 16,
        patience=5,
        loss: str = "mse",
        optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-3),
        max_trials: int = 30,
        executions_per_trial: int = 1,
        num_filters: Optional[List[int]] = None,
        kernel_size: Optional[List[int]] = None,
        dilation_base: Optional[List[int]] = None,
        dropout_rate: Optional[List[float]] = None,
        key_value_size: Optional[List[int]] = None,
        num_attention_heads: Optional[List[int]] = None,
        neurons_output: Optional[List[int]] = None,
        activation: Optional[List[str]] = None,
        kernel_initializer: Optional[List[str]] = None,
        batch_norm_tcn: Optional[List[bool]] = None,
        layer_norm_tcn: Optional[List[bool]] = None,
        padding_encoder: Optional[List[str]] = None,
        padding_decoder: Optional[List[str]] = None,
    ):
        # init parameters
        if num_filters is None:
            num_filters = [6]
        if kernel_size is None:
            kernel_size = [3]
        if dilation_base is None:
            dilation_base = [2]
        if dropout_rate is None:
            dropout_rate = [0.1]
        if neurons_output is None:
            neurons_output = [32]
        if activation is None:
            activation = ["elu"]
        if kernel_initializer is None:
            kernel_initializer = ["he_normal"]
        if batch_norm_tcn is None:
            batch_norm_tcn = [True]
        if layer_norm_tcn is None:
            layer_norm_tcn = [False]
        if padding_encoder is None:
            padding_encoder = ["same"]
        if padding_decoder is None:
            padding_decoder = ["causal"]

        def create_model(
            hp,
        ):
            hp_num_filters = hp.Choice("num_filters", num_filters)
            hp_kernel_size = hp.Choice("kernel_size", kernel_size)
            hp_dilation_base = hp.Choice("dilation_base", dilation_base)
            hp_dropout_rate = hp.Choice("dropout_rate", dropout_rate)
            hp_key_value_size = hp.Choice("key_value_size", key_value_size)
            hp_num_attention_heads = hp.Choice(
                "num_attention_heads", num_attention_heads
            )
            hp_neurons_output = hp.Choice("neurons_output", neurons_output)
            hp_activation = hp.Choice("activation", activation)
            hp_kernel_initializer = hp.Choice("kernel_initializer", kernel_initializer)
            hp_batch_norm_tcn = hp.Choice("batch_norm_tcn", batch_norm_tcn)
            hp_layer_norm_tcn = hp.Choice("layer_norm_tcn", layer_norm_tcn)
            hp_padding_encoder = hp.Choice("padding_encoder", padding_encoder)
            hp_padding_decoder = hp.Choice("padding_decoder", padding_decoder)

            model = tcn_tcn_attention_transfer.TCN_TCN_attention_transfer(
                num_layers_tcn=None,
                num_filters=hp_num_filters,
                kernel_size=hp_kernel_size,
                dilation_base=hp_dilation_base,
                dropout_rate=hp_dropout_rate,
                key_size=hp_key_value_size,
                value_size=hp_key_value_size,
                num_attention_heads=hp_num_attention_heads,
                neurons_output=[hp_neurons_output],
                activation=hp_activation,
                kernel_initializer=hp_kernel_initializer,
                batch_norm_tcn=hp_batch_norm_tcn,
                layer_norm_tcn=hp_layer_norm_tcn,
                autoregressive=self.autoregressive,
                padding_encoder=hp_padding_encoder,
                padding_decoder=hp_padding_decoder,
            )

            model.compile(loss=loss, optimizer=optimizer)
            return model

        tuner = BayesianOptimization(
            create_model,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=results_path,
        )

        cb_early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        tuner.search(
            X_train,
            y_train,
            epochs=15,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[cb_early_stopping],
            shuffle=True,
        )

    def save_model(self, save_path):
        """Create and save json file of model configs

        :param save_path: path to save the config
        :return:
        """
        # Create and save json of model configs

        config_file_dir = os.path.join(save_path, "model_config.json")
        model_file_dir = os.path.join(save_path, "model_weights.h5")
        config_dict = {
            "num_filters": self.num_filters,
            "num_layers_tcn": self.num_layers_tcn,
            "kernel_size": self.kernel_size,
            "dilation_base": self.dilation_base,
            "dropout_rate": self.dropout_rate,
            "key_size": self.key_size,
            "value_size": self.value_size,
            "num_attention_heads": self.num_attention_heads,
            "neurons_output": self.neurons_output,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "batch_norm_tcn": self.batch_norm_tcn,
            "layer_norm_tcn": self.layer_norm_tcn,
            "autoregressive": self.autoregressive,
            "padding_encoder": self.padding_encoder,
            "padding_decoder": self.padding_decoder,
        }

        json.dump(config_dict, open(config_file_dir, "w"))

        # Save model
        self.model.save_weights(model_file_dir)
    
    # def add(self, layer):
    #     self.model.add(tf.keras.layers.Dense(6, activation='relu',trainable ='True'))            

class TCN_TCN_Attention(BaseModel):
    def __init__(self):
        """Model with TCN as encoder and TCN as decoder. Refer to the implementation
        of the model for further information.

        """
        super().__init__()
        self.model = None
        self.num_filters = None
        self.kernel_size = None
        self.dilation_base = None
        self.dropout_rate = None
        self.key_size = None
        self.value_size = None
        self.num_attention_heads = None
        self.neurons_output = None
        self.num_layers_tcn = None
        self.activation = None
        self.kernel_initializer = None
        self.batch_norm_tcn = None
        self.layer_norm_tcn = None
        self.padding_encoder = None
        self.autoregressive = True
        self.current_epoch = None
        self.total_epochs = None

    def rmse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    def build(
        self,
        num_filters: int = 12,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout_rate: float = 0.1,
        key_size: int = 4,
        value_size: int = 4,
        num_attention_heads: int = 1,
        neurons_output: List[int] = None,
        num_layers_tcn: int = None,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        batch_norm_tcn: bool = False,
        layer_norm_tcn: bool = True,
        padding_encoder: str = "same",
        padding_decoder: str = "causal",
        autoregressive: bool = True,
        current_epoch = None,
        total_epochs = None,
    ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.neurons_output = neurons_output if neurons_output is not None else [16]
        self.num_layers_tcn = num_layers_tcn
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm_tcn = batch_norm_tcn
        self.layer_norm_tcn = layer_norm_tcn
        self.padding_encoder = padding_encoder
        self.padding_decoder = padding_decoder
        self.autoregressive = autoregressive
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

        self.model = tcn_tcn_attention.TCN_TCN_attention(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout_rate=dropout_rate,
            key_size=key_size,
            value_size=value_size,
            num_attention_heads=num_attention_heads,
            neurons_output=neurons_output,
            num_layers_tcn=self.num_layers_tcn,
            activation=activation,
            kernel_initializer=kernel_initializer,
            batch_norm_tcn=batch_norm_tcn,
            layer_norm_tcn=layer_norm_tcn,
            padding_encoder=padding_encoder,
            padding_decoder=padding_decoder,
            autoregressive=self.autoregressive,
            current_epoch = current_epoch,
            total_epochs = total_epochs,
        )

    def parameter_search(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        results_path: str,
        batch_size: int = 16,
        patience=5,
        loss = rmse,
        optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-3),
        max_trials: int = 30,
        executions_per_trial: int = 1,
        num_filters: Optional[List[int]] = None,
        kernel_size: Optional[List[int]] = None,
        dilation_base: Optional[List[int]] = None,
        dropout_rate: Optional[List[float]] = None,
        key_value_size: Optional[List[int]] = None,
        num_attention_heads: Optional[List[int]] = None,
        neurons_output: Optional[List[int]] = None,
        activation: Optional[List[str]] = None,
        kernel_initializer: Optional[List[str]] = None,
        batch_norm_tcn: Optional[List[bool]] = None,
        layer_norm_tcn: Optional[List[bool]] = None,
        padding_encoder: Optional[List[str]] = None,
        padding_decoder: Optional[List[str]] = None,
    ):
        # init parameters
        if num_filters is None:
            num_filters = [6]
        if kernel_size is None:
            kernel_size = [3]
        if dilation_base is None:
            dilation_base = [2]
        if dropout_rate is None:
            dropout_rate = [0.1]
        if neurons_output is None:
            neurons_output = [32]
        if activation is None:
            activation = ["elu"]
        if kernel_initializer is None:
            kernel_initializer = ["he_normal"]
        if batch_norm_tcn is None:
            batch_norm_tcn = [True]
        if layer_norm_tcn is None:
            layer_norm_tcn = [False]
        if padding_encoder is None:
            padding_encoder = ["same"]
        if padding_decoder is None:
            padding_decoder = ["causal"]

        def create_model(
            hp,
        ):
            hp_num_filters = hp.Choice("num_filters", num_filters)
            hp_kernel_size = hp.Choice("kernel_size", kernel_size)
            hp_dilation_base = hp.Choice("dilation_base", dilation_base)
            hp_dropout_rate = hp.Choice("dropout_rate", dropout_rate)
            hp_key_value_size = hp.Choice("key_value_size", key_value_size)
            hp_num_attention_heads = hp.Choice(
                "num_attention_heads", num_attention_heads
            )
            hp_neurons_output = hp.Choice("neurons_output", neurons_output)
            hp_activation = hp.Choice("activation", activation)
            hp_kernel_initializer = hp.Choice("kernel_initializer", kernel_initializer)
            hp_batch_norm_tcn = hp.Choice("batch_norm_tcn", batch_norm_tcn)
            hp_layer_norm_tcn = hp.Choice("layer_norm_tcn", layer_norm_tcn)
            hp_padding_encoder = hp.Choice("padding_encoder", padding_encoder)
            hp_padding_decoder = hp.Choice("padding_decoder", padding_decoder)

            model = tcn_tcn_attention.TCN_TCN_attention(
                num_layers_tcn=None,
                num_filters=hp_num_filters,
                kernel_size=hp_kernel_size,
                dilation_base=hp_dilation_base,
                dropout_rate=hp_dropout_rate,
                key_size=hp_key_value_size,
                value_size=hp_key_value_size,
                num_attention_heads=hp_num_attention_heads,
                neurons_output=[hp_neurons_output],
                activation=hp_activation,
                kernel_initializer=hp_kernel_initializer,
                batch_norm_tcn=hp_batch_norm_tcn,
                layer_norm_tcn=hp_layer_norm_tcn,
                autoregressive=self.autoregressive,
                padding_encoder=hp_padding_encoder,
                padding_decoder=hp_padding_decoder,
            )

            model.compile(loss=loss, optimizer=optimizer)
            return model

        tuner = BayesianOptimization(
            create_model,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=results_path,
        )

        cb_early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        tuner.search(
            X_train,
            y_train,
            epochs=15,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[cb_early_stopping],
            shuffle=True,
        )

    def save_model(self, save_path):
        """Create and save json file of model configs

        :param save_path: path to save the config
        :return:
        """
        # Create and save json of model configs

        config_file_dir = os.path.join(save_path, "model_config.json")
        model_file_dir = os.path.join(save_path, "model_weights.h5")
        config_dict = {
            "num_filters": self.num_filters,
            "num_layers_tcn": self.num_layers_tcn,
            "kernel_size": self.kernel_size,
            "dilation_base": self.dilation_base,
            "dropout_rate": self.dropout_rate,
            "key_size": self.key_size,
            "value_size": self.value_size,
            "num_attention_heads": self.num_attention_heads,
            "neurons_output": self.neurons_output,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "batch_norm_tcn": self.batch_norm_tcn,
            "layer_norm_tcn": self.layer_norm_tcn,
            "autoregressive": self.autoregressive,
            "padding_encoder": self.padding_encoder,
            "padding_decoder": self.padding_decoder,
        }

        json.dump(config_dict, open(config_file_dir, "w"))

        # Save model
        self.model.save_weights(model_file_dir)
    
    # def add(self, layer):
    #     self.model.add(tf.keras.layers.Dense(6, activation='relu',trainable ='True'))


class TCN_TCN_Normal(BaseModel):
    def __init__(self):
        """Model with TCN as encoder and TCN as decoder. Refer to the implementation
        of the model for further information.

        """
        super().__init__()
        self.model = None
        self.num_filters = None
        self.kernel_size = None
        self.dilation_base = None
        self.dropout_rate = None
        self.neurons_output = None
        self.num_layers_tcn = None
        self.activation = None
        self.kernel_initializer = None
        self.batch_norm_tcn = None
        self.layer_norm_tcn = None
        self.padding_decoder = None
        self.autoregressive = False

    def build(
        self,
        num_filters: int = 12,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout_rate: float = 0.1,
        neurons_output: List[int] = None,
        num_layers_tcn: int = None,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        batch_norm_tcn: bool = False,
        layer_norm_tcn: bool = True,
        padding_encoder: str = "same",
        padding_decoder: str = "causal",
        autoregressive: bool = False,
    ):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate

        self.neurons_output = neurons_output if neurons_output is not None else [16]
        self.num_layers_tcn = num_layers_tcn
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm_tcn = batch_norm_tcn
        self.layer_norm_tcn = layer_norm_tcn
        self.padding_decoder = padding_decoder
        self.autoregressive = autoregressive

        self.model = tcn_tcn_normal.TCN_TCN_Normal(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout_rate=dropout_rate,
            neurons_output=neurons_output,
            num_layers_tcn=self.num_layers_tcn,
            activation=activation,
            kernel_initializer=kernel_initializer,
            batch_norm_tcn=batch_norm_tcn,
            layer_norm_tcn=layer_norm_tcn,
            padding_decoder=padding_decoder,
            autoregressive=self.autoregressive,
        )

    def parameter_search(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        results_path: str,
        batch_size: int = 16,
        patience=5,
        loss: str = "mse",
        optimizer=tf.keras.optimizers.legacy.Adam(lr=0.01, decay=1e-3),
        max_trials: int = 20,
        executions_per_trial: int = 1,
        num_filters: Optional[List[int]] = None,
        kernel_size: Optional[List[int]] = None,
        dilation_base: Optional[List[int]] = None,
        dropout_rate: Optional[List[float]] = None,
        neurons_output: Optional[List[int]] = None,
        activation: Optional[List[str]] = None,
        kernel_initializer: Optional[List[str]] = None,
        batch_norm_tcn: Optional[List[bool]] = None,
        layer_norm_tcn: Optional[List[bool]] = None,
        padding_decoder: Optional[List[str]] = None,
    ):
        # init parameters
        if num_filters is None:
            num_filters = [6]
        if kernel_size is None:
            kernel_size = [3]
        if dilation_base is None:
            dilation_base = [2]
        if dropout_rate is None:
            dropout_rate = [0.1]
        if neurons_output is None:
            neurons_output = [32]
        if activation is None:
            activation = ["elu"]
        if kernel_initializer is None:
            kernel_initializer = ["he_normal"]
        if batch_norm_tcn is None:
            batch_norm_tcn = [True]
        if layer_norm_tcn is None:
            layer_norm_tcn = [False]
        if padding_decoder is None:
            padding_decoder = ["causal"]

        def create_model(
            hp,
        ):
            hp_num_filters = hp.Choice("num_filters", num_filters)
            hp_kernel_size = hp.Choice("kernel_size", kernel_size)
            hp_dilation_base = hp.Choice("dilation_base", dilation_base)
            hp_dropout_rate = hp.Choice("dropout_rate", dropout_rate)
            hp_neurons_output = hp.Choice("neurons_output", neurons_output)
            hp_activation = hp.Choice("activation", activation)
            hp_kernel_initializer = hp.Choice("kernel_initializer", kernel_initializer)
            hp_batch_norm_tcn = hp.Choice("batch_norm_tcn", batch_norm_tcn)
            hp_layer_norm_tcn = hp.Choice("layer_norm_tcn", layer_norm_tcn)
            hp_padding_decoder = hp.Choice("padding_decoder", padding_decoder)

            model = tcn_tcn_normal.TCN_TCN_Normal(
                num_layers_tcn=None,
                num_filters=hp_num_filters,
                kernel_size=hp_kernel_size,
                dilation_base=hp_dilation_base,
                dropout_rate=hp_dropout_rate,
                neurons_output=[hp_neurons_output],
                activation=hp_activation,
                kernel_initializer=hp_kernel_initializer,
                batch_norm_tcn=hp_batch_norm_tcn,
                layer_norm_tcn=hp_layer_norm_tcn,
                autoregressive=self.autoregressive,
                padding_decoder=hp_padding_decoder,
            )

            model.compile(loss=loss, optimizer=optimizer)
            return model

        tuner = BayesianOptimization(
            create_model,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=results_path,
        )

        cb_early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        tuner.search(
            X_train,
            y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[cb_early_stopping],
            shuffle=True,
        )

    def save_model(self, save_path):
        """Create and save json file of model configs

        :param save_path: path to save the config
        :return:
        """
        # Create and save json of model configs

        config_file_dir = os.path.join(save_path, "model_config.json")
        model_file_dir = os.path.join(save_path, "model_weights.h5")
        config_dict = {
            "num_filters": self.num_filters,
            "num_layers_tcn": self.num_layers_tcn,
            "kernel_size": self.kernel_size,
            "dilation_base": self.dilation_base,
            "dropout_rate": self.dropout_rate,
            "neurons_output": self.neurons_output,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "batch_norm_tcn": self.batch_norm_tcn,
            "layer_norm_tcn": self.layer_norm_tcn,
            "autoregressive": self.autoregressive,
            "padding_decoder": self.padding_decoder,
        }

        json.dump(config_dict, open(config_file_dir, "w"))

        # Save model
        self.model.save_weights(model_file_dir)


class TCN_GRU(BaseModel):
    def __init__(self):
        """Model with TCN as encoder and GRU as decoder. Refer to the implementation
        of the model for further information.

        """
        super().__init__()
        self.model = None
        self.hidden_units = None
        self.num_filters = None
        self.kernel_size_enc = None
        self.dilation_base = None
        self.output_size_attention = None
        self.dropout_rate = None
        self.gru_output_neurons = None
        self.key_size = None
        self.value_size = None
        self.num_attention_heads = (None,)
        self.activation = None
        self.kernel_initializer = None
        self.padding_enc = None
        self.batch_norm = None
        self.layer_norm = None

    def build(
        self,
        hidden_units: int,
        num_filters: int,
        kernel_size_enc: int,
        dilation_base: int,
        output_size_attention: int,
        dropout_rate: float,
        gru_output_neurons: [int],
        key_size: int,
        value_size: int,
        num_attention_heads: int = 1,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        padding_enc: str = "causal",
        batch_norm: bool = False,
        layer_norm: bool = False,
    ):
        self.hidden_units = hidden_units
        self.num_filters = num_filters
        self.kernel_size_enc = kernel_size_enc
        self.dilation_base = dilation_base
        self.output_size_attention = output_size_attention
        self.dropout_rate = dropout_rate
        self.gru_output_neurons = gru_output_neurons
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.padding_enc = padding_enc
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        self.model = tcn_gru_attention_model.TCN_GRU(
            hidden_units=hidden_units,
            num_filters=num_filters,
            kernel_size_enc=kernel_size_enc,
            dilation_base=dilation_base,
            output_size_attention=output_size_attention,
            dropout_rate=dropout_rate,
            gru_output_neurons=gru_output_neurons,
            key_size=key_size,
            value_size=value_size,
            num_attention_heads=num_attention_heads,
            activation=activation,
            kernel_initializer=kernel_initializer,
            padding_encoder=padding_enc,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )

    def parameter_search(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        results_path: str,
        hidden_units: int,
        num_filters: int,
        kernel_size_enc: int,
        dilation_base: int,
        dropout_rate: float,
        gru_output_neurons: [int],
        key_value_size: int,
        num_attention_heads: int,
        output_size_attention: int,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        padding_enc: str = "causal",
        batch_norm: bool = False,
        layer_norm: bool = False,
        batch_size: int = 32,
        patience=5,
        optimizer=tf.keras.optimizers.legacy.Adam(lr=0.01, decay=1e-3),
        loss: str = "mse",
        max_trials: int = 20,
        executions_per_trial: int = 1,
    ):
        def create_model(
            hp,
        ):
            hp_hidden_units = hp.Choice("hidden_units", hidden_units)
            hp_num_filters = hp.Choice("num_filters", num_filters)
            hp_kernel_size_enc = hp.Choice("kernel_size_enc", kernel_size_enc)
            hp_dilation_base = hp.Choice("dilation_base", dilation_base)
            hp_output_size_attention = hp.Choice(
                "output_size_attention", output_size_attention
            )
            hp_dropout_rate = hp.Choice("dropout_rate", dropout_rate)
            hp_gru_output_neurons = hp.Choice("gru_output_neurons", gru_output_neurons)
            hp_key_value_size = hp.Choice("key_size", key_value_size)
            hp_num_attention_heads = hp.Choice(
                "num_attention_heads", num_attention_heads
            )
            hp_activation = hp.Choice("activation", activation)
            hp_kernel_initializer = hp.Choice("kernel_initializer", kernel_initializer)
            hp_padding_enc = hp.Choice("padding_enc", padding_enc)

            hp_batch_norm = hp.Choice("batch_norm", batch_norm)
            hp_layer_norm = hp.Choice("layer_norm", layer_norm)

            model = tcn_gru_attention_model.TCN_GRU(
                hidden_units=hp_hidden_units,
                num_filters=hp_num_filters,
                kernel_size_enc=hp_kernel_size_enc,
                dilation_base=hp_dilation_base,
                dropout_rate=hp_dropout_rate,
                gru_output_neurons=[hp_gru_output_neurons],
                output_size_attention=hp_output_size_attention,
                key_size=hp_key_value_size,
                value_size=hp_key_value_size,
                num_attention_heads=hp_num_attention_heads,
                activation=hp_activation,
                kernel_initializer=hp_kernel_initializer,
                padding_encoder=hp_padding_enc,
                batch_norm=hp_batch_norm,
                layer_norm=hp_layer_norm,
            )
            model.compile(loss=loss, optimizer=optimizer)
            return model

        tuner = BayesianOptimization(
            create_model,
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=results_path,
        )

        cb_early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        tuner.search(
            X_train,
            y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[cb_early_stopping],
            shuffle=True,
        )

    def save_model(self, save_path):
        # Create and save json of model configs
        config_file_dir = os.path.join(save_path, "model_config.json")
        model_file_dir = os.path.join(save_path, "model_weights.h5")
        config_dict = {
            "hidden_units": self.hidden_units,
            "num_filters": self.num_filters,
            "kernel_size_enc": self.kernel_size_enc,
            "dilation_base": self.dilation_base,
            "dropout_rate": self.dropout_rate,
            "gru_output_neurons": self.gru_output_neurons,
            "key_size": self.key_size,
            "value_size": self.value_size,
            "num_attention_heads": self.num_attention_heads,
            "output_size_attention": self.output_size_attention,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "padding_enc": self.padding_enc,
            "batch_norm": self.batch_norm,
            "layer_norm": self.layer_norm,
        }

        json.dump(config_dict, open(config_file_dir, "w"))

        # Save model
        self.model.save_weights(model_file_dir)

