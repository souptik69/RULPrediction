import tensorflow as tf
from tcn_sequence_models.tf_models.tcn import TCN
from tensorflow.keras.layers import MultiHeadAttention


class Decoder(tf.keras.Model):
    def __init__(
        self,
        max_seq_len: int,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_rate: float,
        key_size: int,
        value_size: int,
        num_attention_heads: int,
        output_neurons: [int],
        num_layers: int,
        activation: str = "elu",
        kernel_initializer: str = "he_normal",
        batch_norm: bool = False,
        layer_norm: bool = False,
        autoregressive: bool = False,
        padding: str = "causal",
    ):
        """TCN Decoder stage
        The Decoder architecture is as follows:
        First a TCN stage is used to encoder the decoder input data.
        After that multi-head cross attention is applied to the TCN output and the
        encoder output.
        The original TCN output is added to the output of the attention and normalized.
        The last stage is the prediction stage (a block of dense layers) that then
        makes the final prediction.

        :param max_seq_len: maximum sequence length that is used to compute the
        number of layers
        :param num_filters: number of channels for CNNs
        :param kernel_size: kernel size of CNNs
        :param dilation_base: dilation base
        :param dropout_rate: dropout rate
        :param key_size: dimensionality of key/query
        :param value_size: dimensionality of value
        :param num_attention_heads: Number of attention heads to be used
        :param output_neurons: list of output neurons. Each entry is a new layer in
        the output stage.
        :param num_layers: number of layer in the TCNs. If None, the needed
        number of layers is computed automatically based on the sequence lengths
        :param activation: the activation function used throughout the decoder
        :param kernel_initializer: the mode how the kernels are initialized
        :param batch_norm: if batch normalization shall be used
        :param layer_norm: if layer normalization shall be used
        :param autoregressive: whether to use autoregression in the decoder or not.
        If True, teacher-forcing is applied during training and autoregression is
        used during inference. If False, groundtruths / predictions of the previous
        step are not used.
        :param padding: Padding mode. One of ['causal', 'same']. If autoregressive =
        True, decoder padding will always be causal and the padding value has
        no effect.
        """
        super(Decoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout_rate = dropout_rate
        self.key_size = key_size
        self.value_size = value_size
        self.num_attention_heads = num_attention_heads
        self.output_neurons = output_neurons
        self.num_layers = num_layers
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.autoregressive = autoregressive
        self.padding = padding if autoregressive is False else "causal"

        self.tcn1 = TCN(
            max_seq_len=self.max_seq_len,
            num_stages=2,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dilation_base=self.dilation_base,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            final_activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            padding=self.padding,
            batch_norm=self.batch_norm,
            layer_norm=self.layer_norm,
            return_sequence=True,
            num_layers=num_layers,
        )

        # Cross attention
        self.attention = MultiHeadAttention(
            key_dim=self.key_size,
            value_dim=self.value_size,
            num_heads=self.num_attention_heads,
            output_shape=self.num_filters,
        )

        # Normalization layer after cross attention
        if self.layer_norm:
            self.normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

        # layers for the final prediction stage
        self.output_layers = []
        for i, neurons in enumerate(self.output_neurons):
            layer_dense = tf.keras.layers.Dense(
                neurons,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
            )
            self.output_layers.append(layer_dense)

        # Last output layer
        self.output_layers.append(tf.keras.layers.Dense(6))

        # Additional dense layers
        self.additional_layers = []
        # layer_flatten = tf.keras.layers.Flatten()
        layer_dense_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_initializer='random_uniform',
        ))
        layer_dropout = tf.keras.layers.Dropout(0.5)
        # layer_reshape = tf.keras.layers.Reshape(batch_size)
        layer_dense_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            6,
            activation='relu',
            kernel_initializer='random_uniform',
        ))
        # self.additional_layers.append(layer_flatten)
        self.additional_layers.append(layer_dense_1)
        self.additional_layers.append(layer_dropout)
        self.additional_layers.append(layer_dense_2)
    # Masking one random feature at every time step of every batch in the data decoder with some random value
    @tf.function
    def random_feature_mask(self, X_decoder):
        batch_size, output_timesteps, num_of_features = X_decoder.shape.as_list()
        batch_size = tf.cast(tf.shape(X_decoder)[0], tf.int32)
        feature_idx = tf.random.uniform(
            shape=(batch_size, output_timesteps),
            minval=0,
            maxval=num_of_features - 1,
            dtype=tf.int32,
        )

        feature_mask = tf.one_hot(feature_idx, depth=num_of_features, axis=-1)[..., :-1]
        feature_mask = tf.cast(feature_mask, dtype=tf.bool)

        X_decoder = tf.cast(X_decoder, dtype=tf.float32)
        new_values = tf.random.normal(
            shape=(batch_size, output_timesteps), mean=0.0, stddev=1.0, dtype=tf.float32
        )

        data_decoder_mod = tf.where(feature_mask, new_values[..., None], X_decoder[:, :, :-1])
        data_decoder_mod = tf.concat([data_decoder_mod, X_decoder[:, :, -1:]], axis=-1)

        return data_decoder_mod

    @tf.function
    def call(self, inputs, training=True):
        if training:
            if self.autoregressive:
                return self._training_call_autoregressive(inputs)
            else:
                return self._call_none_regressive(inputs, training)
        else:
            if self.autoregressive:
                return self._inference_call_autoregressive(inputs)
            else:
                return self._call_none_regressive(inputs, training)

    @tf.function
    def _training_call_autoregressive(self, inputs):
        data_encoder, data_decoder, y_shifted = inputs
        # y_shifted = tf.expand_dims(y_shifted, -1)
        data_decoder = self.random_feature_mask(data_decoder)
        data_decoder = tf.concat([data_decoder, y_shifted], -1)
        # data_decoder = tf.expand_dims(data_decoder,-1)
        out_tcn = self.tcn1(data_decoder, training=True)
        out_attention = self.attention(out_tcn, data_encoder, training=True)
        out = self.normalization_layer(out_tcn + out_attention, training=True)
        # out = tf.concat([out_tcn, out_attention], -1)

        for layer in self.output_layers:
            out = layer(out, training=True)

        for layer in self.additional_layers:
            out = layer(out, training=True)

        return out

    @tf.function
    def _inference_call_autoregressive(self, inputs):
        data_encoder, data_decoder, last_y = inputs
        target_len = data_decoder.shape[1]
        # last_y_reshaped = tf.reshape(last_y, [-1, 1, 1])
        predictions = None

        data_decoder_curr = tf.concat([data_decoder[:, :1, :], last_y], -1)
        for i in range(target_len):
            out_tcn = self.tcn1(data_decoder_curr, training=False)
            out_attention = self.attention(out_tcn, data_encoder, training=False)
            out = self.normalization_layer(out_tcn + out_attention, training=False)

            for layer in self.output_layers:
                out = layer(out, training=False)

            for layer in self.additional_layers:
                out = layer(out, training=False)

            # Add prediction to the prediction tensor
            if predictions is None:
                predictions = out[:, -1, :]
            else:
                predictions = tf.concat([predictions, out[:, -1, :]], 1)
            if i == target_len - 1:
                continue

            last_predictions = tf.concat([last_y, out], axis=1)
            data_decoder_curr = tf.concat(
                [data_decoder[:, : i + 2, :], last_predictions], axis=-1
            )
        predictions = tf.reshape(predictions, shape=tf.shape(out))
        return predictions

    @tf.function
    def _call_none_regressive(self, inputs, training=None):
        data_encoder, data_decoder = inputs
        out_tcn = self.tcn1(data_decoder, training=training)
        out_attention = self.attention(out_tcn, data_encoder, training=training)
        out = self.normalization_layer(out_tcn + out_attention, training=training)

        for layer in self.output_layers:
            out = layer(out, training=training)

        for layer in self.additional_layers:
            out = layer(out, training=training)

        return out



