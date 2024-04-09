import tensorflow as tf
from tcn_sequence_models.tf_models.tcn import TCN
from tensorflow.keras.layers import MultiHeadAttention
from tcn_sequence_models.utils.NoisyTeacher import NoisyTeacherEnforcer
from tcn_sequence_models.utils.Modelextra import ImprovedSelfAttention, SqueezeExciteBlock
# from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

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
        number of layers is computed automatically based on the sequence lenghts
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
        self.noisy_teacher = NoisyTeacherEnforcer()

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
        
        # self.tcn2 = TCN(
        #     max_seq_len=30,
        #     num_stages=2,
        #     num_filters=self.num_filters,
        #     kernel_size=self.kernel_size,
        #     dilation_base=self.dilation_base,
        #     dropout_rate=self.dropout_rate,
        #     activation=self.activation,
        #     final_activation=self.activation,
        #     kernel_initializer=self.kernel_initializer,
        #     padding=self.padding,
        #     batch_norm=self.batch_norm,
        #     layer_norm=self.layer_norm,
        #     return_sequence=False,
        #     num_layers=num_layers,
        # )


        # Cross attention
        self.attention = MultiHeadAttention(
            key_dim=self.key_size,
            value_dim=self.value_size,
            num_heads=self.num_attention_heads,
            output_shape=self.num_filters,
        )

        self.attention_decoder = ImprovedSelfAttention(units=256)
        self.squeeze_excite = SqueezeExciteBlock()
        # Normalization layer after cross attention
        if self.layer_norm:
            self.normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        else:
            self.normalization_layer = tf.keras.layers.BatchNormalization()

        # layers for the final prediction stage
        self.output_layers = []

        for i, neurons in enumerate(self.output_neurons):
            layer_dense =tf.keras.layers.Dense(
                neurons,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
            )
            self.output_layers.append(layer_dense)
        # last output layer
        # self.output_layers.append(tf.keras.layers.Dense(units=1, activation='relu'))
            
        self.intermediate_layer = tf.keras.layers.Dense(480, activation ='relu')
        self.final_output_layer = tf.keras.layers.Dense(units=1, activation=None)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.relu_activation = tf.keras.layers.Activation('relu')
            
        # self.output_layers.append(tf.keras.layers.Dense(units=1, activation=None))
        # self.output_layers.append(tf.keras.layers.Dense(1))




    # @tf.function
    def call(self, inputs, training=True, current_epoch=None, total_epochs=None):
        if training:
            if self.autoregressive:
                return self._training_call_autoregressive(inputs, current_epoch, total_epochs)
            else:
                return self._call_none_regressive(inputs, training)
        else:
            if self.autoregressive:
                return self._inference_call_autoregressive(inputs)
            else:
                return self._call_none_regressive(inputs, training)

    # @tf.function
    def _training_call_autoregressive(self, inputs, current_epoch, total_epochs):
        data_encoder, data_decoder, y_shifted = inputs
        #y_shifted = tf.expand_dims(y_shifted, -1)
        # y_shifted = self.noisy_teacher.add_noise_to_tensor(y_shifted)
        teacher_forcing_ratio = max(0.5 - (0.5 * current_epoch / total_epochs), 0.0)
        # data_decoder = tf.concat([data_decoder, y_shifted], -1)
        if tf.random.uniform(shape=[]) < teacher_forcing_ratio:
            # Use true output value of the previous timestep
            data_decoder = tf.concat([data_decoder, y_shifted], -1)
        else:
            # Use model's predictions
            predictions = self._inference_call_autoregressive([data_encoder, data_decoder, y_shifted[:,:1,:]])
            data_decoder = tf.concat([data_decoder, predictions], -1)

        #data_decoder = tf.expand_dims(data_decoder,-1)
        out_tcn = self.tcn1(data_decoder, training=True)
        # print(out_tcn)
        out_attention = self.attention(out_tcn, data_encoder, training=True)
        out = self.normalization_layer(out_tcn + out_attention, training=True)
        # out = tf.concat([out_tcn, out_attention], -1)

        for layer in self.output_layers:
            out = layer(out, training=True)
        return out

    # @tf.function
    def _inference_call_autoregressive(self, inputs):
        data_encoder, data_decoder, last_y= inputs
        # last_y = tf.zeros_like(data_decoder[:, 0:1, -1:])
        target_len = data_decoder.shape[1]
        # last_y_reshaped = tf.reshape(last_y, [-1, 1, 1])
        last_y = tf.cast(last_y, tf.float32)
        # tf.print(y_shifted.shape)
        predictions = None
        # last_y = tf.ones_like(data_decoder[:, 0:1, -1:])
        # last_y = tf.cast(last_y, tf.float32)
        # tf.print(last_y)

        data_decoder_curr = tf.concat([data_decoder[:, :1, :], last_y], -1)
        for i in range(target_len):
            out_tcn = self.tcn1(data_decoder_curr, training=False)
            out_attention = self.attention(out_tcn, data_encoder, training=False)
            out = self.normalization_layer(out_tcn + out_attention, training=False)

            for layer in self.output_layers:
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
                [data_decoder[:, : i+2, :], last_predictions], axis=-1
            )
        predictions = tf.reshape(predictions,shape=(tf.shape(out)))
        # print(predictions)
        return predictions

    # @tf.function
    def _call_none_regressive(self, inputs, training=None):
        data_encoder, data_decoder = inputs
        # data_encoder = pad_sequences(data_encoder, maxlen=self.max_seq_len, padding='post', truncating='post')
        # data_encoder_copy = tf.identity(data_encoder)
        # tf.print("Shape of Data_encoder:", tf.shape(data_encoder_copy))

        # tf.print('Max seq len',self.max_seq_len)
        out_attention_decoder = self.attention_decoder(data_decoder)
        out_attention_decoder_relu = self.relu_activation(out_attention_decoder)

        # tf.print("Shape of Data_attention_decoder:", tf.shape(out_attention_decoder))

        out_tcn = self.tcn1(out_attention_decoder_relu, training=training)
        out_attention = self.attention(data_encoder,out_tcn, training=training)

        # tf.print("Shape of Data_encoder:", tf.shape(data_encoder))
        # tf.print("Shape of Data_decoder:", tf.shape(data_decoder))
        # tf.print("Shape of Out_tcn:", tf.shape(out_tcn))
        # tf.print("Shape of Out_attention:", tf.shape(out_attention))

        out_normalized = self.normalization_layer(out_attention + data_encoder, training=training)

        # tf.print("Shape of out_normalized:", tf.shape(out_normalized))

        out_squeeze_excite = self.squeeze_excite(out_normalized, training=training)
        out_squeeze_excite_relu = self.relu_activation(out_squeeze_excite)

        # tf.print("Shape of out_squeeze_excite_relu:", tf.shape(out_squeeze_excite_relu))
        # out_slice = out_squeeze_excite_relu[:,-1,:]

        out_flatten = self.flatten_layer(out_squeeze_excite_relu)


        # flattened_out = self.flatten_layer(out_squeeze_excite_relu, training = training)
        # flattened_out = tf.keras.layers.Reshape((-1,))(out_squeeze_excite_relu)
        # tf.print("Shape of flattened_out:", tf.shape(flattened_out))
        # tf.print(flattened_out.shape)
        # for layer in self.output_layers:
        #     out_layer = layer(flattened_out, training=training)

        # batch_size = tf.shape(out_squeeze_excite_relu)[0]
        # timestep_channel_shape = tf.reduce_prod(tf.shape(out_squeeze_excite_relu)[1:])
        # out_squeeze_excite_reshaped = tf.reshape(out_squeeze_excite_relu, [batch_size, timestep_channel_shape])
        # tf.print("Out_squeeze_reshaped:", tf.shape(out_squeeze_excite_reshaped))

        # if self.intermediate_layer.built is False:
        #     batch_size_1 = out_squeeze_excite_relu.shape[0].numpy()
        #     timestep_shape = out_squeeze_excite_relu.shape[1].numpy()
        #     channel_shape = out_squeeze_excite_relu.shape[2].numpy()
        #     timestep_channel_shape_1 = timestep_shape*channel_shape
        #     # timestep_channel_shape = tf.reduce_prod(tf.shape(out_squeeze_excite_relu)[1:]).numpy()
        #     intermediate_layer_input_shape = (batch_size_1, timestep_channel_shape_1)
        #     self.intermediate_layer.build(intermediate_layer_input_shape)

        #  Global Average Pooling 1D
        # out_squeeze_excite_avg = tf.keras.layers.GlobalAveragePooling1D()(out_squeeze_excite_relu)
        # tf.print("Shape of out_squeeze_excite_avg:", tf.shape(out_squeeze_excite_avg))
        # Global Max Pooling 1D
        # out_squeeze_excite_max = tf.keras.layers.GlobalMaxPooling1D()(out_squeeze_excite_relu)
        # tf.print("Shape of out_squeeze_excite_max:", tf.shape(out_squeeze_excite_max))
        # # Concatenate the outputs of global pooling
        # out_squeeze_excite_pooled = tf.keras.layers.concatenate([out_squeeze_excite_avg, out_squeeze_excite_max], axis=-1)
        # tf.print("Shape of out_squeeze_excite_pooled:", tf.shape(out_squeeze_excite_pooled))
        # out_squeeze_excite_reshaped = tf.reshape(out_squeeze_excite_reshaped, (-1, 480))
        # input_shape = out_squeeze_excite_reshaped.shape[1:]
        # dense_layer = tf.keras.layers.Dense(units=480, activation='relu', input_shape=input_shape)
        
        # out_tcn_2 = self.tcn2(out_normalized, training=training)
        # tf.print("Shape of Out_tcn_2:", tf.shape(out_tcn_2))
        # out_slice = out_tcn_2[:, -1, :]
        # tf.print("Shape of Out_slice:", tf.shape(out_squu))

        # tf.print("Shape of flattened out", tf.shape(out_flatten))

        out_layer = self.intermediate_layer(out_flatten, training = training)

        # tf.print("Shape of intermediate_layer", tf.shape(out_layer))

        # flattened_out = self.flatten_layer(out_layer)

        
        #  Global Average Pooling 1D
        # out_squeeze_excite_avg = tf.keras.layers.GlobalAveragePooling1D()(out_layer_reshaped)
        # tf.print("Shape of out_squeeze_excite_avg:", tf.shape(out_squeeze_excite_avg))
        # # Global Max Pooling 1D
        # out_squeeze_excite_max = tf.keras.layers.GlobalMaxPooling1D()(out_layer_reshaped)
        # tf.print("Shape of out_squeeze_excite_max:", tf.shape(out_squeeze_excite_max))
        # # Concatenate the outputs of global pooling
        # out_squeeze_excite_pooled = tf.keras.layers.concatenate([out_squeeze_excite_avg, out_squeeze_excite_max], axis=-1)
        # tf.print("Shape of out_squeeze_excite_pooled:", tf.shape(out_squeeze_excite_pooled))
        # out_squeeze_excite_pooled_1 = tf.squeeze(out_squeeze_excite_pooled, axis=1)
        # tf.print("Shape of out_squeeze_excite_pooled_1:", tf.shape(out_squeeze_excite_pooled_1))
        # flattened_outputs = []
        # @tf.function
        # def flatten_in_loop(batch):
        #     flattened_output = self.flatten_layer(batch)
        #     return flattened_output
        # for i in range(tf.shape(out_layer)[0]):
        #     # Flatten each batch individually
        #     flattened_output = flatten_in_loop(out_layer[i])
        #     flattened_outputs.append(flattened_output)

        

        # for layer in self.output_layers:
        #     out_layer = layer(out_squeeze_excite_reshaped, training = training
        # out_flatten = tf.keras.layers.concatenate(flattened_outputs, axis=0)


        # tf.print("Shape of flattened out", tf.shape(flattened_out))

        out_dense = self.final_output_layer(out_layer,  training = training)

        # tf.print("Shape of last_layer", tf.shape(out_dense))

        out = self.relu_activation(out_dense)

        # tf.print("Shape of output", tf.shape(out))

        out_expanded = tf.expand_dims(out, axis=-1)

        # tf.print("Shape of output_expanded", tf.shape(out_expanded))

        return out_expanded
