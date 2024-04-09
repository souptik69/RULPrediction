import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Dense,  GlobalAveragePooling1D, Reshape, Multiply, Activation


class ImprovedSelfAttention(Layer):
    def __init__(self, units=64):
        super(ImprovedSelfAttention, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.conv_q = Conv1D(self.units, kernel_size=1, strides=1, padding='valid')
        self.conv_k = Conv1D(self.units, kernel_size=1, strides=1, padding='valid')
        self.conv_v = Conv1D(self.units, kernel_size=1, strides=1, padding='valid')
        self.wa = Dense(input_shape[-1])  # Assuming input_shape[-1] is the channel number
        super(ImprovedSelfAttention, self).build(input_shape)
    # @tf.function
    def call(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        dk = tf.cast(self.units, dtype=tf.float32)
        k_transposed = tf.transpose(k, perm=[0, 2, 1])
        attention_weights = tf.nn.softmax(tf.matmul(q, k_transposed) / tf.sqrt(dk), axis=-1)
        a = tf.matmul(attention_weights, v)
        # a = attention_weights
        output = x + self.wa(a)

        return output
    
class SqueezeExciteBlock(Layer):
    def __init__(self, ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]

        self.squeeze = GlobalAveragePooling1D()
        self.excitation1 = Dense(channels // self.ratio, activation='relu')
        self.excitation2 = Dense(channels, activation='sigmoid')
        self.reshape = Reshape((1, channels))
        super(SqueezeExciteBlock, self).build(input_shape)
    # @tf.function
    def call(self, x):
        squeeze = self.squeeze(x)
        excitation = self.excitation1(squeeze)
        excitation = self.excitation2(excitation)
        excitation = self.reshape(excitation)
        # channels = tf.shape(x)[-1]
        # excitation = tf.reshape(excitation, [-1, 1, channels])
        scaled_input = Multiply()([x, excitation])
        return scaled_input
