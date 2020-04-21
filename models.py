import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Add, Lambda, Input


class ConvBlock(K.layers.Layer):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv_2d_1 = Conv2D(32, 8, (4, 4), activation='relu')
        self.conv_2d_2 = Conv2D(64, 4, (2, 2), activation='relu')
        self.conv_2d_3 = Conv2D(64, 3, (1, 1), activation='relu')
        self.flatten = Flatten()

    def call(self, inputs, **kwargs):
        x = self.conv_2d_1(inputs)
        x = self.conv_2d_2(x)
        x = self.conv_2d_3(x)
        x = self.flatten(x)
        return x

    def get_config(self):
        return {'units': self.units}


class DuelingCONVQ(K.models.Model):
    def __init__(self, input_shape, output_shape):
        super(DuelingCONVQ, self).__init__()

        self.model_input = Input(input_shape)
        self.convolution_block = ConvBlock()

        self.action_dense_1 = Dense(512, activation='relu')
        self.action_dense_2 = Dense(output_shape)
        self.action_normalize_1 = Lambda(lambda a: a[:, :] - tf.reduce_mean(a[:, :], axis=-1, keepdims=True),
                                         output_shape=(output_shape,))
        self.value_dense_1 = Dense(512, activation='relu')
        self.value_dense_2 = Dense(1)
        self.value_add_prep = Lambda(lambda s: tf.expand_dims(s[:, 0], -1),
                                     output_shape=(output_shape,))
        self.stream_combine = Add()

    def call(self, inputs, training=None, mask=None):
        x = self.model_input(inputs)
        x = self.convolution_block(x)

        a = self.action_dense_1(x)
        a = self.action_dense_2(a)
        a = self.action_normalize_1(a)

        v = self.value_dense_1(x)
        v = self.value_dense_2(v)
        v = self.value_add_prep(v)

        x = self.stream_combine([v, a])
        return x


class SimpleCONVQ(K.models.Model):
    def __init__(self, input_shape, output_shape):
        super(SimpleCONVQ, self).__init__()

        self.convolution_block = ConvBlock()

        self.dense_1 = Dense(512, activation='relu')
        self.dense_2 = Dense(output_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.convolution_block(inputs)

        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
