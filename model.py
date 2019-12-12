import tensorflow as tf
import os
import librosa
import numpy as np
import scipy
from tensorflow.keras.layers import LeakyReLU, Conv1D, Conv2D, BatchNormalization, Conv2DTranspose, Dropout
from preprocess import get_stft, amplitude_to_decibel
import matplotlib
from matplotlib import pyplot as plt

"""
    Implements the 1D subpizel shuffle

    @param input: tensor of shape batch_size x filters x dim
    @returns : the tensor reshaped to be batch_size x filters / 2 x dim*2

"""
def subpixel_shuffle(input):
    (batch_size, dim, filters) = input.shape
    input = tf.reshape(input, [batch_size, dim, filters // 2, 2])
    input = tf.transpose(input, [0, 1, 3, 2])
    input = tf.reshape(input, [batch_size, dim * 2, filters // 2])
    return input


# Downsampling blocks for bottleneck architecture
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        # I believe these should be Conv1D, because the data is a time series. Right?
        # TODO: decide how many Conv2D layers we want. The paper calls this hyperparameter B, so currently B=4 for us.
        #       The paper  claims to use B=4, but their code on github uses B=7.
        #       Note that this B should be the same in the Encoder and the Decoder

        # Given encoder_conv_b
        #   - number of filters (i.e. output depth) = min(2^(6+b), 512)
        #   - length of filters (remember that it's Conv1D) = max(2^(7-b) + 1, 9)
        #   - stride = 2

        # In the paper, the upsampling block (decoder) consists only of a convolution and ReLu
        self.encoder_conv_1 = Conv1D(filters=128, kernel_size=65, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.encoder_conv_2 = Conv1D(filters=256, kernel_size=33, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.encoder_conv_3 = Conv1D(filters=512, kernel_size=17, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))

        # keeping alpha at 0.2 for now
        # self.relu = LeakyReLU(alpha=0.2)
        # Since dropout and relu all commute, we can use the builtin activation to Conv1d layers i
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()

    """
        Function that returns a list of all the outputs of each encoder block, not just the last encoder block

        @param samples : a batch_size x sample_length tensor of audio samples
        @returns : a list of length B, where the i-th element is the tensor output of encoder layer i
    """
    @tf.function
    def call(self, samples):
        # Here's a helpful article for understanding subpixel stuff (https://medium.com/@hirotoschwert/introduction-to-deep-super-resolution-c052d84ce8cf)

        output_1 = self.batch_norm_1(self.encoder_conv_1(samples))
        output_2 = self.batch_norm_2(self.encoder_conv_2(output_1))
        output_3 = self.batch_norm_3(self.encoder_conv_3(output_2))
        return [output_1, output_2, output_3]


# Upsampling blocks for bottleneck architecture
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        # TODO: Figure out what layer to use for upsampling since keras has no Conv1DTranspose
        #   - keras.layers.UpSampling1D (not learnable, see https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker)
        #   - create our own (https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras)

        # Given encoder_conv_b
        #   - number of filters (i.e. output depth) = min(2^(7+B-b+1), 512)
        #   - length of filters (remember that it's Conv1D) = max(2^(7-(B-b+1)) + 1, 9)
        #   - stride = 1 so that the feature dimension is not reduced by Conv1d
        self.decoder_conv_1 = Conv1D(filters=1024, kernel_size=9, strides=1, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_conv_2 = Conv1D(filters=512, kernel_size=17, strides=1, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_conv_3 = Conv1D(filters=256, kernel_size=33, strides=1, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))

        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.batch_norm_3 = BatchNormalization()
        self.dropout = Dropout(0.5)

    """
        Encoder output should be a list of all the encoder outputs,
        since we need these to stack after the subpixel shuffling layer

        @param bottleneck_output : tensor returned by the bottleneck layer
        @param encoder_output : list of tensors returned by each encoder later.
                                Has length B, where B is the number of layers

    """
    @tf.function
    def call(self, bottleneck_output, encoder_output):
        output = self.dropout(self.batch_norm_1(self.decoder_conv_1(bottleneck_output)))
        output = subpixel_shuffle(output)

        output = tf.concat([output, encoder_output[2]], axis=-1)

        output = self.dropout(self.batch_norm_2(self.decoder_conv_2(output)))
        output = subpixel_shuffle(output)
        output = tf.concat([output, encoder_output[1]], axis=-1)

        output = self.dropout(self.batch_norm_3(self.decoder_conv_3(output)))
        output = subpixel_shuffle(output)
        output = tf.concat([output, encoder_output[0]], axis=-1)

        return output


# Main model: performs autoeocoding and the subpixel shuffling
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.learning_rate = 10e-4   # Paper uses 10e-4
        self.batch_size = 128        # Batch size=128 vs patch size=6000
        self.epochs = 1              # Paper uses 400
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # We need the following two layers, but they don't neatly fall into either the encoder or decoder structure
        self.bottleneck_conv = Conv1D(filters=512, kernel_size=9, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.final_conv = Conv1D(filters=2, kernel_size=9, strides=1, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.dropout = Dropout(0.5)

    @tf.function
    def call(self, samples):
        encoder_out = self.encoder.call(samples)
        bottleneck_out = self.dropout(self.bottleneck_conv(encoder_out[-1]))
        decoder_out = self.decoder.call(bottleneck_out, encoder_out)
        final_out = tf.cast(samples, tf.float32) + subpixel_shuffle(self.final_conv(decoder_out))
        return final_out

    @tf.function
    def loss_function(self, encoded, originals):
        encoded = tf.dtypes.cast(encoded, tf.float32)
        originals = tf.dtypes.cast(originals, tf.float32)
        return (1/len(encoded)) * tf.math.sqrt(tf.reduce_sum(tf.math.square(tf.norm(originals - tf.reshape(encoded, originals.shape)))))


    # Re-implementation of scipy.stats.signaltonoise which is deprecated :(
    def snr_function(self, encoded, originals):
        # Using equation for "SNR Calculation - Complicated" from sciencing.com

        # CONVERTING TO DECIBELS FIRST:
        encoded, originals = amplitude_to_decibel(tf.squeeze(encoded)), amplitude_to_decibel(tf.squeeze(originals))
        mean = tf.square(scipy.linalg.norm(originals, axis=1))
        std = tf.square(scipy.linalg.norm(originals-encoded, axis=1))
        snrs = np.where(std==0, 0, 10* tf.math.log(mean/std))
        return tf.reduce_mean(snrs)

    def lsd(self, encoded, originals, n_fft=2048, step=10):
        S_y = tf.signal.stft(np.array(tf.squeeze(originals)), n_fft, step)
        S_x = tf.signal.stft(np.array(tf.squeeze(encoded)), n_fft, step)
        logspec_y = tf.square(np.log1p(tf.abs(S_y)))
        logspec_x = tf.square(np.log1p(tf.abs(S_x)))
        squared_diff = tf.square(logspec_y - logspec_x)

        # Use for non display variable
        # if os.environ.get('DISPLAY','') == '':
        #     print('no display found. Using non-interactive Agg backend')
        #     matplotlib.use('Agg')

        # plt.imshow(logspec_y.numpy().T, aspect=10)
        # plt.tight_layout()
        # plt.savefig('original_spectrum.png')
        #
        # plt.imshow(logspec_x.numpy().T, aspect=10)
        # plt.tight_layout()
        # plt.savefig('predicted_spectrum.png')

        return tf.reduce_mean(tf.math.sqrt(squared_diff))

# TODO: Reverse data preprocessing to turn the output of our model into a listenable .wav file?
