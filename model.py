import tensorflow as tf
import numpy as np
import scipy
from tensorflow.keras.layers import LeakyReLU, Conv1D, BatchNormalization, Conv2DTranspose, Dropout

# Downsampling blocks for bottleneck architecture
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        # I believe these should be Conv1D, because the data is a time series. Right?
        # TODO: decide how many Conv2D layers we want. The paper calls this hyperparameter B, so currently B=3 for us.
        #       The paper uses B=4. Note that this B should be the same in the Encoder and the Decoder

        # Given encoder_conv_b
        #   - number of filters (i.e. output depth) = max(2^(6+b), 512)
        #   - length of filters (remember that it's Conv1D) = min(2^(7-b) + 1, 9)
        #   - stride = 2


        # In the paper, the upsampling block (decoder) consists only of a convolution and ReLu
        self.encoder_conv_1 = Conv1D(filters=512, kernel_size=9, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.encoder_conv_2 = Conv1D(filters=512, kernel_size=9, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.encoder_conv_3 = Conv1D(filters=512, kernel_size=9, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))

        # keeping alpha at 0.2 for now
        self.relu = LeakyReLU(alpha=0.2)


        self.batch_norm = BatchNormalization()
        # TODO: change stride, kernel no, num filters to match our preprocessed output


    @tf.function
    def call(self, samples):
        # Here's a helpful article for understanding subpixel stuff (https://medium.com/@hirotoschwert/introduction-to-deep-super-resolution-c052d84ce8cf)
        # Added a bunch of leaky relus!

        output = self.batch_norm(self.relu(self.encoder_conv_1(samples))))
        output = self.batch_norm(self.relu(self.encoder_conv_2(output))))
        output = self.batch_norm(self.relu(self.encoder_conv_3(output))))
        return output


# Upsampling blocks for bottleneck architecture
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        # TODO: Figure out what layer to use for upsampling since keras has no Conv1DTranspose
        #   - keras.layers.UpSampling1D (not learnable, see https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker)
        #   - create our own (https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras)

        # Given encoder_conv_b
        #   - number of filters (i.e. output depth) = max(2^(7+B-b+1), 512)
        #   - length of filters (remember that it's Conv1D) = min(2^(7-(B-b+1)) + 1, 9)
        #   - stride = ???????
        self.decoder_deconv_1 = Conv2DTranspose(filters=1024, kernel_size=9, strides=2, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_deconv_2 = Conv2DTranspose(filters=512, kernel_size=9, strides=2, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_deconv_3 = Conv2DTranspose(filters=512, kernel_size=9, strides=2, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(0.5)

    @tf.function
    def call(self, encoder_output):

        # TODO: we need subpixel shuffling layer here ! each layer has a dimshuffle
        output = self.dropout(self.batch_norm(self.decoder_deconv_1(encoder_output)))
        output = self.dropout(self.batch_norm(self.decoder_deconv_2(output)))
        output = self.dropout(self.batch_norm(self.decoder_deconv_3(output)))
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

    @tf.function
    def call(self, samples):
        return self.decoder.call(self.encoder.call(samples))

    @tf.function
    def loss_function(self, encoded, originals):
        encoded = tf.dtypes.cast(encoded, tf.float32)
        originals = tf.dtypes.cast(originals, tf.float32)
        return (1/len(encoded)) * tf.math.sqrt(tf.reduce_sum(tf.math.square(tf.norm(originals - tf.reshape(encoded, originals.shape)))))


    # Re-implementation of scipy.stats.signaltonoise which is deprecated :(
    def accuracy_function(self, encoded, originals):
        # Using equation for "SNR Calculation - Complicated" from sciencing.com
        signal = tf.reduce_mean(originals, axis=[1,2])
        noise = tf.math.reduce_std(originals - encoded, axis=[1,2])
        snr = 20 * tf.math.log(signal / noise) / tf.math.log(10.0)
        return snr


# TODO: Reverse data preprocessing to turn the output of our model into a listenable .wav file?
