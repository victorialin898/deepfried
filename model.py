import tensorflow as tf
import numpy as np
import scipy

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
        self.encoder_conv_1 = tf.keras.layers.Conv1D(filters=512, kernel_size=9, strides=2, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        # Add keras.layers.BatchNormalization()
        self.encoder_conv_2 = tf.keras.layers.Conv1D(filters=512, kernel_size=9, strides=2, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        # Add keras.layers.BatchNormalization()
        self.encoder_conv_3 = tf.keras.layers.Conv1D(filters=512, kernel_size=9, strides=2, padding='same', activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        # Add keras.layers.BatchNormalization()

        self.dropout_rate = 0.001
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
       
        # TODO: change stride, kernel no, num filters to match our preprocessed output

    @tf.function
    def call(self, samples):
        # TODO: we need subpixel shuffling layer here ! each layer has a dimshuffle
        # Here's a helpful article for understanding subpixel stuff (https://medium.com/@hirotoschwert/introduction-to-deep-super-resolution-c052d84ce8cf)

        output = self.dropout(self.encoder_conv_1(samples))
        output = self.dropout(self.encoder_conv_2(output))
        output = self.dropout(self.encoder_conv_3(output))
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
        self.decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=9, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=9, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=9, strides=2, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))

        # TODO: leakurelu vs relu? which is better

    @tf.function
    def call(self, encoder_output):
        # Why does the decoder have no dropout?
        return self.decoder_deconv_3(self.decoder_deconv_2(self.decoder_deconv_1(encoder_output)))


# Main model: performs autoeocoding and the subpixel shuffling
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.learning_rate = 0.01   # Paper uses .0001
        self.batch_size = 100       # Batch size vs patch size?
        self.epochs = 1             # Paper uses 400
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    @tf.function
    def call(self, samples):
       return self.decoder.call(self.encoder.call(samples))
    
    @tf.function
    def loss_function(self, encoded, originals):
      encoded = tf.dtypes.cast(encoded, tf.float32)
      originals = tf.dtypes.cast(originals, tf.float32)
      return (1/len(encoded)) * tf.math.sqrt(tf.reduce_sum(tf.math.square(tf.norm(originals - tf.reshape(encoded, originals.shape)))))


    def accuracy_function(self, encoded):
        # SNR: signal to noise ratio
        return scipy.stats.signaltonoise(encoded, axis=0, ddof=0)

# TODO: Reverse data preprocessing to turn the output of our model into a listenable .wav file?