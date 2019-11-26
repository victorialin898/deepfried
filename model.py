import tensorflow as tf
import numpy as np

# Downsampling blocks for bottleneck architecture
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_conv_1 = tf.keras.layers.Conv2D(10, 3, (2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.encoder_conv_2 = tf.keras.layers.Conv2D(10, 3, (2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.encoder_conv_3 = tf.keras.layers.Conv2D(10, 3, (2,2), padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))

        self.dropout_rate = 0.001
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
       
        # TODO: change stride, kernel no, num filters to match our preprocessed output

    @tf.function
    def call(self, samples):
        # TODO: we need subpixel shuffling layer here ! each layer has a dimshuffle

        output = tf.nn.relu(self.dropout(self.encoder_conv_1(samples)))
        output = tf.nn.relu(self.dropout(self.encoder_conv_2(output)))
        output = tf.nn.relu(self.dropout(self.encoder_conv_3(output)))
        return output
        

# Upsampling blocks for bottleneck architecture
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()        
        self.decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(10, 3, (2,2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(10, 3, (2,2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
        self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(1, 3, (2,2), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))

        # TODO: leakurelu vs relu? which is better

    @tf.function
    def call(self, encoder_output):
        return self.decoder_deconv_3(self.decoder_deconv_2(self.decoder_deconv_1(encoder_output)))


# Main model: performs autoeocoding and the subpixel shuffling
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.learning_rate = 0.01
        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    @tf.function
    def call(self, samples):
       return self.decoder.call(self.encoder.call(samples))
    
    @tf.function
    def loss_function(self, encoded, originals):
      encoded = tf.dtypes.cast(encoded, tf.float32)
      originals = tf.dtypes.cast(originals, tf.float32)
      return tf.reduce_sum(tf.math.square(originals - tf.reshape(encoded, originals.shape)))