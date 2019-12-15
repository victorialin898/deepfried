import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import get_dataset_iterator, get_demos
from model import Model
import sys
import time
import soundfile as sf
from scipy.signal import decimate
from scipy import interpolate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

patch_len = 6000
scale = 2

"""
    Creates corrupted, low resolution versions of a batch of time series
    @param batch : a tensor of (batch_size, 6000)
    @returns : Tensor of (batch_size, 6000)
"""
def corrupt_batch(batch):
    corrupted = decimate(batch, scale, axis=-2)  # Applies an order 8 Chebyshev type I low pass filter
    corrupted = corrupted.flatten()
    new_length = len(corrupted) * scale

    # Cubic interpolation to bring the time series to original size (decimating will reduce by factor of scale)
    f = interpolate.splrep(np.arange(new_length, step=scale), corrupted)    
    upscaled = tf.reshape(interpolate.splev(np.arange(new_length), f), (batch.shape[0], -1, 1))
    return upscaled

"""
    Trains model in batches on corrupted and original audio files.
    @param model : Model to use
    @param train_data_iterator : dataset iterator
"""
def train(model, train_data_iterator):
    for iteration, batch in enumerate(train_data_iterator):
        batch = tf.expand_dims(batch, -1)
        batch_corrupted = corrupt_batch(batch)
        assert(batch_corrupted.shape == batch.shape)

        with tf.GradientTape() as tape:
            batch_sharpened = model.call(batch_corrupted)
            loss = model.loss_function(batch_sharpened, batch)
            accuracy = model.snr_function(batch_sharpened, batch)
            lsd = model.lsd(batch_sharpened, batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if iteration % 1  == 0:
            print("BATCH %d LOSS %f SNR %f LSD %f"%(iteration, loss, accuracy, lsd))

"""
    Tests model in batches on corrupted and original audio files.
    @param model : Model to use
    @param train_data_iterator : dataset iterator
    @returns : average loss over input test data, sharpened audio files
"""
def test(model, test_data_iterator):
    losses = []
    accuracies = []
    lsds = []

    for iteration, batch in enumerate(test_data_iterator):
        batch = tf.expand_dims(batch, -1)
        batch_corrupted = corrupt_batch(batch)
        batch_sharpened = model.call(batch_corrupted)
        loss = model.loss_function(batch_sharpened, batch)
        accuracy = model.snr_function(batch_sharpened, batch)
        losses.append(loss)
        accuracies.append(accuracy)
        lsds.append(model.lsd(batch_sharpened, batch))

    return tf.reduce_mean(losses), tf.reduce_mean(accuracies), tf.reduce_mean(lsds)


"""
    Creates demos from files placed in the /demo directory. Generates corrupted and 
    sharpened versions and saves in /demo/output.
    @param model : Model to use
"""
def test_demo(model):
    wav_filepaths, wav_files, sampling_rates = get_demos()
    output_dir = 'demo/output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(wav_files)):

        batch = tf.expand_dims(tf.expand_dims(wav_files[i], 0), -1)
        batch_corrupted = corrupt_batch(batch)
        batch_sharpened = model.call(batch_corrupted)
       
        sf.write(file=os.path.join(output_dir, str(i + 1)+'_sharpened.wav'), data=tf.squeeze(batch_sharpened), samplerate=sampling_rates[i])
        sf.write(file=os.path.join(output_dir, str(i + 1)+'_corrupted.wav'), data=tf.squeeze(batch_corrupted), samplerate=sampling_rates[i])
        sf.write(file=os.path.join(output_dir, str(i + 1)+'_original.wav'), data=tf.squeeze(batch), samplerate=sampling_rates[i])
        

def main():
    model = Model()

    print("Running preprocessing...")
    train_data_iterator, test_data_iterator = get_dataset_iterator(batch_size=model.batch_size, VCTK=True)
    print("Finished reprocessing.")

    # print("Beginning training...")
    # for epoch in range(model.epochs):
    #     print("EPOCH %d" % epoch)
    #     train(model, train_data_iterator)
    # print("Training complete.")

    print("Beginning testing...")
    loss, accuracy, lsds = test(model, test_data_iterator)
    print("Average loss: " + str(loss.numpy()))
    print("Average accuracy (SNR): " + str(accuracy.numpy()))
    print("Average accuracy (LSD): " + str(lsds.numpy()))

    test_demo(model)

if __name__ == '__main__':
   main()
