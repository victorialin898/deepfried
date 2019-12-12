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

patch_len = 6000
scale = 2

"""
    @param batch : a tensor of (batch_size, 6000)
"""
def corrupt_batch(batch):
    # raise Exception(batch.shape)
    corrupted = decimate(batch, scale, axis=-2)
    #f = interpolate.interp1d(np.arange(corrupted.shape[1]), corrupted, kind='cubic', axis=-2)
    corrupted = corrupted.flatten()
    new_length = len(corrupted) * scale
    f = interpolate.splrep(np.arange(new_length, step=scale), corrupted)
    upscaled = tf.reshape(interpolate.splev(np.arange(new_length), f), (batch.shape[0], -1, 1))
    #upscaled = f(np.arange(0.0, corrupted.shape[1] - 1, (corrupted.shape[1] - 1) /batch.shape[1]))
    return upscaled

"""
    Trains model in batches on corrupted and original audio files.
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
        print("BATCH %d LOSS %f SNR %f LSD %f"%(iteration, loss, accuracy, lsd))

"""
    Tests model in batches on corrupted and original audio files.

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

def test_demo(model):
    wav_filepaths, wav_files, sampling_rates = get_demos()
    output_dir = './demo/output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(wav_files)):

        batch = tf.expand_dims(tf.expand_dims(wav_files[i], 0), -1)
        print("starting to corrupt")
        batch_corrupted = corrupt_batch(batch)
        print("starting to sharpen")
        batch_sharpened = model.call(batch_corrupted)
        print("Done with batch")
        # raise Exception(os.path.join(output_dir, os.path.basename(wav_filepaths[i])))

        sf.write(file=os.path.join(output_dir, str(i + 1)+'_sharpened.wav'), data=batch_sharpened, samplerate=sampling_rates[i])
    # raise Exception(wav_filepaths, wav_files, sampling_rates)
    # for iteration, b in enumerate(demos_data_iterator):
    #     batch, sr = b[0], b[1]
    #     print("TEST DMEOMEOMDOEMDEMODMOEDMOEMOE")
    #     # print(batch.shape)
    #     print(sr.numpy())
    #     print(sr)
    #     batch = tf.expand_dims(batch, -1)
    #     print("starting to corrupt")
    #     batch_corrupted = corrupt_batch(batch)
    #     print("starting to sharpen")
    #     batch_sharpened = model.call(batch_corrupted)

        #sf.write(file=os.path.join('output/', str(iteration + 1)+'_hr.wav'), data=batch[0], samplerate=sr)
        #sf.write(file=os.path.join('output/', str(iteration + 1)+'_lr.wav'), data=batch_corrupted[0], samplerate=sr)
        #sf.write(file=os.path.join('output/', str(iteration + 1)+'_pr.wav'), data=batch_sharpened[0], samplerate=sr)
#
def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"VCTK","PIANO"}:
        print("USAGE: python assignment.py <Data Set>")
        print("<Data Set>: [VCTK/PIANO]")
        exit()

    model = Model()

    print("Running preprocessing...")
    if sys.argv[1] == "VCTK":
        train_data_iterator, test_data_iterator = get_dataset_iterator(batch_size=model.batch_size, VCTK=True)
    elif sys.argv[1] == "PIANO":
        train_data_iterator, test_data_iterator = get_dataset_iterator(batch_size=model.batch_size, VCTK=False)
    print("Finished reprocessing.")

    test_demo(model)
    raise Exception("DEMO TESTED")


    print("Beginning training...")
    for _ in range(model.epochs):
        train(model, train_data_iterator)
    print("Training complete.")

    #print("Beginning testing...")
    #loss, accuracy, lsds = test(model, test_data_iterator)
    #print("Average loss: " + str(loss.numpy()))
    #print("Average accuracy (SNR): " + str(accuracy.numpy()))
    #print("Average accuracy (LSD): " + str(lsds.numpy()))



if __name__ == '__main__':
   main()
