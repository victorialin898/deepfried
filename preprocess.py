import numpy as np
import tensorflow as tf
from scipy import signal, io, interpolate
import glob
import librosa

patch_len = 6000    # Length per time series sample
scale = 2           # Downsampling factor for our corruption


"""
    Extracts data from .wav files from the VCTK dataset, and returns them as audio time series of patch_len

    @param batch_size : Number of patches per batch
    @param train_size : Number of desired patches in training set
    @param test_size : Number of desired patches in the test set
    @param VCTK : Kept from when we were planning on working with the piano set, for now is not needed
    @returns : an iterator each for the train set and the test set
"""
def get_dataset_iterator(batch_size=128, train_size=18000, test_size=5000, VCTK=True):

    """
        Helper function to extract patches from specified audio file

        @param file_path : path to a wav file, string
        @returns : tensor of audio patches generated from audio, shape=(len(audio) - 6000, 6000)
        since 6000 is our default patch_len
    """
    # @tf.function
    def extract_patches(file_path):
        audio, sr = tf.audio.decode_wav(tf.io.read_file(file_path))
        audio = tf.squeeze(audio)

        # add dimensions so that we can treat the audio file like an image
        audio = tf.expand_dims(tf.expand_dims(tf.expand_dims(audio, -1), 0), 0)

        # borrow the extract_patches function to create patches. you can think of this
        # as convolving on a 1d image to create patches
        patches = tf.image.extract_patches(images=audio, sizes=[1, 1, patch_len, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        patches = tf.squeeze(patches)
        return tf.data.Dataset.from_tensor_slices(patches)

    # NOTE: you must run download.py first
    dir_path = './data/VCTK-Corpus/wav48/**/*.wav'
    dataset = tf.data.Dataset.list_files(dir_path)

    # NOTE: To parallelize this, we have to use dataset.map(), and then flatten after
    dataset = dataset.flat_map(map_func=extract_patches)

    # Repeat the dataset for 4 epochs
    dataset = dataset.repeat(4)

    # Need to shuffle again after creating patches
    dataset = dataset.shuffle(buffer_size=250000)

    # Split dataset in train and test
    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = dataset.skip(train_size).take(test_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)

    return train_dataset, test_dataset

"""
    Retrieves time series representations for demo files located in /demo directory
    @returns : lists of filepaths, files, and sampling rates
"""
def get_demos():
    wav_filepaths = glob.glob("./demo/*.wav")
    wav_files, sampling_rates = zip(*[librosa.load(f) for f in wav_filepaths])
    wav_files = list(wav_files)
    sampling_rates = list(sampling_rates)
    for i in range(len(wav_files)):
        wav_files[i] = wav_files[i][:patch_len * 10]
    return wav_filepaths, wav_files, sampling_rates


"""
    Uses librosa's library to retrieve the short-time Fourier transform of a given signal. This
    is used in the log-spectral calculation in model.py
    @returns : short time fourier transform
"""
def get_stft(signal, n_fft):
    return librosa.stft(signal, n_fft)

"""
    Calls librosa's amplitude_to_db on a given signal. Used in SNR calculations in model.py
    @returns : tensor with db units instead of amplitude
"""
def amplitude_to_decibel(signal):
    return librosa.core.amplitude_to_db(signal)
