import numpy as np
import tensorflow as tf
from scipy import signal, io, interpolate
import glob
import librosa

patch_len = 6000
demo_len =  48000
scale = 2
"""
    TODO:
        - Using the parameter VCTK, distinguish between which audio files we want to load with librosa.
"""

"""
    Extracts data from .wav files, either VCTK/PIANO, and returns them as audio time series
    @param VCTK : whether to to load and return the voice data. If false, loads the piano data.
    @returns : an iterator each for the train set and the test set
"""
def get_dataset_iterator(batch_size=128, train_size=8000, test_size=2000, VCTK=True):
    # TODO: figure out exactly how many training and testing samples we want
    """
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

    # NOTE: if we want to parallelize this, we have to use dataset.map(), and then flatten after
    dataset = dataset.flat_map(map_func=extract_patches)

    # need to shuffle again after creating patches
    dataset = dataset.shuffle(buffer_size=250000)

    # split dataset in train and test
    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = dataset.skip(train_size).take(test_size)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)

    return train_dataset, test_dataset

"""
gets a few demos
"""
def get_demos(num_demo=1, train_size=8000):
    wav_filepaths = glob.glob("./demo/*.wav")
    wav_files, sampling_rates = zip(*[librosa.load(f) for f in wav_filepaths])
    wav_files = list(wav_files)
    sampling_rates = list(sampling_rates)
    for i in range(len(wav_files)):
        print(wav_files[i].shape)
        # raise Exception(wav.shape, len(wav))
        # if wav_files[i].shape[0] % 2 != 0:
        wav_files[i] = wav_files[i][:patch_len * 10]
        print(wav_files[i].shape)
    return wav_filepaths, wav_files, sampling_rates


def get_stft(signal, n_fft):
    return librosa.stft(signal, n_fft)

def amplitude_to_decibel(signal):
    return librosa.core.amplitude_to_db(signal)

# ----- old code left below for reference -----
    # print(dataset)
    # get list of files
    # load files into dataset
    # flat_map a func that takes a audio file outputs a dataset of tuples (hr patch, lr patch)
    # split dataset into train and test dataset iterators
    # return both
    #
    # audio_type = ['hr', 'lr', 'pr', 'sp']
    #
    # """
    #     Helper function for get_data. Using glob, recursively finds all files that
    #     match on the input Unix-like regex
    # """
    # def search_by_wav_file_type(wav, filepath):
    #     return glob.glob(filepath%(wav), recursive=True)
    #
    # if VCTK:
    #     filepath = "./data/VCTK-Corpus/wav48/**/*.wav"  # VCTK folder holds 2 sets of data: multispeaker (msp) and single speaker (sp1)
    # else:
    #     filepath = "./data/piano/**/*.%s.wav"
    #
    # # raise Exception([f for f in glob.glob(filepath)])
    # #load all file_paths
    # #librosa loads all of them, saving them as Tuple(<Audio Time Series[np.array], Sampling Rate [float])
    # # files = {aud: np.array(list(map(librosa.load, search_by_wav_file_type(aud, filepath)))) for aud in audio_type}
    # # raise Exception(files)
    # # Retrieve high resolution ones which we will corrupt: array of variable length arrays size (data points ,)
    # # originals = np.squeeze(files['hr'][:,:1])
    # # demo_sr = np.squeeze(files['hr'][:,1:][:20])
    # originals = np.array([librosa.load(f)[0] for f in glob.glob(filepath)])
    # raise Exception(originals)
    #
    # # Cubic interpolation so they are all the same size, discretize train and test samples
    # patch_length = 6000
    # upscaled = np.array([interpolate.interp1d(range(len(x)), x, kind='cubic')(np.arange(patch_length) * len(x)/patch_length) for x in originals])
    #
    # # Corrupting process: scipy.signal.decimate uses chebyshev of order 8
    # downsample_factor = 2
    # corrupted = signal.decimate(upscaled, downsample_factor, axis=1)
    #
    # # Paper implements an 88%-6%-6% split on train, test, validation, we will use 90-10 for now
    # cutoff_index = int(len(originals)*0.90)
    # # raise Exception(len(originals))
    #
    # train_corrupted = corrupted[:cutoff_index]
    # train_originals = originals[cutoff_index:]
    # test_corrupted = corrupted[:cutoff_index]
    # test_originals =  originals[cutoff_index:]
    #
    #
    # print("Generated %d samples of HR patch length: %d, LR patch length = %d, downsampling factor = %d" %(upscaled.shape[0], upscaled.shape[1], corrupted.shape[1], downsample_factor))
    #
    # # Figure out validation
    # return train_corrupted, train_originals, test_corrupted, test_originals, demo_sr
