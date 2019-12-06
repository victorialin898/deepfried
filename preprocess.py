import numpy as np
import tensorflow as tf
from scipy import signal, io, interpolate
import glob
import librosa

"""
    TODO:
        - Using the parameter VCTK, distinguish between which audio files we want to load with librosa.
"""

"""
    Extracts data from .wav files, either VCTK/PIANO, and returns them as audio time serie
    @param VCTK : whether to to load and return the voice data. If false, loads the piano data.
    @returns : a tuple of the form (train_corrupted, train_originals, test_corrupted, test_originals, demo_sr)
"""
def get_data(VCTK=False):

    audio_type = ['hr', 'lr', 'pr', 'sp']

    """
        Helper function for get_data. Using glob, recursively finds all files that
        match on the input Unix-like regex
    """
    def search_by_wav_file_type(wav, filepath):
        return glob.glob(filepath%(wav), recursive=True)

    if VCTK:
        filepath = "./data/vctk/**/*.%s.wav"  # VCTK folder holds 2 sets of data: multispeaker (msp) and single speaker (sp1)
    else:
        filepath = "./data/piano/**/*.%s.wav"

    #load all file_paths
    #librosa loads all of them, saving them as Tuple(<Audio Time Series[np.array], Sampling Rate [float])
    files = {aud: np.array(list(map(librosa.load, search_by_wav_file_type(aud, filepath)))) for aud in audio_type}

    # Retrieve high resolution ones which we will corrupt: array of variable length arrays size (data points ,)
    originals = np.squeeze(files['hr'][:,:1])
    demo_sr = np.squeeze(files['hr'][:,1:][:20])


    # Cubic interpolation so they are all the same size, discretize train and test samples
    patch_length = 6000
    upscaled = np.array([interpolate.interp1d(range(len(x)), x, kind='cubic')(np.arange(patch_length) * len(x)/patch_length) for x in originals])

    # Corrupting process: scipy.signal.decimate uses chebyshev of order 8
    downsample_factor = 2
    corrupted = signal.decimate(upscaled, downsample_factor, axis=1)

    # Paper implements an 88%-6%-6% split on train, test, validation, we will use 90-10 for now
    cutoff_index = int(len(originals)*0.90)

    train_corrupted = corrupted[:cutoff_index]
    train_originals = originals[cutoff_index:]
    test_corrupted = corrupted[:cutoff_index]
    test_originals =  originals[cutoff_index:]


    print("Generated %d samples of HR patch length: %d, LR patch length = %d, downsampling factor = %d" %(upscaled.shape[0], upscaled.shape[1], corrupted.shape[1], downsample_factor))

    # Figure out validation
    return train_corrupted, train_originals, test_corrupted, test_originals, demo_sr
