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
    @returns : a tuple of the form (train_corrupted, train_originals, test_corrupted, test_originals)
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

    # Retrieve high resolution ones which we will corrupt
    originals = files['hr']

    # Corrupting process: scipy.signal.decimate uses chebyshev of order 8 TODO: this gives padlen errors?
    downsample_factor = 2
    corrupted = signal.decimate(originals, downsample_factor)

    # TODO: Cubic interpolation so they are all the same size, discretize train and test samples
    upscaled = interpolate.interp1d(range(len(originals)), originals, kind='cubic')

    # Paper implements an 88%-6%-6% split on train, test, validation
    cutoff_index = int(len(originals)*0.88)

    train_corrupted = corrupted[:cutoff_index]
    train_originals = originals[cutoff_index:]
    test_corrupted = corrupted[:cutoff_index]
    test_originals =  originals[cutoff_index:]

    # 6% should be left for validation, how to validate here?
    return train_corrupted, train_originals, test_corrupted, test_originals