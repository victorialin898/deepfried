import numpy as np
import tensorflow as tf
from scipy import signal, io
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
    def search_by_wav_file_type(wav):
        return glob.glob("./data/**/*.%s.wav"%(wav), recursive=True)

    #load all file_paths
    #librosa loads all of them, saving them as Tuple(<Audio Time Series[np.array], Sampling Rate [float])
    files = {aud: np.array(list(map(librosa.load, search_by_wav_file_type(aud)))) for aud in audio_type}
    
    #get that hr is high res, lr is low res; what is pr and sp?
    train_corrupted = files['lr']
    train_originals = files['hr']
    test_corrupted = files['pr']
    test_originals = files['sp']

    return train_corrupted, train_originals, test_corrupted, test_originals