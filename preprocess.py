import numpy as np
import tensorflow as tf
from scipy import signal, io
import glob
import librosa

def get_data(VCTK=False):
    """
    Do things
    """

    audio_type = ['hr', 'lr', 'pr', 'sp']

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

