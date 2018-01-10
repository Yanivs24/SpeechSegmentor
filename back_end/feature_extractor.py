import numpy as np
import librosa
from librosa.feature.spectral import melspectrogram


def extract_mfcc(path, sr, win_size, norm=False, noise=None):
    '''
    params:
        path        - path to wav file
        sr          - sample rate
        window_size - size of each mfcc window in milliseconds

    return:
        numpy 2D array containing the Mel-frequency cepstral coefficients
    '''
    data, rate = librosa.load(path, sr=sr)
    window_samples = int(float(rate) * win_size / 1000)
    if noise:
        data = noise.inject_noise(data)
    data = librosa.feature.mfcc(data, sr=rate, n_fft=window_samples, hop_length=window_samples)
    if norm:
        data = np.subtract(data, np.mean(data, axis=0)) / np.std(data, axis=0)
    return data
    

# Supported feature extraction methods
feature_extractors = {
    'mfcc': extract_mfcc
}