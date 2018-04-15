import numpy as np
import librosa
import torch
from torch.autograd import Variable


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

def get_mfcc_features(wav_path, sample_rate, win_size):
    ''' Extract features (MFCCs) and convert them to a torch tensor '''

    features = extract_mfcc(wav_path, sample_rate, win_size)
    features = torch.FloatTensor(features.transpose())

    # Reshape it to a 3d tensor (batch) with one sequence
    torch_batch = Variable(features.view(1, features.size(0), -1))
    lengths = Variable(torch.LongTensor([torch_batch.size(1)]))
    return torch_batch, lengths

def decode_wav(model, wav_path, sample_rate=16000, win_size=100):
    ''' Decode single wav file using the model '''
    batch, lengths = get_mfcc_features(wav_path, sample_rate, win_size)
    return model(batch, lengths)
    

# Supported feature extraction methods
feature_extractors = {
    'mfcc': extract_mfcc
}