import numpy as np
import librosa
from librosa.feature.spectral import melspectrogram
from vad import VAD


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

def trim_nonspeech(wav_path, sample_rate):
    ''' Get a wav file and trim the non-speech frames (using Voice Activity Detector)'''

    frames, rate = librosa.load(wav_path, sample_rate)
    voice_times = get_speech_times(frames, sample_rate, threshold=0)

    trimmed_frames = np.array([])
    for start_t, end_t in voice_times:
        trimmed_frames = np.append(trimmed_frames,
                                   frames[int(start_t*sample_rate): int(end_t*sample_rate)])

    return trimmed_frames, voice_times

def get_voice_activity(frames, sample_rate, threshold=0, hop_size=0.01):
    ''' Get the times in which the VAD detected voice 

        returns a list of tuples each contains speech-utterance boundaries
    '''
    detector = VAD(fs=sample_rate, win_size_sec=0.03, win_hop_sec=hop_size)
    decisions = detector.detect_speech(frames, threshold=threshold)

    speech_times = []
    old_dec = False
    current_start = 0
    for i, dec in enumerate(decisions):
        if dec and not old_dec:
            current_start = i
        if old_dec and not dec: 
            speech_times.append((current_start*hop_size, i*hop_size))

        old_dec = dec

    return speech_times

# Supported feature extraction methods
feature_extractors = {
    'mfcc': extract_mfcc
}