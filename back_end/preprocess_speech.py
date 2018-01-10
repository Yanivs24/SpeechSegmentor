import numpy as np
import librosa
import soundfile as sf
from vad import VAD


def get_speech_times(frames, sample_rate, threshold=0, win_size=0.03, hop_size=0.01):
    '''
    Get the times in which the VAD (Voice Activity Detector) detected voice 

    Returns: 
        a list of tuples each contains speech-utterance boundaries
    '''

    detector = VAD(fs=sample_rate, win_size_sec=win_size, win_hop_sec=hop_size)
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


def trim_nonspeech(wav_path_in, sample_rate, wav_path_out):
    '''
    Get a wav file and create a new wav after cropping the non-speech frames 
    '''

    frames, rate = librosa.load(wav_path_in, sample_rate)
    voice_times = get_speech_times(frames, sample_rate)

    cropped_frames = np.array([])
    for start_t, end_t in voice_times:
        cropped_frames = np.append(cropped_frames,
                                   frames[int(start_t*sample_rate): int(end_t*sample_rate)])

    # Write cropped wav file as 16-bit Signed Integer PCM (using PySoundFile)
    sf.write(wav_path_out, cropped_frames, sample_rate, subtype='PCM_16')

    return cropped_frames





