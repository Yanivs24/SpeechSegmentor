import numpy as np
import librosa
import soundfile as sf
import cPickle as pickle
from vad import VAD


def get_voice_times(frames, sample_rate, threshold=0, win_size=0.03, hop_size=0.01):
    '''
    Get the times in which the VAD (Voice Activity Detector) detected voice 

    Returns: 
        a list of tuples each contains a voice-segment boundaries
    '''

    detector = VAD(fs=sample_rate, win_size_sec=win_size, win_hop_sec=hop_size)
    decisions = detector.detect_speech(frames, threshold=threshold)

    voice_times = []
    old_dec = False
    current_start = 0
    for i, dec in enumerate(decisions):
        if dec and not old_dec:
            current_start = i
        if old_dec and not dec: 
            voice_times.append((current_start*hop_size, i*hop_size))

        old_dec = dec

    return voice_times

def trim_nonspeech(wav_path_in, sample_rate, wav_path_out):
    '''
    Get a wav file and create a new wav after cropping the non-speech frames 
    '''

    frames, rate = librosa.load(wav_path_in, sample_rate)
    voice_times = get_voice_times(frames, sample_rate)

    cropped_frames = np.array([])
    for start_t, end_t in voice_times:
        cropped_frames = np.append(cropped_frames,
                                   frames[int(start_t*sample_rate): int(end_t*sample_rate)])

    # Write cropped wav file as 16-bit Signed Integer PCM (using PySoundFile)
    sf.write(wav_path_out, cropped_frames, sample_rate, subtype='PCM_16')

    return voice_times

def trim_nonspeech_dir(wav_dir_path_in, sample_rate, wav_dir_path_out):
    '''
    Get a directory path and trim all the wav files in it using VAD. 
    The trimmed wav files are placed in 'wav_dir_path_out'. 
    For each trimmed wav file, we also create a text file (.trim) that
    stores the voice segments boundaries - this is done in order 
    to synchronise between absolute time-indexes and our trimmed file.
    '''
    wav_files  = [fn for fn in os.listdir(wav_dir_path_in) if fn.endswith('.wav')]
    
    # Trim all wav files
    for wav_file in wav_files:
        src_file_path = os.path.join(wav_dir_path_in, wav_file) 
        dst_file_path = os.path.join(wav_dir_path_out, wav_file)
        trim_file_path = os.path.join(wav_dir_path_out, "%s.trim" % os.path.splitext(wav_file)[0])
        # trim wav file 
        print 'Trimming "%s" and storing the output in "%s"..' % (wav_file, wav_dir_path_out)
        voice_times = trim_nonspeech(src_file_path, sample_rate, dst_file_path)
        # Save a text file that describes the trimmed parts (for later use)
        print 'Saving trim description file in "%s"' % trim_file_path
        with open(trim_file_path, 'wb') as f:
            pickle.dump(voice_times, f, protocol=pickle.HIGHEST_PROTOCOL)

def fix_segmentation_after_trimming(seg, voice_segments):
    '''
    Get a segmentation (time-indexes) and a list of voice segments that were not cropped and
    find the appropriate indexes in the trimmed file
    '''
    new_seg = []
    last_t = 0
    last_voice_index = 0

    for t in seg:
        # Calc voice length since last index until this time (t)
        voice_len = 0
        while True:
            start, end = voice_segments[last_voice_index]
            real_start = max(start, last_t)
            if t < start:
                # We not reached next segment
                break
            elif t <= end:
                # t is inside the current segment - take the relative voice
                voice_len += t-real_start
                break
            else:
                # t is after this segment - take it and keep going
                voice_len += end-real_start
                last_voice_index += 1

        last_t = t
        last_aligned_index = 0 if not new_seg else new_seg[-1]
        new_seg.append(last_aligned_index + voice_len)

    return new_seg






    
