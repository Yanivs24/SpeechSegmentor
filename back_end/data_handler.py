import os
import numpy as np
import cPickle as pickle
import librosa
import soundfile as sf
import torch
import random
import h5py
from shutil import copyfile
from collections import OrderedDict
from torch.utils.data import Dataset

from feature_extractor import feature_extractors
from preprocess_speech import trim_nonspeech, fix_segmentation_after_trimming

FEATURES_DIR   = 'tmp_files/features'

WAV_EXTENSION  = 'wav'
WAV_PREFIX     = 'sw0'
MARK_EXTENSION = 'mrk'
MARK_PREFIX    = 'sw'
SEG_EXTENSION  = 'seg' 
H5_EXTENSION   = 'h5' 

PREASPIRATION_NUM_OF_FEATURES = 8


def load_switchboard(preprocessed_data_path, features_type, sample_rate, win_size, run_over=False, **kwargs):
    '''
    Load preprocessed switchboard data from a directory, extract features and return a dataset that
    can be used directly by the torch model.
    '''
    print "Loading switchboard dataset files from '%s'" % preprocessed_data_path

    dataset_filename = 'switchboard_%s.dat' % features_type
    dataset_path = os.path.join(FEATURES_DIR, dataset_filename)

    # Check if the dataset already exists
    if not run_over and os.path.exists(dataset_path):
        print 'Loading switchboard processesd dataset from %s' % dataset_path
        return load_serialized_data(dataset_path)

    # Get feature extractor
    if not feature_extractors.has_key(features_type):
        raise KeyError("The features type %s does not exist." % features_type)
    extractor = feature_extractors[features_type]

    # Loop over the files and extract features and labels (segmentations) from them
    wav_names = [os.path.splitext(fn)[0] for fn in os.listdir(preprocessed_data_path) if fn.endswith(WAV_EXTENSION)]
    seg_names = [os.path.splitext(fn)[0] for fn in os.listdir(preprocessed_data_path) if fn.endswith(SEG_EXTENSION)]
    file_names = list(set(wav_names) & set(seg_names))
    print 'Constructing dataset from %s files..' % str(len(file_names))
    dataset = []
    for file in file_names:
        wav_file_path = os.path.join(preprocessed_data_path, '{0}.{1}'.format(file, WAV_EXTENSION))
        seg_file_path = os.path.join(preprocessed_data_path, '{0}.{1}'.format(file, SEG_EXTENSION))

        features = extractor(wav_file_path, sample_rate, win_size, **kwargs)
        seg = load_serialized_data(seg_file_path)

        # Convert segmentation from float times into indexes due to 'win_size'
        seg = [int(i*1e3/win_size) for i in seg]

        # Convert the features into torch tensor
        features = torch.FloatTensor(features.transpose())

        # Add the conversation to the dataset
        dataset.append((features, seg))

    # Save the dataset for later use
    print 'Constructed dataset of %s examples.' % str(len(dataset))
    print "Saving dataset to %s.." % dataset_path
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset

def load_switchboard_after_embeddings(embeddings_data_path, hop_size):
    '''
    Load preprocessed switchboard data from a directory after extracting speech
    turn embedding using pyannote.audio from switchboard's wav files. 

    Params:
        embeddings_data_path - path to a directory containing the .h5 files and their
                               corresponding .seg files
        hop_size             - the hop size that was used to extract the speaker embeddings -  
                               we use it to convert the segmentation from time to indexes.
    '''
    print "Loading switchboard speaker embeddings from '%s'" % embeddings_data_path

    # Get all the the conversations file-names
    h5_names = [os.path.splitext(fn)[0] for fn in os.listdir(embeddings_data_path) if fn.endswith(H5_EXTENSION)]
    seg_names = [os.path.splitext(fn)[0] for fn in os.listdir(embeddings_data_path) if fn.endswith(SEG_EXTENSION)]
    file_names = list(set(h5_names) & set(seg_names))

    print 'Constructing dataset from %s files..' % str(len(file_names))
    dataset = []
    for file in file_names:
        h5_file_path  = os.path.join(embeddings_data_path, '{0}.{1}'.format(file, H5_EXTENSION))
        seg_file_path = os.path.join(embeddings_data_path, '{0}.{1}'.format(file, SEG_EXTENSION))

        # Read features (speaker embeddings) from the .h5 file 
        file_h5 = h5py.File(h5_file_path, 'r')
        features = np.array(file_h5['features'])

        # Get the segmentation and convert it from float times into indexes due to 'hop_size'
        seg = load_serialized_data(seg_file_path)
        seg = [int(t/hop_size) for t in seg]
        # Remove illegal indexes from the right edge of the list and add the right boundary
        seg = filter(lambda x: x < len(features), seg)
        seg.append(len(features)-1)

        # Remove duplicates 
        seg = OrderedDict((x, True) for x in seg).keys()

        # TEMP DEBUG - remove small speech turns
        #seg = [seg[0]]+[seg[i] for i in range(1, len(seg)-1) if (seg[i]-seg[i-1] > 4) and (seg[i+1]-seg[i] > 4)] + [seg[-1]]

        # Convert the features into torch tensor
        features = torch.FloatTensor(features)

        # Add the conversation to the dataset
        dataset.append((features, seg))

    print 'Constructed dataset of %s examples.' % str(len(dataset))
    return dataset

def preprocess_switchboard_dataset_step1(wav_dir_path, mark_dir_path, result_dir_path, trim_non_speech=True, sample_rate=16000):
    '''
    Perform the first preprocessing stage for the swithcboard dataset.
    The preprocessing contains the following steps:
        1) Removing all non-speech frames from the wav files (if trim_non_speech flag is on)
        2) Extracting the speaker turn-changes from the .mark files (i.e. the segmentation)
        3) Fixing the segmentation indexes due to the trimmed wav file (if trim_non_speech flag is on)

    In the end of this process we'll have trimmed wav files (with only voice frames) and
    their corresponding label files (the segmentation).
    '''

    # Get all files that has both wav file and mark file
    file_ids = switchboard_get_annotated_ids(wav_dir_path, mark_dir_path)
    print '==> Preprocessing dataset of %s files..' % str(len(file_ids))

    # Loop over the files, trim them, and extract suitable labels (segmentations)
    for file_id in file_ids:
        src_wav_file_path  = os.path.join(wav_dir_path, '{0}{1}.{2}'.format(WAV_PREFIX, file_id, WAV_EXTENSION))
        src_mark_file_path = os.path.join(mark_dir_path, '{0}{1}.{2}'.format(MARK_PREFIX, file_id, MARK_EXTENSION))

        dst_wav_file_path = os.path.join(result_dir_path, '{0}.wav'.format(file_id))
        dst_vad_file_path = os.path.join(result_dir_path, '{0}.vad'.format(file_id))
        dst_seg_file_path = os.path.join(result_dir_path, '{0}.seg'.format(file_id))

        # Trim wav file and write the result to a new file
        if trim_non_speech:
            print 'Trimming "%s" and storing the result in "%s"..' % (src_wav_file_path, dst_wav_file_path)
            voice_times = trim_nonspeech(src_wav_file_path, sample_rate, dst_wav_file_path)
            print 'Saving VAD output file in "%s"' % dst_vad_file_path
            with open(dst_vad_file_path, 'wb') as f:
                pickle.dump(voice_times, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print 'Copying "%s" to "%s"..' % (src_wav_file_path, dst_wav_file_path)
            copyfile(src_wav_file_path, dst_wav_file_path)

        # Get segmentation 
        seg = switchboard_extract_segmentation(src_mark_file_path)
        # fix it due to the voice times after trimming
        if trim_non_speech:
            seg = fix_segmentation_after_trimming(seg, voice_times)

        # Store the segmentation
        print 'Saving segmentation file in "%s"' % dst_seg_file_path
        with open(dst_seg_file_path, 'wb') as f:
            pickle.dump(seg, f, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess_switchboard_dataset_step2(dataset_dir_path, result_dir_path, max_duration=100, sample_rate=16000):
    ''' 
    Trim (slice) the conversations and their corresponding segmentations according
    to some max duration set by max_duration (in seconds).

    Note:
    This should be called after calling 'preprocess_switchboard_dataset_step1' and 
    'dataset_dir_path' should be the result path of this step.
    '''

    # Get all the the conversations file-names
    wav_names = [os.path.splitext(fn)[0] for fn in os.listdir(dataset_dir_path) if fn.endswith(WAV_EXTENSION)]
    seg_names = [os.path.splitext(fn)[0] for fn in os.listdir(dataset_dir_path) if fn.endswith(SEG_EXTENSION)]
    file_names = list(set(wav_names) & set(seg_names))

    for file in file_names:
        wav_file_path = os.path.join(dataset_dir_path, '{0}.{1}'.format(file, WAV_EXTENSION))
        seg_file_path = os.path.join(dataset_dir_path, '{0}.{1}'.format(file, SEG_EXTENSION))
        wav_dst_base_path = os.path.join(result_dir_path, '{0}.{1}'.format(file+'_trimmed%s', WAV_EXTENSION))
        seg_dst_base_path = os.path.join(result_dir_path, '{0}.{1}'.format(file+'_trimmed%s', SEG_EXTENSION))

        # Get the frames from the wav
        frames, rate = librosa.load(wav_file_path, sample_rate)

        # Get the segmentation
        seg = load_serialized_data(seg_file_path)

        print 'Trimming file "%s" to max duration of %s' % (wav_file_path, str(max_duration))
        trim_count = 0
        while True:
            duration_left = float(len(frames)) / sample_rate
            if (duration_left <= 0.3*max_duration) or (max(seg) == 0):
                break

            print 'duration_left: ', duration_left
            print 'seg: ', seg

            # Find the trimmimg index (the nearest time in the segmentation)
            nearest_idx = (np.abs(np.array(seg)-max_duration)).argmin()

            # Trim frames and update the rest
            amount_of_trimmed_frames = int(seg[nearest_idx] * sample_rate)
            trimmed_frames = frames[:amount_of_trimmed_frames]
            frames = frames[amount_of_trimmed_frames:]

            # Trim seg and update and normalize the rest
            trimmed_seg = seg[:nearest_idx+1]
            seg = seg[nearest_idx:]
            seg = [seg[i] - seg[0] for i in range(len(seg))]

            # Write trimmed wav file as 16-bit Signed Integer PCM (using PySoundFile)
            trimmed_wav_path = wav_dst_base_path % str(trim_count)
            print 'Writing trimmed wav file to %s' % trimmed_wav_path
            sf.write(trimmed_wav_path, trimmed_frames, sample_rate, subtype='PCM_16')
            # Write trimmed segmentation
            trimmed_seg_path = seg_dst_base_path % str(trim_count)
            print 'Writing trimmed seg file to %s' % trimmed_seg_path
            with open(trimmed_seg_path, 'wb') as f:
                pickle.dump(trimmed_seg, f, protocol=pickle.HIGHEST_PROTOCOL)

            trim_count += 1

def switchboard_extract_segmentation(mark_file_path):
    '''
    Get .mrk file from the switchboard corpus and extract the speaker turn-change
    times (i.e. the segmentation).

    params:
        mark_file_path - the .mrk file mark

    return:
        list of turn-changes
    '''
    with open(mark_file_path) as f:
        lines = f.readlines()

    # Split each line to its fields
    lines_fields = []
    for line in lines:
        fields = line.split()
        # ignore if there are not enough fields
        if len(fields) < 4:
            continue
        # ignore non-speech marks
        if '*' in fields:
            continue

        # I encountered lines that start with '*' or '@' - strip them
        fields[0] = fields[0].strip('*@')

        lines_fields.append(fields)

    # Build a list of time-indexes each represents a speaker turn-change
    segmentation = [0]
    prev_speaker = ''
    prev_speaker_end_time = None
    # Loop over the transcript and search for speaker turn-changes
    for i,fields in enumerate(lines_fields):

        speaker_start_time = float(fields[1])
        speaker_end_time   = speaker_start_time + float(fields[2])

        # The first letter of the first field is the speaker (A or B)
        speaker = fields[0][0]
        if speaker not in ('A', 'B'):
            raise ValueError("Found illegal speaker in the file: %s" % mark_file_path)

        # If this line contains a new speaker - use the median of the times
        # as the turn-change time
        if prev_speaker and prev_speaker != speaker:
            median_time = 0.5 * (prev_speaker_end_time + speaker_start_time)
            segmentation.append(median_time)
            # TODO: add speaker for diarization

        prev_speaker          = speaker
        prev_speaker_end_time = speaker_end_time

    return segmentation

def switchboard_get_annotated_ids(wav_path, trans_path):

    wav_ids  = [f[len(WAV_PREFIX): -len(WAV_EXTENSION)-1] for f in os.listdir(wav_path) if f.endswith(WAV_EXTENSION)]
    mark_ids = [f[len(MARK_PREFIX): -len(MARK_EXTENSION)-1] for f in os.listdir(trans_path) if f.endswith(MARK_EXTENSION)]

    # intesection between wav files and mark files (as we need both)
    annotated_ids = set(wav_ids) & set(mark_ids)

    return sorted(list(annotated_ids))

def load_serialized_data(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def load_preaspiration(dataset_path):
    '''    
    Get pre-aspiration dataset from a preprocessed file
    '''

    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    raw_dataset = [l.strip().split(',') for l in lines]
    print 'Got %s training examples!' % (len(raw_dataset))

    dataset = []
    for ex in raw_dataset:
        # Get flat vector  and labels
        flat_seq = np.fromstring(ex[0], sep=' ')
        left_label = int(ex[1]) - 1 
        right_label = int(ex[2]) - 1 

        # Ignore bad labels
        if left_label < 0 or right_label < 0:
            continue

        # each PREASPIRATION_NUM_OF_FEATURES values is a frame
        num_of_frames = float(len(flat_seq)) / PREASPIRATION_NUM_OF_FEATURES
        # handle error
        if not num_of_frames.is_integer():
            raise ValueError("Input vector size is not a multiple of the features amount (%s)" % NUM_OF_FEATURES_PER_FRAME)

        # cast to integer
        num_of_frames = int(num_of_frames)

        # For pytorch - reshape it to a 2d tensor of vectors of dim 'PREASPIRATION_NUM_OF_FEATURES' -
        # this represents the speech segment. Also, make it a torch tensor.
        flat_seq = torch.from_numpy(flat_seq).float()
        speech_seq = flat_seq.view(-1, PREASPIRATION_NUM_OF_FEATURES)

        # append to the dataset (for segmentation task)
        dataset.append((speech_seq, (0, left_label, right_label, len(speech_seq)-1)))
    
    return dataset

def create_simple_dataset(dataset_size, seq_len, max_seg_size=10):

    dataset = []
    for _ in range(dataset_size):

        ex = torch.zeros(seq_len, 1)

        # Get random segmentation
        seg_size = np.random.randint(max_seg_size)
        seg = [0] + sorted(set(np.random.randint(1, seq_len-1, seg_size))) + [seq_len-1]

        for i in range(len(seg)-1):
            ex[seg[i]:seg[i+1]] = i
        ex[seq_len-1] = i

        dataset.append((ex, seg))

    return dataset


'''                 DATASETS                      '''
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
class switchboard_dataset(Dataset):
    def __init__(self, dataset_path, feature_type, sample_rate, win_size, **kwargs):
        self.data = load_switchboard(dataset_path, feature_type, sample_rate, win_size, **kwargs)
        self.input_size = self.data[0][0].size(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class switchboard_dataset_after_embeddings(Dataset):
    def __init__(self, dataset_path, hop_size=0.25):
        self.data = load_switchboard_after_embeddings(dataset_path, hop_size)
        self.input_size = self.data[0][0].size(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class preaspiration_dataset(Dataset):
    def __init__(self, dataset_path):
        self.data = load_preaspiration(dataset_path)
        self.input_size = self.data[0][0].size(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class toy_dataset(Dataset):
    def __init__(self, dataset_size, seq_len):
        self.data = create_simple_dataset(dataset_size, seq_len)
        self.input_size = self.data[0][0].size(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
