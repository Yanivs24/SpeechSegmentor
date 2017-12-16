import os
import numpy as np
import cPickle as pickle
import torch
import random
from torch.utils.data import Dataset
from feature_extractor import feature_extractors

FEATURES_DIR   = 'tmp_files/features'

WAV_EXTENSION  = 'wav'
WAV_PREFIX     = 'sw0'
MARK_EXTENSION = 'mrk'
MARK_PREFIX    = 'sw'

# Max allowed size of swith-board speech utterance (in seconds)
SB_MAX_EXAMPLE_SIZE = 100

PREASPIRATION_NUM_OF_FEATURES = 8

def load_switchboard(wav_path, trans_path, features_type, sample_rate, win_size, run_over=False, **kwargs):
    print "Loading switchboard wav files from '%s'" % wav_path
    print "Loading switchboard transcripts files from '%s'" % trans_path

    dataset_filename = 'switchboard_%s.dat' % features_type
    dataset_path = os.path.join(FEATURES_DIR, dataset_filename)

    # Check if the dataset already exists
    if not run_over and os.path.exists(dataset_path):
        print 'Loading switchboasrd processesd dataset from %s' % dataset_path
        return load_dataset_from_file(dataset_path)

    # Get feature extractor
    if not feature_extractors.has_key(features_type):
        raise KeyError("The features type %s does not exist." % features_type)
    extractor = feature_extractors[features_type]

    # Loop over the files and extract features and labels (segmentations) from them
    file_ids = get_annotated_ids(wav_path, trans_path)
    print 'Constructing dataset from %s files..' % str(len(file_ids))
    dataset = []
    for file_id in file_ids:
        wav_file_path  = os.path.join(wav_path, '{0}{1}.{2}'.format(WAV_PREFIX, file_id, WAV_EXTENSION))
        mark_file_path = os.path.join(trans_path, '{0}{1}.{2}'.format(MARK_PREFIX, file_id, MARK_EXTENSION))

        snippets = switchboard_extract_segmentation(mark_file_path, win_size)
        features = extractor(wav_file_path, sample_rate, win_size, **kwargs)

        # Convert the features into torch tensor
        features = torch.FloatTensor(features.transpose())

        # We have the whole wav-file features and we crop it for each snippet 
        for boundaries, seg in snippets:
            snippet_features = features[boundaries[0]: boundaries[1]+1, :]
            dataset.append((snippet_features, seg))

    # Save the dataset for later use
    print 'Constructed dataset of %s examples.' % str(len(dataset))
    print "Saving dataset to %s.." % dataset_path
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset

def switchboard_extract_segmentation(mark_file_path, win_size):
    '''
    Get .mrk file from the switchboard corpus and extract the speaker turn-change
    times (i.e. the segmentation).

    params:
        mark_file_path - the .mrk file mark
        win_size       - the resolution in ms (to convert times to indexes)
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

    # Build a list of examples extracted from this file, each one represents a 
    # snippet from the wav file and built from:
    #   1) The start-index and end-index of this snippet 
    #   2) Sequence of indexes - the segmentation
    # NOTE: all the indexes / segmentations are relative to win_size
    snippets = []
    current_segmentation = [0]
    prev_speaker = ''
    prev_speaker_end_time = None
    snippet_start_time = 0
    new_snippet = True
    # Loop over the transcript and search for speaker turn-changes
    for i,fields in enumerate(lines_fields):

        # We work relatively to the current snippet's start time
        if new_snippet:
            new_snippet = False
            snippet_start_time = float(fields[1])

        speaker_start_time = float(fields[1]) - snippet_start_time
        speaker_end_time   = speaker_start_time + float(fields[2])

        # The first letter of the first field is the speaker (A or B)
        speaker = fields[0][0]
        if speaker not in ('A', 'B'):
            raise ValueError("Found illegal speaker in the file: %s" % mark_file_path)

        # If the margin is more than 250ms - consider it as a non-speech segment
        # TODO: add segment type for diarization
        # if (speaker_start_time - prev_speaker_end_time) > 0.25:
        #     current_segmentation.append(int(prev_speaker_end_time*1e3/win_size))
        #     current_segmentation.append(int(speaker_start_time*1e3/win_size))

        # If this line contains a new speaker - and the mergin between their speech time
        # is not big - use the median of the times as the turn-change time
        if prev_speaker and prev_speaker != speaker:
            median_time = 0.5 * (prev_speaker_end_time + speaker_start_time)
            current_segmentation.append(int(median_time*1e3/win_size))
            # TODO: add speaker for diarization

        # Finish this example/snippet (and start a new one if it's not the last line)
        if (speaker_end_time >= SB_MAX_EXAMPLE_SIZE) or (i == len(lines_fields)-1):

            # Add the relative end-index of the last speaker to this segmentation
            current_segmentation.append(int(speaker_end_time*1e3/win_size))

            # Get snippet absolute time-indexes and add the snippet
            snippet_start_index = int(snippet_start_time*1e3/win_size)
            snippet_end_index   = int((snippet_start_time+speaker_end_time)*1e3/win_size)
            snippets.append(((snippet_start_index, snippet_end_index),
                            current_segmentation[:]))

            # Start a new snippet
            new_snippet = True
            current_segmentation = [0]
            speaker = ''
            prev_speaker_end_time = 0

        prev_speaker          = speaker
        prev_speaker_end_time = speaker_end_time

    return snippets

def get_annotated_ids(wav_path, trans_path):

    wav_ids  = [f[len(WAV_PREFIX): -len(WAV_EXTENSION)-1] for f in os.listdir(wav_path) if f.endswith(WAV_EXTENSION)]
    mark_ids = [f[len(MARK_PREFIX): -len(MARK_EXTENSION)-1] for f in os.listdir(trans_path) if f.endswith(MARK_EXTENSION)]

    # intesection between wav files and mark files (as we need both)
    annotated_ids = set(wav_ids) & set(mark_ids)

    return list(annotated_ids)

def load_dataset_from_file(dataset_path):
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

def create_simple_dataset(dataset_size, seq_len):

    dataset = []
    for _ in range(dataset_size):

        ex = torch.zeros(seq_len, 1)

        start = random.randrange(1, seq_len-2)
        end   = random.randrange(start+1, seq_len-1)

        ex[start:end+1] = 1

        dataset.append((ex, (0, start, end, seq_len-1)))

    return dataset


'''                 DATASETS                      '''
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
class switchboard_dataset(Dataset):
    def __init__(self, wav_path, trans_path, feature_type, sample_rate, win_size, **kwargs):
        self.data = load_switchboard(wav_path, trans_path, feature_type, sample_rate, win_size, **kwargs)
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
