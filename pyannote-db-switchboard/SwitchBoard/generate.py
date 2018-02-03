#!/usr/bin/env python

import os
from pyannote.audio.features.utils import get_wav_duration

WAV_EXTENSION  = '.wav'

SWITCHBOARD_WAV_PATH = '/home/yaniv/Projects/SpeechSegmentor/data/swbI_release2/preprocessed/trimmed'
DATA_FILES_PATH      = '/home/yaniv/Projects/SpeechSegmentor/pyannote-db-switchboard/SwitchBoard/data'


def generate_mdtm_files(wav_dir_path, dst_dir_path):
    subsets_wav_files  = dict()

    # Train - all the files
    subsets_wav_files['train'] = [os.path.splitext(fn)[0] for fn in os.listdir(wav_dir_path) if fn.endswith(WAV_EXTENSION)]
    # Validation - empty for now
    subsets_wav_files['dev'] = []
    # test - empty for now
    subsets_wav_files['test'] = []

    for subset in ('train', 'dev', 'test'):
        data_file_path = os.path.join(dst_dir_path, 'switchboard-main.{}.mdtm'.format(subset))

        # Write descriptions to the data file
        with open(data_file_path, 'w') as datafile:
            for file_name in subsets_wav_files[subset]:
                datafile.write('{uri} {channel} {start} {duration} {modality} {confidence} {gender} {label}\n'.format(
                uri = file_name,
                channel = 1,
                start = 0,
                duration = get_wav_duration(os.path.join(wav_dir_path, file_name) + WAV_EXTENSION),
                modality = 'speaker',
                confidence = 'NA',
                gender = 'male',
                label = file_name
                ))

if __name__ == '__main__':
    generate_mdtm_files(SWITCHBOARD_WAV_PATH, DATA_FILES_PATH)
    
