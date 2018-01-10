#!/usr/bin/python

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena


import numpy as np
import random
import argparse
import sys
import os

FILE_WITH_FEATURE_FILE_LIST = 'feature_names.txt'
FILE_WITH_LABELS            = 'labels.txt'
FIRST_FEATURE_INDEX = 1
LAST_FEATURE_INDEX  = 9
WINDOW_STD          = 20


def get_feature_files(feature_path):
    full_path = os.path.join(feature_path, FILE_WITH_FEATURE_FILE_LIST)
    with open(full_path) as f:
        file_names = f.readlines()

    return [line.strip() for line in file_names]

def get_labels(feature_path):
    full_path = os.path.join(feature_path, FILE_WITH_LABELS)
    with open(full_path) as f:
        file_labels = f.readlines()

    return [map(int, line.strip().split()) for line in file_labels[1:]]

def read_features(file_name):
    return np.loadtxt(file_name, skiprows=1)[:, FIRST_FEATURE_INDEX:LAST_FEATURE_INDEX]

def write_examples(data_set, output_path):
    with open(output_path, 'w') as f:
        for x,i,y in data_set:
            f.write("%s,%s,%s\n" % (' '.join(map(str, x)), str(i), str(y)))

def build_dataset(feature_path, output_path):
    # get feature files, one file for each example (voice segment)
    print 'Reading features and labels from %s' % feature_path

    # get names of feature files 
    feature_files_list = get_feature_files(feature_path)

    # get their corresponding labels
    labels_list = get_labels(feature_path)

    print 'Extracting frames from the feature files..'
    data_set = []
    # run over all feature files
    for file, labels in zip(feature_files_list, labels_list):

        # Fix labels - these labels assumes counting from 1 - so decrement
        labels = (labels[0]-1, labels[1]-1)

        # get feature matrix and the segment size
        fe_matrix = read_features(file) 
        full_segment_size = fe_matrix.shape[0]

        # Choose the size of the windows surrounding the event randomly - this is done in order to 
        # avoid a situation in which the constant length of the windows affects the BiRNN (the RNN can learn
        # things from the length of the windows)
        left_index  = max(0, int(np.random.normal(0.5*labels[0], WINDOW_STD)))
        right_index = min(full_segment_size, int(np.random.normal(0.5*(full_segment_size+labels[1]), WINDOW_STD)))

        if left_index >= right_index:
            print 'skipping example - too short'
            continue

        # Crop the segment using the random window-sizes
        # print 'Croping segment to %s:%s' % (left_index, right_index)
        cur_frame = fe_matrix[left_index:right_index, :].flatten()

        # Get the relative labels after cropping the segment
        cropped_size = right_index - left_index
        relative_lbl_left  = min(cropped_size, max(0, labels[0] - left_index))
        relative_lbl_right = min(cropped_size, max(0, labels[1] - left_index))

        # Build example from the flatten frame and the new labels - 
        # add it to the data set
        data_set.append((cur_frame, relative_lbl_left, relative_lbl_right))

        # Debug:
        # print 'Real labels: %s,%s' % labels
        # print 'Cropped to : %s,%s' % (left_index, right_index)
        # print 'New labels : %s,%s' % (relative_lbl_left, relative_lbl_right)

    # shuffle data 
    random.shuffle(data_set)

    print '%s examples were extracted from the files' % len(data_set)

    # write data set to file
    print 'Write examples to: "%s"' % output_path
    write_examples(data_set, output_path)


if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_path", help="A path to a directory containing the extracted feature-files and the labels")
    parser.add_argument("output_path", help="The path to the output file")
    args = parser.parse_args()

    build_dataset(args.feature_path, args.output_path)

