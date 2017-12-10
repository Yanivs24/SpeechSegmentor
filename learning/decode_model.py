#!/usr/bin/python

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from torch.autograd import Variable
import torch

from model.model import SpeechSegmentor

FILE_WITH_FEATURE_FILE_LIST = 'feature_names.txt'
FILE_WITH_LABELS            = 'labels.txt'
FIRST_FEATURE_INDEX = 1
LAST_FEATURE_INDEX  = 9
NUM_OF_FEATURES_PER_FRAME = LAST_FEATURE_INDEX-FIRST_FEATURE_INDEX


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
    numpy_features =  np.loadtxt(file_name, skiprows=1)[:, FIRST_FEATURE_INDEX:LAST_FEATURE_INDEX]

    # For pytorch - reshape it to a 3d tensor (batch) with one sequence
    torch_tensor = Variable(torch.from_numpy(numpy_features).float())
    torch_batch = torch_tensor.view(1, -1, NUM_OF_FEATURES_PER_FRAME)
    lengths = Variable(torch.LongTensor([torch_batch.size(1)]))
    return torch_batch, lengths

def decode_files(model, feature_path):

    # get names of feature files to decode
    feature_files_list = get_feature_files(feature_path)

    # get their corresponding labels
    labels_list = get_labels(feature_path)

    # run over all feature files
    left_err = 0
    right_err = 0
    X = []
    Y = []
    for file, labels in zip(feature_files_list, labels_list):

        # get a feature matrix and convert it into a pytorch tensor
        features_tensor, lengths = read_features(file) 

        # Fix labels - these labels assume counting from 1 - so decrement
        labels = (labels[0]-1, labels[1]-1)

        # Predict using the model
        segmentations, _ = model(features_tensor, lengths)
         

        print segmentations
        print labels

        # Debug:
        predicted_labels = [1, 2]

        # store pre-aspiration durations
        X.append(labels[1]-labels[0])
        Y.append(predicted_labels[1]-predicted_labels[0])

        # not found - zeros vector
        if predicted_labels[1] <= predicted_labels[0]:
            print 'Warning - event has not found in: %s' % file
            

        left_err += np.abs(labels[0]-predicted_labels[0])
        right_err += np.abs(labels[1]-predicted_labels[1])

    print 'left_err: ',  float(left_err)/len(feature_files_list)
    print 'right_err: ', float(right_err)/len(feature_files_list)

    X = np.array(X)
    Y = np.array(Y)

    print "Mean of labeled/predicted preaspiration: %sms, %sms" % (str(np.mean(X)), str(np.mean(Y)))
    print "Standard deviation of labeled/predicted preaspiration: %sms, %sms" % (str(np.std(X)), str(np.std(Y)))
    print "max of labeled/predicted preaspiration: %sms, %sms" % (str(np.max(X)), str(np.max(Y)))
    print "min of labeled/predicted preaspiration: %sms, %sms" % (str(np.min(X)), str(np.min(Y)))


    thresholds = [2, 5, 10, 15, 20, 25, 50]
    print "Percentage of examples with labeled/predicted PA difference of at most:"
    print "------------------------------"
    
    for thresh in thresholds:
        print "%d msec: " % thresh, 100*(len(X[abs(X-Y)<thresh])/float(len(X)))


if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_path", help="A path to a directory containing the extracted feature-files and the labels")
    parser.add_argument("params_path", help="A path to a file containing the model parameters (after training)")
    args = parser.parse_args()

    # Construct a model with the pre-trained parameters
    model = SpeechSegmentor(is_cuda=False, load_from_file=args.params_path)

    # Decode the given files
    decode_files(model, args.feature_path)