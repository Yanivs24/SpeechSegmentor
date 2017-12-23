#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from torch.autograd import Variable
import torch

from model.model import SpeechSegmentor
from back_end.feature_extractor import extract_mfcc


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

def read_features(wav_path, sample_rate, win_size):
    ''' Extract features (MFCCs) and convert them to a torch tensor '''

    features = extract_mfcc(wav_path, sample_rate, win_size)
    features = torch.FloatTensor(features.transpose())

    # Reshape it to a 3d tensor (batch) with one sequence
    torch_batch = Variable(features.view(1, features.size(0), -1))
    lengths = Variable(torch.LongTensor([torch_batch.size(1)]))
    return torch_batch, lengths

def decode_wav(model, wav_path, sample_rate=16000, win_size=100):
    ''' Decode single wav file using the model '''
    batch, lengths = read_features(wav_path, sample_rate, win_size)
    return model(batch, lengths)

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
    parser.add_argument("params_path", help="A path to a file containing the model parameters (after training)")
    parser.add_argument("feature_path", help="A path to a directory containing the extracted feature-files and the labels")
    parser.add_argument('--no-cuda',  help='disables training with CUDA (GPU)', action='store_true', default=False)
    args = parser.parse_args()

    args.is_cuda = not args.no_cuda and torch.cuda.is_available()

    # Construct a model with the pre-trained parameters
    model = SpeechSegmentor(load_from_file=args.params_path, is_cuda=args.is_cuda)

    # Decode the given files
    decode_files(model, args.feature_path)