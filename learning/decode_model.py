#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

from torch.autograd import Variable
import torch

from model.model import SpeechSegmentor
sys.path.append('./back_end')
from feature_extractor import extract_mfcc
from data_handler import switchboard_dataset, switchboard_dataset_after_embeddings, preaspiration_dataset, toy_dataset


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

def convert_to_model_format(features, is_cuda):
    ''' 
    Reshape a feature vector to a 3d tensor (batch) with one sequence 
    TODO: use bigger batches as in training
    '''

    torch_batch = Variable(features.view(1, features.size(0), -1))
    lengths = Variable(torch.LongTensor([torch_batch.size(1)]))

    if is_cuda:
        torch_batch = torch_batch.cuda()
        lengths = lengths.cuda()

    return torch_batch, lengths


def decode_data(model, dataset, is_cuda):

    # Loop over all dataset examples
    left_err = 0
    right_err = 0
    X = []
    Y = []
    for features, labels in dataset:

        # Convert it to a batch tensor with one object (TODO: use a bigger batche)
        batch, lengths = convert_to_model_format(features, is_cuda) 

        # Predict using the model
        segmentations, _ = model(batch, lengths)
        segmentations = segmentations[0]
         
        print 'Predicted:', segmentations
        print 'Gold:', labels

        if len(segmentations) != len(labels):
            print 'Bad length - the predicted length is %d while the gold length is %d' %  \
                                                   (len(segmentations), len(labels))
            continue

        # Ignore the fixed boundaries:
        predicted_labels = segmentations[1:-1]
        gold_labels      = labels[1:-1]

        # store pre-aspiration durations
        X.append(gold_labels[1]-gold_labels[0])
        Y.append(predicted_labels[1]-predicted_labels[0])

        # not found - zeros vector
        if predicted_labels[1] <= predicted_labels[0]:
            print 'Warning - event has not found in: %s' % file
            
        left_err += np.abs(gold_labels[0]-predicted_labels[0])
        right_err += np.abs(gold_labels[1]-predicted_labels[1])

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
    parser.add_argument("decode_path", help="A path to a directory containing the extracted features for the decoding")
    parser.add_argument("--dataset", help="dataset/testset to be decoded: sb(switchboard)/pa/toy", default='pa')
    parser.add_argument('--no-cuda',  help='disables training with CUDA (GPU)', action='store_true', default=False)
    args = parser.parse_args()

    args.is_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.is_cuda:
        print '==> Decoding on GPU using cuda'
    else:
        print '==> Decoding on CPU'

    if args.dataset == 'sb':
        print '==> Decoding preprocessed switchboard testset '
        dataset = switchboard_dataset_after_embeddings(dataset_path=args.decode_path,
                                                       hop_size=0.5) # hop_size should be the same as used 
                                                                     # in get_embeddings.sh
    elif args.dataset == 'pa':
        print '==> Decoding preaspiration testset'
        dataset = preaspiration_dataset(args.decode_path)
    # Synthetic simple dataset for debugging
    elif args.dataset == 'toy':
        print '==> Decoding toy testset'
        dataset = toy_dataset(dataset_size=1000, seq_len=100)
    else:
        raise ValueError("%s - illegal dataset" % args.dataset)

    # Construct a model with the pre-trained parameters
    model = SpeechSegmentor(rnn_input_dim=dataset.input_size, 
                            load_from_file=args.params_path,
                            is_cuda=args.is_cuda)

    # Decode the given files
    decode_data(model, dataset, args.is_cuda)