#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

from torch.autograd import Variable
import torch

from model.model import SpeechSegmentor
sys.path.append('./back_end')
from feature_extractor import extract_mfcc
from data_handler import switchboard_dataset_after_embeddings, preaspiration_dataset, toy_dataset, timit_dataset


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

def convert_to_batches(data, batch_size, is_cuda, fixed_k):
    """
    Data: list of tuples each from the format: (tensor, label)
          where the tensor should be 2d contatining features
          for each time frame.  

    Output: A list contating tuples from the format: (batch_tensor, lengths, labels)      
    """

    num_of_examples = len(data)

    # Loop over the examples and gather them into padded batches
    batches = []
    i = 0
    while i < num_of_examples:

        # Gather examples with the same number of segments - k (batching can only work if
        # k is fixed along the batch examples)
        if fixed_k:
            first_k = len(data[i][1])
            count = 0
            while (i+count < num_of_examples) and (first_k == len(data[i+count][1])):
                count += 1

            cur_batch_size = min(count, batch_size)
        else:
            cur_batch_size = min(num_of_examples-i, batch_size)

        # Get tensors and labels
        tensors = [ex[0] for ex in data[i:i+cur_batch_size]]
        labels = [ex[1] for ex in data[i:i+cur_batch_size]]
        
        # Get sizes
        max_length = max([ten.size(0) for ten in tensors])
        features_length = tensors[0].size(1)

        # Get a padded batch tensor
        padded_batch = torch.zeros(cur_batch_size, max_length, features_length)

        lengths = []
        for j,ten in enumerate(tensors):
            current_length = ten.size(0)
            padded_batch[j] = torch.cat([ten, torch.zeros(max_length - current_length, features_length)])
            lengths.append(current_length)

        # Convert to variables
        padded_batch = Variable(padded_batch)
        lengths = Variable(torch.LongTensor(lengths))

        if is_cuda:
            padded_batch = padded_batch.cuda()
            lengths = lengths.cuda()

        # Add it to the batches list along with the real lengths and labels
        batches.append((padded_batch, lengths, labels))

        # Progress i for the next batch
        i += cur_batch_size

    return batches

def decode_data(model, dataset, batch_size, is_cuda, use_k):

    left_err = 0
    right_err = 0
    X = []
    Y = []

    # Convert data to torch batches and loop over them
    batches = convert_to_batches(dataset, batch_size, is_cuda, use_k)
    for batch, lengths, segmentations in batches:

        # K Value to be sent to the model
        real_k = len(segmentations[0]) if use_k else None 

        # Predict using the model
        preds, _ = model(batch, lengths, real_k)
        
        # Loop over the predictions of the batch
        for pred, gold in zip(preds, segmentations):
            print 'Predicted:', pred
            print 'Gold:', gold

            # Relevant for predictions where k is unknown
            if len(pred) != len(gold):
                print 'Bad length - the predicted length is %d while the gold length is %d' %  \
                                                       (len(pred), len(gold))
                continue

            # Timit case (TODO: support all)
            X.extend(gold)
            Y.extend(pred)

            # Preaspiration 
            # # Store event's duration
            # X.append(gold[1]-gold[0])
            # Y.append(pred[1]-pred[0])

            # # not found - zeros vector
            # if pred[1] <= pred[0]:
            #     print 'Warning - bad prediction: %s' % str(pred)

            # left_err += np.abs(gold[0]-pred[0])
            # right_err += np.abs(gold[1]-pred[1])

    #print 'left_err: ',  float(left_err)/len(dataset)
    #print 'right_err: ', float(right_err)/len(dataset)

    X = np.array(X)
    Y = np.array(Y)

    # print "Mean of labeled/predicted preaspiration: %sms, %sms" % (str(np.mean(X)), str(np.mean(Y)))
    # print "Standard deviation of labeled/predicted preaspiration: %sms, %sms" % (str(np.std(X)), str(np.std(Y)))
    # print "max of labeled/predicted preaspiration: %sms, %sms" % (str(np.max(X)), str(np.max(Y)))
    # print "min of labeled/predicted preaspiration: %sms, %sms" % (str(np.min(X)), str(np.min(Y)))


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
    parser.add_argument("--dataset", help="dataset/testset to be decoded: sb(switchboard)/pa/toy/timit", default='pa')
    parser.add_argument('--batch_size', help='Size of training batch', default=32, type=int)
    parser.add_argument('--use_cuda',  help='disables training with CUDA (GPU)', action='store_true', default=False)
    parser.add_argument('--use_k', help='Apply inference when k (# of segments) is known for each example', action='store_true', default=False)
    args = parser.parse_args()

    args.is_cuda = args.use_cuda and torch.cuda.is_available()

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
    elif args.dataset == 'timit':
        print '==> Using timit dataset'
        dataset = timit_dataset(args.decode_path)
    else:
        raise ValueError("%s - illegal dataset" % args.dataset)

    # Construct a model with the pre-trained parameters
    model = SpeechSegmentor(rnn_input_dim=dataset.input_size, 
                            load_from_file=args.params_path,
                            is_cuda=args.is_cuda)

    # Decode the data
    decode_data(model=model, 
                dataset=dataset,
                batch_size=args.batch_size,
                is_cuda=args.is_cuda, 
                use_k=args.use_k)