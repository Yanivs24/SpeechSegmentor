#!/usr/bin/python

import argparse
import sys

import numpy as np
import torch

from model.model import SpeechSegmentor
from train_model import convert_to_batches

sys.path.append('./back_end')
from data_handler import (preaspiration_dataset,
                          switchboard_dataset_after_embeddings, timit_dataset,
                          toy_dataset)

def decode_data(model, dataset_name, dataset, batch_size, is_cuda, use_k):

    labels = []
    predictions = []

    # Convert data to torch batches and loop over them
    batches = convert_to_batches(dataset, batch_size, is_cuda, use_k)
    for batch, lengths, segmentations in batches:

        # k Value to be sent to the model
        real_k = len(segmentations[0]) if use_k else None

        # Predict using the model
        preds, _ = model(batch, lengths, real_k)

        # Loop over the predictions of the batch
        for pred, gold in zip(preds, segmentations):
            print 'Predicted:', pred
            print 'Gold:', gold

            # If k is unknown, we expect to get the same size
            if use_k and len(pred) != len(gold):
                raise  ValueError('Bad length - the predicted length is %d while the gold length is %d' %  \
                                   (len(pred), len(gold)))

            # Store gold-labels and predictions of the current example
            labels.append(gold)
            predictions.append(pred)

    # Evaluate the performance according to the specific task
    # Fixed k
    if use_k:
        if dataset_name == 'pa':
            eval_performance_pa(labels, predictions)
        elif dataset_name == 'timit':
            eval_performance_timit(labels, predictions, use_k=True)
        else:
            print 'Performance evaluating with fixed K is not supported for %s' % dataset_name
    # k is not fixed
    else:
        if dataset_name == 'timit':
            eval_performance_timit(labels, predictions, use_k=False)
        else:
            print 'Performance evaluating with unknown K is supported for %s' % dataset_name

def eval_performance_pa(labels, predictions):
    ''' Evaluate performence for the preaspiration task '''

    gold_durations = []
    pred_durations = []
    left_err  = 0
    right_err = 0

    # Store event's duration for each example
    for gold, pred in zip(labels, predictions):
        gold_durations.append(gold[1]-gold[0])
        pred_durations.append(pred[1]-pred[0])
        # not found
        if pred[1] <= pred[0]:
             print 'Warning - bad prediction: %s' % str(pred)

        left_err += np.abs(gold[0]-pred[0])
        right_err += np.abs(gold[1]-pred[1])

    print 'left_err: ',  float(left_err)/len(labels)
    print 'right_err: ', float(right_err)/len(labels)

    Y     = np.array(gold_durations)
    Y_tag = np.array(pred_durations)

    print "Mean of labeled/predicted preaspiration: %sms, %sms" % (str(np.mean(Y)), str(np.mean(Y_tag)))
    print "Standard deviation of labeled/predicted preaspiration: %sms, %sms" % (str(np.std(Y)), str(np.std(Y_tag)))
    print "max of labeled/predicted preaspiration: %sms, %sms" % (str(np.max(Y)), str(np.max(Y_tag)))
    print "min of labeled/predicted preaspiration: %sms, %sms" % (str(np.min(Y)), str(np.min(Y_tag)))

    thresholds = [2, 5, 10, 15, 20, 25, 50]
    print "Percentage of examples with labeled/predicted difference of at most:"
    print "------------------------------"
    for thresh in thresholds:
        print "%d msec: " % thresh, 100*(len(Y[abs(Y-Y_tag)<thresh])/float(len(Y)))

def eval_performance_timit(labels, predictions, use_k):
    ''' Evaluate performence for the timit task '''

    # Here each index is 10ms wide
    thresholds = [1, 2, 3, 4]
    gold_all = []
    pred_all = []
    precisions = np.zeros(4)
    recalls    = np.zeros(4)

    for gold, pred in zip(labels, predictions):
        gold_all.extend(gold)
        pred_all.extend(pred)

        pred, gold = np.array(pred), np.array(gold)

        # More conservative matching algorithm (each boundary is used once)
        # for i,y_hat in enumerate(pred):
        #     # Find all golds within a 20ms window of the found boundary
        #     golds_in_win = gold[np.abs(gold-y_hat)<=2]
        #     # Miss - go to the next boundary
        #     if len(golds_in_win) == 0:
        #         continue

        #     # Hit
        #     precisions[2] += 1
        #     recalls[2] +=1

        #     # Find the closest hit
        #     closest = golds_in_win[np.abs(golds_in_win-y_hat).argmin()]

        #     # Remove our match from the golds, because we don't want to
        #     # use it again
        #     gold[gold==closest] = -100

        # Count for precision
        for y_hat in pred:
            min_dist = min(np.abs(gold-y_hat))
            for i in range(len(thresholds)):
                precisions[i] += (min_dist<=thresholds[i])
        # Count for recall
        for y in gold:
            min_dist = min(np.abs(pred-y))
            for i in range(len(thresholds)):
                recalls[i] += (min_dist<=thresholds[i])

    Y, Y_tag   = np.array(gold_all), np.array(pred_all)

    # Compare element-wise, relevant only if k is fixed
    if use_k:
        print "Percentage of examples with labeled/predicted difference of at most:"
        print "------------------------------"
        # Here each index is 10ms wide
        for thresh in thresholds:
            print "%d msec: " % (thresh*10), 100*(len(Y[abs(Y-Y_tag)<=thresh])/float(len(Y)))

    precisions = precisions / float(len(Y_tag))
    recalls    = recalls / float(len(Y))
    print "Proportion of labeled/predicted precision and recall of at most 10ms,20ms,30ms,40ms:"
    print "------------------------------"
    print "Precision: ", precisions
    print "Recall: ", recalls
    print "F1 score: ", 2*precisions*recalls / (precisions+recalls)

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
    parser.add_argument('--max_segment_size', help='Max searched segment size (in indexes)', default=52, type=int)
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
                            is_cuda=args.is_cuda,
                            max_segment_size=args.max_segment_size)

    # Decode the data
    decode_data(model=model,
                dataset_name=args.dataset,
                dataset=dataset,
                batch_size=args.batch_size,
                is_cuda=args.is_cuda,
                use_k=args.use_k)
