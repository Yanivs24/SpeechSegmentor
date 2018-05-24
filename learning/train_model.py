#!/usr/bin/python

import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
sys.path.append('./back_end')
from data_handler import (general_dataset, preaspiration_dataset,
                          switchboard_dataset_after_embeddings, timit_dataset,
                          toy_dataset)
from model.model import SpeechSegmentor



DEV_SET_PROPORTION        = 0.3
TXT_SUFFIX = '.txt'
DATA_SUFFIX = '.data'


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

def train_model(model, train_data, dev_data, learning_rate, batch_size, iterations,
                is_cuda, patience, use_k, use_taskloss, params_file):
    '''
    Train the network
    '''

    # Preprocess data into batches
    train_batches = convert_to_batches(train_data, batch_size, is_cuda, use_k)
    dev_batches   = convert_to_batches(dev_data, batch_size, is_cuda, use_k)

    # Use SGD optimizer
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

    best_dev_loss = 0
    consecutive_no_improve = 0
    print 'Start training the model..'
    for ITER in xrange(iterations):
        print '-------- Epoch #%d --------' % (ITER+1)

        random.shuffle(train_batches)
        model.train()

        # Run train epochs
        train_closs = 0.0
        for batch, lengths, segmentations in train_batches:

            # Clear gradients (Pytorch accumulates gradients)
            model.zero_grad()

            # Value to be sent to the model
            real_k = len(segmentations[0]) if use_k else None

            # Forward pass on the network
            start_time = time.time()
            pred_segmentations, pred_scores = model(batch,
                                                    lengths,
                                                    k=real_k,
                                                    gold_seg=segmentations)
            print("Forward: %s seconds ---" % (time.time() - start_time))

            # Get gold scores
            start_time = time.time()
            gold_scores = model.get_score(batch, lengths, segmentations)
            print("Get score: %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            # Structural loss
            if use_taskloss:
                batch_loss = pred_scores - gold_scores
            # Hinge loss with margin (ReLU to zero out negative losses)
            else:
                batch_loss = nn.ReLU()(1 + pred_scores - gold_scores)

            loss = torch.mean(batch_loss)
            print "The avg loss is %s" % str(loss)
            train_closs += float(loss.data[0])

            # Back propagation
            loss.backward()
            optimizer.step()
            print("Backwards: %s seconds ---" % (time.time() - start_time))

        # Evaluation mode
        model.eval()

        # Check performance on the dev set
        dev_closs = 0.0
        dev_ctaskloss = 0
        dev_precision_counter = 0
        dev_recall_counter    = 0
        dev_gold_counter      = 0
        dev_pred_counter      = 0
        for batch, lengths, segmentations in dev_batches:

            # Clear gradients (Pytorch accumulates gradients)
            model.zero_grad()

            # Value to be sent to the model
            real_k = len(segmentations[0]) if use_k else None

            # Forward pass on the network
            pred_segmentations, pred_scores = model(batch,
                                                    lengths,
                                                    k=real_k)

            # Get gold scores
            gold_scores = model.get_score(batch, lengths, segmentations)

            # Update counters for precision and recall (with 2 indexes forgiveness collar)
            for gold_seg, pred_seg in zip(segmentations, pred_segmentations):
                pred, gold = np.array(pred_seg), np.array(gold_seg)
                # Count for precision
                for y_hat in pred:
                    min_dist = min(np.abs(gold-y_hat))
                    dev_precision_counter += (min_dist<=2)
                # Count for recall
                for y in gold:
                    min_dist = min(np.abs(pred-y))
                    dev_recall_counter += (min_dist<=2)
                # Add amounts
                dev_pred_counter += len(pred_seg)
                dev_gold_counter += len(gold_seg)

            # Structural loss
            if use_taskloss:
                batch_loss = pred_scores - gold_scores
            # Hinge loss with margin (ReLU to zero out negative losses)
            else:
                batch_loss = nn.ReLU()(1 + pred_scores - gold_scores)

            loss = torch.mean(batch_loss)

            print segmentations
            print '------------------------------------------------------------'
            print pred_segmentations

            print "The dev avg loss is %s" % str(loss)
            taskloss = 0
            if use_taskloss:
                taskloss = torch.mean(model.get_task_loss(pred_segmentations, segmentations))
                print "The dev avg taskloss is %s" % str(taskloss)
            dev_closs += float(loss.data[0])
            dev_ctaskloss += taskloss

        # Average train and dev losses
        avg_train_loss = train_closs / len(train_batches)
        avg_dev_loss = dev_closs / len(dev_batches)
        if use_taskloss:
            avg_dev_taskloss = dev_ctaskloss / len(dev_batches)

        # Evaluate performence
        dev_precision = float(dev_precision_counter) / dev_pred_counter
        dev_recall    = float(dev_recall_counter) / dev_gold_counter
        dev_f1        = (2 * (dev_precision*dev_recall) / (dev_precision+dev_recall))

        print "#####################################################################"
        print "Results for Epoch #%d" % (ITER+1)
        print "Train avg loss %s | Dev avg loss: %s" % (avg_train_loss, avg_dev_loss)
        if use_taskloss:
            print "Dev avg taskloss: %f" % avg_dev_taskloss
        print "Dev precision: %f" % dev_precision
        print "Dev recall: %f" % dev_recall
        print "Dev F1 score: %f" % dev_f1
        print "#####################################################################"

        # check if it's the best loss so far (for now we use F1 score)
        if dev_f1 > best_dev_loss:
            best_dev_loss = dev_f1
            consecutive_no_improve = 0
            # store parameters after each loss improvement
            print 'Best dev loss so far - storing parameters in %s' % params_file
            model.store_params(params_file)
        else:
            consecutive_no_improve += 1

        # After #patience consecutive epochs without loss improvements - stop training
        if consecutive_no_improve == patience:
            print 'No loss improvements - stop training!'
            return

    print 'Tranining process has finished!'


if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="A path to the training set")
    parser.add_argument("params_path", help="A path to a file in which the trained model parameters will be stored")
    parser.add_argument("--dataset", help="Which dataset to use: sb(switchboard)/pa/toy/timit/vot/word/vowel", default='sb')
    parser.add_argument('--learning_rate', help='The learning rate', default=0.0001, type=float)
    parser.add_argument('--num_iters', help='Number of iterations (epochs)', default=5000, type=int)
    parser.add_argument('--batch_size', help='Size of training batch', default=20, type=int)
    parser.add_argument('--patience', help='Num of consecutive epochs to trigger early stopping', default=5, type=int)
    parser.add_argument('--use_cuda',  help='disables training with CUDA (GPU)', action='store_true', default=False)
    parser.add_argument("--init_params", help="Start training from a set of pretrained parameters", default='')
    parser.add_argument('--use_task_loss', help='Train with strucutal loss using task loss (always on when k is known)', action='store_true', default=False)
    parser.add_argument('--use_k', help='Apply inference when k (num of segments) is known for each example', action='store_true', default=False)
    parser.add_argument('--task_loss_coef', help='Task loss coefficient', default=0.001, type=float)
    parser.add_argument('--max_segment_size', help='Max searched segment size (in indexes)', default=52, type=int)
    parser.add_argument('--init_lstm_params', help='Load pretrained LSTM weights and used them as a fixed embedding layer', default='')
    args = parser.parse_args()

    args.is_cuda = args.use_cuda and torch.cuda.is_available()

    if args.is_cuda:
        print '==> Training on GPU using cuda'
    else:
        print '==> Training on CPU'

    # Always use task-loss when k in known
    if args.use_k:
        args.use_task_loss = True

    if args.dataset == 'sb':
        print '==> Using preprocessed switchboard dataset '
        dataset = switchboard_dataset_after_embeddings(dataset_path=args.train_path,
                                                       hop_size=0.5) # hop_size should be the same as used
                                                                     # in get_embeddings.sh
    elif args.dataset == 'pa':
        print '==> Using preaspiration dataset'
        dataset = preaspiration_dataset(args.train_path)
    # Synthetic simple dataset for debugging
    elif args.dataset == 'toy':
        print '==> Using toy dataset'
        dataset = toy_dataset(dataset_size=1000, seq_len=100)
    elif args.dataset == 'timit':
        print '==> Using TIMIT dataset'
        dataset = timit_dataset(args.train_path)
    elif args.dataset == 'vot' or args.dataset == 'word':
        print '==> Using VOT dataset'
        dataset = general_dataset(args.train_path, TXT_SUFFIX)
    elif args.dataset == 'vowel':
        print '==> Using Vowel dataset'
        dataset = general_dataset(args.train_path, DATA_SUFFIX)
    else:
        raise ValueError("%s - illegal dataset" % args.dataset)

    print '\n===> Got %s examples' % str(len(dataset))

    # split the dataset into training set and validation set
    train_set_size = int((1-DEV_SET_PROPORTION) * len(dataset))
    train_data = dataset[:train_set_size]
    dev_data   = dataset[train_set_size:]

    # create a new model
    model = SpeechSegmentor(rnn_input_dim=dataset.input_size,
                            load_from_file=args.init_params,
                            is_cuda=args.is_cuda,
                            use_task_loss=args.use_task_loss,
                            task_loss_coef=args.task_loss_coef,
                            max_segment_size=args.max_segment_size,
                            load_lstm_from_file=args.init_lstm_params)

    # train the model
    train_model(model=model,
                train_data=train_data,
                dev_data=dev_data,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                iterations=args.num_iters,
                is_cuda=args.is_cuda,
                patience=args.patience,
                use_k=args.use_k,
                use_taskloss=args.use_task_loss,
                params_file=args.params_path)
