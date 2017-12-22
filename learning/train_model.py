#!/usr/bin/python

import time
import signal
import argparse
import random
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model.model import SpeechSegmentor
from back_end.data_handler import switchboard_dataset, preaspiration_dataset, toy_dataset

DEV_SET_PROPORTION        = 0.1


def convert_to_batches(data, batch_size, is_cuda):
    """
    Data: list of tuples each from the format: (tensor, label)
          where the tensor should be 2d contatining features
          for each time frame.  

    Output: A list contating tuples from the format: (batch_tensor, lengths, labels)      
    """

    # Loop over the examples and gather them into padded batches
    batches = []
    for i in range(0, len(data), batch_size):

        # Get tensors and labels
        tensors = [ex[0] for ex in data[i:i+batch_size]]
        labels = [ex[1] for ex in data[i:i+batch_size]]

        # Get sizes
        current_batch_size = len(tensors)
        max_length = max([ten.size(0) for ten in tensors])
        features_length = tensors[0].size(1)

        # Get a padded batch tensor
        padded_batch = torch.zeros(current_batch_size, max_length, features_length)

        lengths = []
        for j,ten in enumerate(tensors):
            current_length = ten.size(0)
            padded_batch[j] = torch.cat([ten, torch.zeros(max_length - current_length, features_length)])
            lengths.append(current_length)

        # Convert to cuda variables
        padded_batch = Variable(padded_batch)
        lengths = Variable(torch.LongTensor(lengths))

        if is_cuda:
            padded_batch = padded_batch.cuda()
            lengths = lengths.cuda()

        # Add it to the batches list along with the real lengths and labels
        batches.append((padded_batch, lengths, labels))

    return batches

def train_model(model, train_data, dev_data, learning_rate, batch_size, iterations, is_cuda, patience, params_file):
    ''' 
    Train the network 
    '''

    # Preprocess data into batches
    train_batches = convert_to_batches(train_data, batch_size, is_cuda)
    dev_batches   = convert_to_batches(dev_data, batch_size, is_cuda)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    best_dev_loss = 1e3
    best_iter = 0
    consecutive_no_improve = 0
    print 'Start training the model..'
    for ITER in xrange(iterations):
        print '-------- Epoch #%d --------' % ITER

        random.shuffle(train_batches)
        model.train()

        # Run train epochs
        train_closs = 0.0
        train_accuracy = 0
        for batch, lengths, segmentations in train_batches:

            # Clear gradients (Pytorch accumulates gradients)
            model.zero_grad()

            # Forward pass on the network
            start_time = time.time()

            # Predict
            pred_segmentations, pred_scores = model(batch, lengths, segmentations)
       
            print("Forward: %s seconds ---" % (time.time() - start_time))

            # Get gold scores
            start_time = time.time()
            gold_scores = model.get_score(batch, segmentations)
            print("Get score: %s seconds ---" % (time.time() - start_time))

            #print 'pred score: %s' % str(pred_scores.data.cpu().numpy())
            #print 'gold score: %s' % str(gold_scores.data.cpu().numpy())
            print segmentations
            print '------------------------------------------------------------'
            print pred_segmentations

            start_time = time.time()
            # Hinge loss with margin (ReLU to zero out negative losses)
            batch_loss = nn.ReLU()(1 + pred_scores - gold_scores)

            loss = torch.mean(batch_loss)
            print "Batch losses: ", batch_loss
            print "The avg loss is %s" % str(loss)
            train_closs += float(loss.data[0])

            # back propagation
            loss.backward()
            optimizer.step()
            print("Backwards: %s seconds ---" % (time.time() - start_time))
            
        # Evaluation mode
        model.eval()

        # Check performance on the dev set
        dev_closs = 0.0
        dev_accuracy = 0
        for batch, lengths, segmentations in dev_batches:

            # Clear gradients (Pytorch accumulates gradients)
            model.zero_grad()

            # Forward pass on the network
            pred_segmentations, pred_scores = model(batch, lengths)

            # Get gold scores
            gold_scores = model.get_score(batch, segmentations)

            # Should be the structural hinge loss here
            loss = torch.mean(1 + pred_scores - gold_scores)
            print "The dev avg loss is %s" % str(loss)
            print "dev segmentations:\n%s" % str(pred_segmentations)
            dev_closs += float(loss.data[0])

        # Average train and dev losses
        avg_train_loss = train_closs / len(train_batches)
        avg_dev_loss = dev_closs / len(dev_batches)

        print "#####################################################################"
        print "Train avg loss %s | Dev avg loss: %s" % (avg_train_loss, avg_dev_loss)
        print "#####################################################################"

        # check if it's the best (minimum) loss so far
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
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

    print 'Learning process has finished!'


if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="A path to the training set")
    parser.add_argument("params_path", help="A path to a file in which the trained model parameters will be stored")
    parser.add_argument("--dataset", help="Which dataset to use: sb(switchboard)/pa/toy", default='sb')
    parser.add_argument('--learning_rate', help='The learning rate', default=0.001, type=float)
    parser.add_argument('--num_iters', help='Number of iterations (epochs)', default=5000, type=int)
    parser.add_argument('--batch_size', help='Size of training batch', default=5, type=int)
    parser.add_argument('--patience', help='Num of consecutive epochs to trigger early stopping', default=10, type=int)
    parser.add_argument('--no-cuda',  help='disables training with CUDA (GPU)', action='store_true', default=False)
    args = parser.parse_args()

    args.is_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.is_cuda:
        print '==> Training on GPU using cuda'
    else:
        print '==> Training on CPU'
    
    if args.dataset == 'sb':
        print '==> Using switchboard dataset'
        dataset = switchboard_dataset(wav_path='data/swbI_release2/audio/',
                                      trans_path='data/swbI_release2/trans/',
                                      feature_type='mfcc',
                                      sample_rate=16000, 
                                      win_size=100, # In ms
                                      run_over=False)
    elif args.dataset == 'pa':
        print '==> Using preaspiration dataset'
        dataset = preaspiration_dataset(args.train_path)
    # Synthetic simple dataset for debugging
    elif args.dataset == 'toy':
        print '==> Using toy dataset'
        dataset = toy_dataset(dataset_size=4000, seq_len=100)
    else:
        raise ValueError("%s - illegal dataset" % args.dataset)

    # split the dataset into training set and validation set
    train_set_size = int((1-DEV_SET_PROPORTION) * len(dataset))
    train_data = dataset[:train_set_size]
    dev_data   = dataset[train_set_size:]

    # create a new model 
    model = SpeechSegmentor(rnn_input_dim=dataset.input_size, is_cuda=args.is_cuda)

    # train the model
    train_model(model,
                train_data,
                dev_data,  
                args.learning_rate,
                args.batch_size,
                args.num_iters,
                args.is_cuda,
                args.patience,
                args.params_path)

