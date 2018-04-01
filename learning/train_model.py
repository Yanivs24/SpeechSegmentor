#!/usr/bin/python

import sys
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
sys.path.append('./back_end')
from data_handler import switchboard_dataset, switchboard_dataset_after_embeddings, preaspiration_dataset, toy_dataset

DEV_SET_PROPORTION        = 0.3


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

        # Convert to variables
        padded_batch = Variable(padded_batch)
        lengths = Variable(torch.LongTensor(lengths))

        if is_cuda:
            padded_batch = padded_batch.cuda()
            lengths = lengths.cuda()

        # Add it to the batches list along with the real lengths and labels
        batches.append((padded_batch, lengths, labels))

    return batches

def train_model(model, train_data, dev_data, learning_rate, batch_size, iterations,
                is_cuda, patience, use_k, use_taskloss, params_file):
    ''' 
    Train the network 
    '''

    # Preprocess data into batches
    train_batches = convert_to_batches(train_data, batch_size, is_cuda)
    dev_batches   = convert_to_batches(dev_data, batch_size, is_cuda)

    # Use SGD optimizer
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

    best_dev_loss = 1e3
    best_iter = 0
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

            print segmentations
            print '------------------------------------------------------------'
            print pred_segmentations

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
        dev_correct_counter = 0
        dev_gold_counter    = 0
        dev_pred_counter    = 0
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

            # Update counters for precision and recall
            for gold_seg, pred_seg in zip(segmentations, pred_segmentations):
                # Correct predictions (with 2 indexes forgiveness collar)
                for y in pred_seg:
                    if filter(lambda t: abs(t-y) <= 2, gold_seg):
                        dev_correct_counter += 1 
                # Count boundaries                        
                dev_gold_counter += len(gold_seg)
                dev_pred_counter += len(pred_seg)

            # Structural loss
            if use_taskloss:
                batch_loss = pred_scores - gold_scores
            # Hinge loss with margin (ReLU to zero out negative losses)
            else:
                batch_loss = nn.ReLU()(1 + pred_scores - gold_scores)

            loss = torch.mean(batch_loss)

            taskloss = torch.mean(model.get_task_loss(pred_segmentations, segmentations))

            print segmentations
            print '------------------------------------------------------------'
            print pred_segmentations

            print "The dev avg loss is %s" % str(loss)
            print "The dev avg taskloss is %s" % str(taskloss)
            dev_closs += float(loss.data[0])
            dev_ctaskloss += taskloss

        # Average train and dev losses
        avg_train_loss = train_closs / len(train_batches)
        avg_dev_loss = dev_closs / len(dev_batches)
        avg_dev_taskloss = dev_ctaskloss / len(dev_batches)

        print "#####################################################################"
        print "Results for Epoch #%d" % (ITER+1)
        print "Train avg loss %s | Dev avg loss: %s" % (avg_train_loss, avg_dev_loss)
        print "Dev avg taskloss: %f" % avg_dev_taskloss
        print "Dev precision: %f" % (float(dev_correct_counter) / dev_pred_counter)
        print "Dev recall: %f" % (float(dev_correct_counter) / dev_gold_counter)
        print "#####################################################################"

        # check if it's the best (minimum) task loss so far
        if avg_dev_taskloss < best_dev_loss:
            best_dev_loss = avg_dev_taskloss
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
    parser.add_argument("--dataset", help="Which dataset to use: sb(switchboard)/pa/toy", default='sb')
    parser.add_argument('--learning_rate', help='The learning rate', default=0.0001, type=float)
    parser.add_argument('--num_iters', help='Number of iterations (epochs)', default=5000, type=int)
    parser.add_argument('--batch_size', help='Size of training batch', default=20, type=int)
    parser.add_argument('--patience', help='Num of consecutive epochs to trigger early stopping', default=5, type=int)
    parser.add_argument('--use_cuda',  help='disables training with CUDA (GPU)', action='store_true', default=False)
    parser.add_argument("--init_params", help="Start training from a set of pretrained parameters", default='')
    parser.add_argument('--use_task_loss', help='Train with strucutal loss using task loss (always on when k is known)', action='store_true', default=False)
    parser.add_argument('--use_k', help='Apply inference when k (num of segments) is known for each example', action='store_true', default=False)
    parser.add_argument('--task_loss_coef', help='Task loss coefficient', default=0.001, type=float)
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
        #print '==> Using switchboard dataset'
        # dataset = switchboard_dataset(dataset_path=args.train_path,
        #                               feature_type='mfcc',
        #                               sample_rate=16000, 
        #                               win_size=100, # In ms
        #                               run_over=True)
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
                            task_loss_coef=args.task_loss_coef)

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