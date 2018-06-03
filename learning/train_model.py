#!/usr/bin/python
from __future__ import print_function

import argparse
import random
import sys
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

sys.path.append('./back_end')
from tensorboardX import SummaryWriter

from data_handler import (preaspiration_dataset,
                          switchboard_dataset_after_embeddings, timit_dataset,
                          toy_dataset, general_dataset)
from model.model import SpeechSegmentor


DEV_SET_PROPORTION        = 0.3
TXT_SUFFIX = '.txt'
DATA_SUFFIX = '.data'
writer = SummaryWriter()


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


    epoch_metrics = pd.DataFrame(columns=['train_epoch_loss', 'dev_epoch_loss',
                                          'dev_precision', 'dev_recall',
                                          'dev_f1'])
    best_dev_loss = float("inf")
    consecutive_no_improve = 0
    print('Start training the model..')
    for ITER in range(iterations):
        print('-------- Epoch #%d --------' % (ITER+1))

        random.shuffle(train_batches)
        model.train()

        # Run train epochs
        train_closs = 0.0
        n_batches = len(train_batches)
        for batch_idx, (batch, lengths, segmentations) in enumerate(tqdm(train_batches)):

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
            print("The avg loss is %s" % str(loss))
            train_closs += float(loss.data[0])

            writer.add_scalars('metrics',
                               {
                                 "train_batch_loss": float(loss.data[0])
                               }, ITER * n_batches + batch_idx)

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

            # For debugging
            print(segmentations)
            print('------------------------------------------------------------')
            print(pred_segmentations)

            print("The dev avg loss is %s" % str(loss))
            taskloss = 0
            if use_taskloss:
                taskloss = torch.mean(model.get_task_loss(pred_segmentations, segmentations))
                print("The dev avg taskloss is %s" % str(taskloss))
            dev_closs += float(loss.data[0])
            dev_ctaskloss += taskloss

        # Average train and dev losses
        avg_train_loss = train_closs / len(train_batches)
        avg_dev_loss = dev_closs / len(dev_batches)
        if use_taskloss:
            avg_dev_taskloss = dev_ctaskloss / len(dev_batches)

        # Evaluate performence
        EPS = 1e-5
        dev_precision = float(dev_precision_counter) / (dev_pred_counter+EPS)
        dev_recall    = float(dev_recall_counter) / (dev_gold_counter+EPS)
        dev_f1        = (2 * (dev_precision*dev_recall) / (dev_precision+dev_recall+EPS))

        print("#####################################################################")
        print("Results for Epoch #%d" % (ITER+1))
        print("Train avg loss %s | Dev avg loss: %s" % (avg_train_loss, avg_dev_loss))
        if use_taskloss:
            print("Dev avg taskloss: %f" % avg_dev_taskloss)
        print("Dev precision: %f" % dev_precision)
        print("Dev recall: %f" % dev_recall)
        print("Dev F1 score: %f" % dev_f1)
        print("#####################################################################")

        # log tensorboard metrics
        writer.add_scalars('metrics',
                           {
                             "train_epoch_loss": avg_train_loss,
                             "dev_epoch_loss": avg_dev_loss,
                             "dev_precision": dev_precision,
                             "dev_recall": dev_recall,
                             "dev_f1": dev_f1
                           }, ITER + 1)
        # log metrics to dataframe for later evaluation
        epoch_metrics.loc[ITER] = (avg_train_loss, avg_dev_loss, dev_precision,
                                   dev_recall, dev_f1)

        # Check if it's the best loss so far on the validation set
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            consecutive_no_improve = 0
            # store parameters after each loss improvement
            print('Best dev loss so far - storing parameters in %s' % params_file)
            model.store_params(params_file)
        else:
            consecutive_no_improve += 1

        # After #patience consecutive epochs without loss improvements - stop training
        if consecutive_no_improve == patience:
            print('No loss improvements - stop training!')
            return epoch_metrics

    print('Tranining process has finished!')
    return epoch_metrics


def main(args):
    args.is_cuda = args.use_cuda and torch.cuda.is_available()

    if args.is_cuda:
        print('==> Training on GPU using cuda')
    else:
        print('==> Training on CPU')

    # Always use task-loss when k in known
    if args.use_k:
        args.use_task_loss = True

    if args.dataset == 'sb':
        print('==> Using preprocessed switchboard dataset')
        dataset = switchboard_dataset_after_embeddings(dataset_path=args.train_path,
                                                       hop_size=0.5) # hop_size should be the same as used
                                                                     # in get_embeddings.sh
        if args.val_path:
            dev_data = switchboard_dataset_after_embeddings(dataset_path=args.val_path,
                                                           hop_size=0.5) # hop_size should be the same as used
                                                                         # in get_embeddings.sh

    elif args.dataset == 'pa':
        print('==> Using preaspiration dataset')
        dataset = preaspiration_dataset(args.train_path)
        if args.val_path:
            dev_data = preaspiration_dataset(args.val_path)

    # Synthetic simple dataset for debugging
    elif args.dataset == 'toy':
        print('==> Using toy dataset')
        dataset = toy_dataset(dataset_size=1000, seq_len=100)
        args.val_path = None

    elif args.dataset == 'timit':
        print('==> Using TIMIT dataset')
        dataset = timit_dataset(args.train_path)
        if args.val_path:
            dev_data = timit_dataset(args.val_path)

    elif args.dataset == 'vot' or args.dataset == 'word':
        print('==> Using VOT dataset')
        dataset = general_dataset(args.train_path, TXT_SUFFIX)
        args.max_segment_size = dataset.max_seg_size
        if args.val_path:
            dev_data = general_dataset(args.val_path, TXT_SUFFIX)

    elif args.dataset == 'vowel':
        print('==> Using Vowel dataset')
        dataset = general_dataset(args.train_path, DATA_SUFFIX)
        args.max_segment_size = dataset.max_seg_size
        if args.val_path:
            dev_data = general_dataset(args.val_path, DATA_SUFFIX)

    else:
        raise ValueError("%s - illegal dataset" % args.dataset)

    print('\n===> Got %s examples' % str(len(dataset)))

    if not args.val_path:
        # split the dataset into training set and validation set
        train_set_size = int((1-DEV_SET_PROPORTION) * len(dataset))
        train_data = dataset[:train_set_size]
        dev_data   = dataset[train_set_size:]
    else:
        train_data = dataset

    # create a new model
    model = SpeechSegmentor(rnn_input_dim=dataset.input_size,
                            load_from_file=args.init_params,
                            is_cuda=args.is_cuda,
                            use_task_loss=args.use_task_loss,
                            task_loss_coef=args.task_loss_coef,
                            max_segment_size=args.max_segment_size,
                            load_lstm_from_file=args.init_lstm_params)

    # train the model
    return train_model(model=model,
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


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="A path to the training set")
    parser.add_argument("params_path", help="A path to a file in which the trained model parameters will be stored")
    parser.add_argument("--val_path", help="A path to the training set", default=None)
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
    main(args)
