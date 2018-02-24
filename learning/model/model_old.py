#!/usr/bin/python

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import time
import random


NUM_OF_FEATURES_PER_FRAME = 8


def sort_and_pack(tensor, lengths):
    seq_lengths = lengths
    sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
    index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(tensor)
    sorted_inputs = tensor.gather(0, index_sorted_idx.long())
    packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
    return packed_seq, sorted_idx


def unpack_and_unsort(packed, sorted_idx):
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    # unsort the output
    _, original_idx = sorted_idx.sort(0, descending=False)
    unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
    output = unpacked.gather(0, unsorted_idx.long())
    return output

class NotFoundRNNsError(Exception):
    pass


class SpeechSegmentor(nn.Module):
    def __init__(self, rnn_input_dim=NUM_OF_FEATURES_PER_FRAME, rnn_output_dim=60, mlp_hid_dim=80, is_cuda=True, use_srnn=False, load_from_file=''):

        super(SpeechSegmentor, self).__init__()

        # Use BiRNN and summation instead of segmental RNN (much faster)
        self.SIMPLE_MODE = not use_srnn

        self.rnn_output_dim = rnn_output_dim

        self.is_cuda = is_cuda

        # Network parameters:

        # BiLSTM (2D LSTM)
        self.BiRNN = nn.LSTM(rnn_input_dim, rnn_output_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.0)

        # Forward RNN
        self.RNN_F = nn.LSTM(rnn_input_dim, rnn_output_dim, num_layers=1, batch_first=True, dropout=0.3)

        # Backward RNN
        self.RNN_B = nn.LSTM(rnn_input_dim, rnn_output_dim, num_layers=1, batch_first=True, dropout=0.3)

        # MLP hidden layer
        if self.SIMPLE_MODE:
            self.mlp_linear1 = nn.Linear(2*rnn_output_dim, mlp_hid_dim)
        # This is because the concatat in phi
        else:
            self.mlp_linear1 = nn.Linear(6*rnn_output_dim, mlp_hid_dim)

        # MLP activation function
        self.mlp_activation = nn.ReLU()

        # MLP output layer (returns a scalar score)
        self.mlp_linear2 = nn.Linear(mlp_hid_dim, 1)

        # Make all the params cuda tensors
        if is_cuda:
            self.cuda()

        # If given - load the model's parameters from a file
        if load_from_file:
            self.load_params(load_from_file)

    def calc_birnn(self, batch, lengths):
        '''
        Get a batch with sequences and calculate its 2-layer BiRNN.
        Parameters:
            batch : A 3D torch tensor (batch_size, sequence_size, input_size)

        Returns:
            self.BiRNN_out (the BiRNN output)

        '''

        batch_size = batch.size(0)

        # Create random hidden states
        hidden = (Variable(torch.zeros(4, batch_size, self.rnn_output_dim)), 
                  Variable(torch.zeros(4, batch_size, self.rnn_output_dim)))

        if self.is_cuda:
            hidden = hidden[0].cuda(), hidden[1].cuda()

        packed_seq, sorted_idx = sort_and_pack(batch, lengths)
        out, hidden = self.BiRNN(packed_seq, hidden)
        out = unpack_and_unsort(out, sorted_idx)

        self.BiRNN_out = out

    def calc_all_rnns(self, batch):
        '''
        Get a sequence and calculate all the possible RNNs of its subsequences (segments).

        Parameters:
            batch : A 3D torch tensor (batch_size, sequence_size, input_size)

        Returns:
            (Forward python dict, Backward python dict)

        Notes:
            (*) We encode every segment by traversing its elements both forward and backwards using
                two different LSTMs (Total (sequence_size + sequence_size ** 2) possibilities)
        
            (*) We use dynamic programing algorithm that takes O(n**2).

            (*) This should be called before each forward()ing of a new examle

            (*) Although we work with batches, we will calculate all the possible segments
                RNN due to the max length. The caller should later use only the releveant
                segments according to the sequences' real lengths (which are not given here).
        '''

        # Get batch size and the max sequence length
        batch_size = batch.size(0)
        seq_length = batch.size(1)

        # Create random hidden states
        hidden_f = (Variable(torch.randn(1, batch_size, self.rnn_output_dim)), 
                    Variable(torch.randn(1, batch_size, self.rnn_output_dim)))
        hidden_b = (Variable(torch.randn(1, batch_size, self.rnn_output_dim)), 
                    Variable(torch.randn(1, batch_size, self.rnn_output_dim)))

        if self.is_cuda:
            hidden_f = hidden_f[0].cuda(), hidden_f[1].cuda()
            hidden_b = hidden_b[0].cuda(), hidden_b[1].cuda() 

        # Will hold the RNN outputs and the hidden states of the segments (2 directions)
        self.forward_rnns_dict = {}
        self.backward_rnns_dict = {}

        hidden_forward_rnns_dict = {}
        hidden_backward_rnns_dict = {}

        # First, get the diagonal (i.e. RNN(i, i))
        for i in range(seq_length):
            out, hidden = self.RNN_F(batch[:, i:i+1, :], hidden_f)
            hidden_forward_rnns_dict[(i, i)] = hidden
            self.forward_rnns_dict[(i, i)] = out[:, -1]

            out, hidden = self.RNN_B(batch[:, i:i+1, :], hidden_b)
            hidden_backward_rnns_dict[(i, i)] = hidden
            self.backward_rnns_dict[(i, i)] = out[:, -1]

        # Get all the rest (i.e. RNN(i, j), i != j)
        for i in range(seq_length):
            for j in range(i+1, seq_length):
                # Get elements above the diagonal - RNN_F(i, j)
                out, hidden = self.RNN_F(batch[:, j:j+1, :], hidden_forward_rnns_dict[(i, j-1)])
                hidden_forward_rnns_dict[(i, j)] = hidden
                self.forward_rnns_dict[(i, j)] = out[:, -1]

                # Get elements below the diagonal RNN_B(seq_length-1-i, seq_length-1-j)
                mirror_i = seq_length - 1 - i
                mirror_j = seq_length - 1 - j
                out, hidden = self.RNN_B(batch[:, mirror_j:mirror_j+1, :], hidden_backward_rnns_dict[(mirror_i, mirror_j+1)])
                hidden_backward_rnns_dict[(mirror_i, mirror_j)] = hidden
                self.backward_rnns_dict[(mirror_i, mirror_j)] =  out[:, -1]

    def mlp_layer(self, input):
        ''' 
        Propagate the given vector through a one hidden layer MLP with one output.
        '''
        hidden = self.mlp_activation(self.mlp_linear1(input))
        return self.mlp_linear2(hidden)

    def get_local_score(self, batch, y_start, y_end):
        ''' 
        Get the local score of the segment seq[y_start: y_end] for each sequence
        in the given batch.

        Parameters:        
            batch :  A 3D torch tensor (batch_size, sequence_size, input_size)
            y_start: Index in seq
            y_end:   Index in seq
        Returns:
            score tensor of dim: (batch_size, 1)

        Notes:
            Here we calculate the feature representation of the given segment 
            (i.e. phi(seq, y_i, y_i+1)) and feed it into a MLP that produces one output (score).
        '''

        seq_length = batch.size(1)

        try:
            if self.SIMPLE_MODE:
                features = torch.sum(self.BiRNN_out[:, y_start: y_end+1 ,:], dim=1)

            else:         
            
                # Get BiRNN(seq, y_start)
                birnn_y_start = torch.cat((self.forward_rnns_dict[0, y_start], 
                                           self.backward_rnns_dict[seq_length - 1, y_start]),
                                           dim=1)

                # Get BiRNN(seq, y_end)
                birnn_y_end = torch.cat((self.forward_rnns_dict[0, y_end], 
                                         self.backward_rnns_dict[seq_length - 1, y_end]),
                                         dim=1)

                # Concatenate the BiRNNs with RNN_F and RNN_B of the segment
                # This is the actual Phi function
                features = torch.cat((birnn_y_start, 
                                      birnn_y_end,
                                      self.forward_rnns_dict[y_start, y_end], 
                                      self.backward_rnns_dict[y_end, y_start]), 
                                      dim=1)

        except AttributeError:
            func_name = 'calc_birnn' if self.SIMPLE_MODE else 'calc_all_rnns'
            raise NotFoundRNNsError('"%s" must be called before calculating score or forward()ing!') % func_name

        return self.mlp_layer(features)



    def get_score(self, batch, segmentations):
        '''
        Get the score of the given segmentations of the sequences in 'batch'.

        Parameters:        
            batch : A 3D torch tensor: (batch_size, sequence_size, input_size)
            segmentations: A 2D python list, each internal list is a timing sequence of indexes 
            associated with a sequence in the batch

        Returns:
            Tensor of scores (for each sequence in 'batch')

        Notes:
            Before calling this function, "calc_all_rnns" or "calc_birnn" should be called with 'batch'.
        '''

        batch_size = batch.size(0)

        scores = Variable(torch.zeros(batch_size, 1))
        if self.is_cuda:
            scores = scores.cuda()

        for batch_ind, seg in enumerate(segmentations):
            local_scores = [self.get_local_score(batch, seg[i], seg[i+1])[batch_ind] for i in range(len(seg)-1)]
            scores[batch_ind] = sum(local_scores)

        return scores

    def get_local_task_loss(self, pred_segmentations, gold_segmentations, prev_y, new_y):

        batch_size = len(pred_segmentations)

        task_losses = torch.zeros(batch_size)

        # print 'pred:', pred_segmentations[0][prev_y]
        # print 'gold pred: ', gold_segmentations[0]
        # print 'new pred:', new_y
        # size = len(pred_segmentations[0][prev_y])
        # if len(gold_segmentations[0]) <= size:
        #     print 'More preds the gold - add penalty!'
        # else:
        #     print 'gold pred:', gold_segmentations[0][size]
        # print '\n'

        # Get distance of the new index for each element in the batch
        for batch_index in range(batch_size):
            # Amount of predicted points so far
            predicted_len = len(pred_segmentations[batch_index][prev_y])
            gold_len      = len(gold_segmentations[batch_index])

            # Prediction sequence is longer than gold - penalty
            if len(gold_segmentations[batch_index]) <= predicted_len:
                gold_y = 0
            # Take corresponding gold label
            else:
                gold_y = gold_segmentations[batch_index][predicted_len]

            task_losses[batch_index] = max(0, abs(gold_y - new_y) - 5)

            # Penalty - if this is the last segmnet of the prediction, and the size
            # of the prediction is different from the size of the gold segmentations:
            #if new_y == gold_segmentations[batch_index][-1] and ((predicted_len+1) != gold_len):
            #    task_losses[batch_index] += 20 * abs(predicted_len + 1 - gold_len)

        return task_losses

    def get_task_loss(self, pred_segmentations, gold_segmentations):

        batch_size = len(pred_segmentations)

        task_losses = torch.zeros(batch_size)

        # Calc loss
        for batch_index in range(batch_size):
            current_pred_len = len(pred_segmentations[batch_index])
            current_gold_len = len(gold_segmentations[batch_index])

            for i in range(current_pred_len):
                pred_yi = pred_segmentations[batch_index][i]
                if (i >= current_gold_len):
                    gold_yi = 0
                else:
                    gold_yi = gold_segmentations[batch_index][i]
                task_losses[batch_index] += max(0, abs(pred_yi - gold_yi) - 5)   

            #if current_gold_len > current_pred_len:
            #    task_losses[batch_index] += 20 * abs(current_gold_len - current_pred_len)


        return task_losses


    def forward(self, batch, lengths, gold_seg=None):
        '''
        Get the segmentation with the highest score using a practical dynamic
        programming algorithm.

        Parameters:
            batch :  A 3D torch tensor: (batch_size, sequence_size, input_size)
            lengths: A 1D tensor containing the lengths of the batch sequences
            [gold_seg]: A python list containing batch_size lists with the gold
                        segmentations. If given, we use the structural hinge loss
                        with margin.

        Notes:
            The algorithm complexity is O(n**2)
        '''
    
        # First, calculate all the possible segment-RNNs of the sequences in batch
        start_time = time.time()
        if self.SIMPLE_MODE:
            self.calc_birnn(batch, lengths)
        else:
            self.calc_all_rnns(batch)
        print("calc_all_rnns: %s seconds ---" % (time.time() - start_time))

        batch_size = batch.size(0)
        max_length = batch.size(1)

        # Dynamic programming algorithm for inference (with batching)
        best_scores = torch.zeros(batch_size, max_length)
        if self.is_cuda:
            best_scores = best_scores.cuda()
        segmentations = [[[0]] for _ in xrange(batch_size)]

        # Note: We don't use torch variables during the following dynamic programming
        # algorithm since we don't want to affect the computation graph
        for i in range(1, max_length):
            # Get scores of subsequences of seq[:i] that ends with i
            current_scores = torch.zeros(batch_size, i)
            if self.is_cuda:
                current_scores = current_scores.cuda()

            for j in range(i):
                current_scores[:, j] = best_scores[:, j] + self.get_local_score(batch, j, i)[:, 0].data
                #if gold_seg is not None:                
                #    current_scores[:, j] += self.get_local_task_loss(segmentations, gold_seg, j, i) 

            # Choose the best scores and the corresponding indexes
            max_scores, k = torch.max(current_scores, 1)
            k = k.cpu().numpy() # Convert indexes to numpy

            # Add current best score and best segmentation
            best_scores[:, i] = max_scores            
            for batch_index in range(batch_size):
                currrent_segmentation = segmentations[batch_index][k[batch_index]] + [i]
                # In case the gold segmentation is given and equals to our prediction, 
                # we will take the second best segmentation
                if gold_seg is not None and currrent_segmentation == gold_seg[batch_index]:
                    print "yayyyyyyyy!!!!!!!!!!!!!!\n\n\n"
                    second_best_k = torch.sort(current_scores[batch_index], 0, descending=True)[1][2]
                    currrent_segmentation = segmentations[batch_index][second_best_k] + [i]

                segmentations[batch_index].append(currrent_segmentation)

        # Get real segmentations according to the real lengths of the sequences 
        # in the batch
        final_segmentations = []
        for i, seg in enumerate(segmentations):
            last_index = lengths[i].data.cpu().numpy()[0] - 1
            final_segmentations.append(seg[last_index])
            
        # Get the scores of the best segmentations 
        final_scores = self.get_score(batch, final_segmentations) 
        
        # If gold seg' is given, add the task loss to the score (batch-wise)
        #if gold_seg is not None:
        #    final_scores += Variable(self.get_task_loss(final_segmentations, gold_seg))

        return final_segmentations, final_scores


    def get_normalization(self, batch, lengths):
        '''
        Get the sum of all the possible scores using practical dynamic
        programming algorithm. This is the partition function (Z(x)).

        Parameters:
            batch :  A 3D torch tensor: (batch_size, sequence_size, input_size)
            lengths: A 1D tensor containing the lengths of the batch sequences

        Notes:
            The algorithm complexity is O(n**2)
        '''
    
        # First, calculate all the possible segment-RNNs of the sequences in batch
        start_time = time.time()
        if self.SIMPLE_MODE:
            self.calc_birnn(batch)
        else:
            self.calc_all_rnns(batch)
        print("calc_all_rnns: %s seconds ---" % (time.time() - start_time))

        batch_size = batch.size(0)
        max_length = batch.size(1)

        # Dynamic programming algorithm for sum (with batching)
        # Note: We don't use torch variables during the following dynamic programming
        # algorithm since we don't want to affect the computation graph
        sums = Variable(torch.ones(batch_size, max_length)) * np.exp(-50)
        if self.is_cuda:
            sums = sums.cuda()

        # In each iteration, sums[i] will hold the sum of all the possible exponent-scores 
        # of the the subsequences that ends in index i-1 (i.e. sequence[:i]).
        for i in range(1, max_length):
            sums[:, i] = sum(sums[:, j].clone()*torch.exp(self.get_local_score(batch, j, i)[:, 0]) for j in range(i))

        # Now, each batch partition function is placed in sums[batch_index, batch_length], so
        # extract them
        sums_res = Variable(torch.zeros(batch_size, 1))
        if self.is_cuda:
            sums_res = sums_res.cuda()

        for i in range(batch_size):
            last_index = lengths[i].data.cpu().numpy()[0] - 1
            sums_res[i, 0] = sums[i, last_index]
    
        print sums_res
        return sums_res

    def store_params(self, fpath):
	   torch.save(self.state_dict(), fpath)

    def load_params(self, fpath):
	   self.load_state_dict(torch.load(fpath))

