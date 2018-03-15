import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Max length of segment - in indexes
MAX_SEGMENT_SIZE = 70
DEFAULT_FEATURE_SIZE = 20


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
    def __init__(self, rnn_input_dim=DEFAULT_FEATURE_SIZE,
                 rnn_output_dim=50, sum_mlp_hid_dims=(100, 100),
                 output_mlp_hid_dim=100, is_cuda=True, use_srnn=False,
                 use_task_loss=False, task_loss_coef=0.01, load_from_file=''):

        super(SpeechSegmentor, self).__init__()

        # Use BiRNN and summation instead of segmental RNN (much faster)
        self.SUM_MODE = not use_srnn

        self.rnn_output_dim = rnn_output_dim

        self.is_cuda = is_cuda

        self.use_task_loss = use_task_loss

        self.task_loss_coef = task_loss_coef

        # Network parameters:

        # BiLSTM (2D LSTM)
        self.BiRNN = nn.LSTM(rnn_input_dim, rnn_output_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)

        # Forward RNN (for segmental RNN)
        self.RNN_F = nn.LSTM(rnn_input_dim, rnn_output_dim, num_layers=1, batch_first=True, dropout=0.3)

        # Backward RNN (for segmental RNN)
        self.RNN_B = nn.LSTM(rnn_input_dim, rnn_output_dim, num_layers=1, batch_first=True, dropout=0.3)

        if self.SUM_MODE:
            # Sum MLP hidden layers (we use them only in sum mode)
            self.mlp_linear1 = nn.Linear(2 * rnn_output_dim, sum_mlp_hid_dims[0])
            self.mlp_linear2 = nn.Linear(sum_mlp_hid_dims[0], sum_mlp_hid_dims[1])
            # MLP output layer (gets as input concatenation of 2 BiRNN outputs plus the sum MLP output)
            self.mlp_output1 = nn.Linear(4 * rnn_output_dim + sum_mlp_hid_dims[1], output_mlp_hid_dim)
        else:
            # This is due the segmental RNN concatenation in phi
            self.mlp_output1 = nn.Linear(6 * rnn_output_dim, output_mlp_hid_dim)

        # We return a scalar score
        self.mlp_output2 = nn.Linear(output_mlp_hid_dim, 1)

        # MLP activation function
        self.mlp_activation = nn.ReLU()

        # Make all the params cuda tensors
        if is_cuda:
            self.cuda()

        # If given - load the model's parameters from a file
        if load_from_file:
            self.load_params(load_from_file)


    def calc_birnn_sums(self, batch, lengths):
        '''
        Get a batch with sequences and calculate its 2-layer BiRNN.
        Parameters:
            batch : A 3D torch tensor (batch_size, sequence_size, input_size)

        Returns:
             self.BiRNN_sums - will be used to get the local score of segments:
                               local_score[i,j] = BiRNN_sums[j] - BiRNN_sums[i]

        '''
        batch_size   = batch.size(0)
        seq_length   = batch.size(1)
        input_length = batch.size(2)

        # Create random hidden states
        # The 4 is due to a bidirectional RNN with two layers (2x2=4)
        hidden = (Variable(torch.zeros(4, batch_size, self.rnn_output_dim)), 
                  Variable(torch.zeros(4, batch_size, self.rnn_output_dim)))

        if self.is_cuda:
            hidden = hidden[0].cuda(), hidden[1].cuda()

        packed_seq, sorted_idx = sort_and_pack(batch, lengths)
        out, hidden = self.BiRNN(packed_seq, hidden)
        out = unpack_and_unsort(out, sorted_idx)

        # Calc the accumulator of the output (i.e. self.BiRNN_sums[:,i,:] will be the sum of 
        # out[:, 0:i, :])
        self.BiRNN_sums = out.clone()
        for i in range(1, seq_length):
            self.BiRNN_sums[:, i, :] = self.BiRNN_sums[:, i-1, :] + out[:, i, :]

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

    def mlp_sum_layer(self, input):
        ''' 
        Propagate the given vector through a MLP with one hidden layers.
        '''
        hidden1 = self.mlp_activation(self.mlp_linear1(input))
        return self.mlp_linear2(hidden1)

    def mlp_output_layer(self, input):
        ''' 
        Propagate the given vector through a one hidden layer MLP with scalar output.
        '''
        hidden = self.mlp_activation(self.mlp_output1(input))
        return self.mlp_output2(hidden)

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
            (i.e. phi(seq, y_i, y_i+1)) and feed it into a MLP that produces scalar output (score).
        '''

        seq_length = batch.size(1)

        try:
            if self.SUM_MODE:
                if y_start == 0:
                    # Get birnn sum over the segment
                    birnn_sum = self.BiRNN_sums[:, y_end, :]
                    # Get BiRNN(seq, y_start)
                    birnn_y_start = self.BiRNN_sums[:, y_start, :]
                else:
                    # Get birnn sum over the segment
                    birnn_sum = self.BiRNN_sums[:, y_end, :] - self.BiRNN_sums[:, y_start-1, :]
                    # Get BiRNN(seq, y_start)
                    birnn_y_start = self.BiRNN_sums[:, y_start, :] - self.BiRNN_sums[:, y_start-1, :]

                # Get BiRNN(seq, y_end)
                birnn_y_end = self.BiRNN_sums[:, y_end, :] - self.BiRNN_sums[:, y_end-1, :]

                # Concatenate the BiRNNs with the MLP of the RNNs sum - 
                # this is the actual Phi function
                features = torch.cat((birnn_y_start, birnn_y_end, self.mlp_sum_layer(birnn_sum)),  
                                      dim=1)

            else:         
            
                # Get BiRNN(seq, y_start)
                birnn_y_start = torch.cat((self.forward_rnns_dict[0, y_start], 
                                           self.backward_rnns_dict[seq_length - 1, y_start]),
                                           dim=1)

                # Get BiRNN(seq, y_end)
                birnn_y_end = torch.cat((self.forward_rnns_dict[0, y_end], 
                                         self.backward_rnns_dict[seq_length - 1, y_end]),
                                         dim=1)

                # Concatenate the BiRNNs with RNN_F and RNN_B of the segment -
                # this is the actual Phi function
                features = torch.cat((birnn_y_start, 
                                      birnn_y_end,
                                      self.forward_rnns_dict[y_start, y_end], 
                                      self.backward_rnns_dict[y_end, y_start]), 
                                      dim=1)

        except AttributeError:
            func_name = 'calc_birnn_sum' if self.SUM_MODE else 'calc_all_rnns'
            raise NotFoundRNNsError('"%s" must be called before calculating score or forward()ing!') % func_name

        # Get the scalar score
        return self.mlp_output_layer(features)

    def get_score(self, batch, segmentations):
        '''
        Get the score of the given segmentations of the sequences in 'batch'.

        Parameters:        
            batch : A 3D torch tensor (batch_size, sequence_size, input_size)
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

        # Get distance of the new index for each element in the batch
        for batch_index in range(batch_size):

            # Amount of predicted points so far
            gold_seg = gold_segmentations[batch_index]

            # get the distance from the closest point to new_y
            min_dist = min(abs(new_y - x) for x in gold_seg)
            task_losses[batch_index] = max(0, min_dist - 3)

        return task_losses

    def get_task_loss(self, pred_segmentations, gold_segmentations):

        batch_size = len(pred_segmentations)
        task_losses = torch.zeros(batch_size)

        # Calc loss
        for batch_index in range(batch_size):
            current_pred_seg = pred_segmentations[batch_index]
            current_gold_seg = gold_segmentations[batch_index]

            for i,y in enumerate(current_pred_seg):

                min_dist = min(abs(y - x) for x in current_gold_seg)

                task_losses[batch_index] += max(0, min_dist - 3)

        return task_losses

    def forward(self, batch, lengths, gold_seg=None):
        '''
        Get the segmentation with the highest score using a practical dynamic
        programming algorithm.

        Parameters:
            batch :     A 3D torch tensor: (batch_size, sequence_size, input_size)
            lengths:    A 1D tensor containing the lengths of the batch sequences
            [gold_seg]: A python list containing batch_size lists with the gold
                        segmentations. If given, we will return the best segmentation
                        excluding the gold one, for the structural hinge loss with 
                        margin algorithm (see Kiperwasser, Eliyahu, and Yoav Goldberg
                        "Simple and accurate dependency parsing using bidirectional LSTM feature representations).

        Notes:
            The algorithm complexity is O(n**2)
        '''
    
        # First, calculate all the possible segment-RNNs or sum-RNNs of the sequences in batch
        start_time = time.time()
        if self.SUM_MODE:
            self.calc_birnn_sums(batch, lengths)
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
            current_scores = torch.zeros(batch_size, min(i, MAX_SEGMENT_SIZE))
            if self.is_cuda:
                current_scores = current_scores.cuda()

            start_index = max(0, i-MAX_SEGMENT_SIZE)
            for j in range(start_index, i):
                current_scores[:, j-start_index] = best_scores[:, j] + self.get_local_score(batch, j, i)[:, 0].data
                # Add local task loss
                if self.use_task_loss and gold_seg is not None:
                    segment_task_loss = self.get_local_task_loss(segmentations, gold_seg, j, i)                
                    if self.is_cuda:
                        segment_task_loss = segment_task_loss.cuda()
                    current_scores[:, j-start_index] += self.task_loss_coef * segment_task_loss 
        
            # Choose the best scores and their corresponding indexes
            max_scores, k = torch.max(current_scores, 1)
            k =  start_index + k.cpu().numpy() # Convert indexes to numpy (relative to the starting index)

            # Add current best score and best segmentation
            best_scores[:, i] = max_scores            
            for batch_index in range(batch_size):
                currrent_segmentation = segmentations[batch_index][k[batch_index]] + [i]
                # In case the gold segmentation is given and equals to our prediction, 
                # we will take the second best segmentation
                if gold_seg is not None and tuple(currrent_segmentation) == tuple(gold_seg[batch_index]):
                    print "Got gold segmentation!!!\n\n"
                    print "Current segmentation: ", currrent_segmentation
                    print "Gold segmentation: ", gold_seg[batch_index]
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
        if self.use_task_loss and gold_seg is not None:
            task_loss = Variable(self.get_task_loss(final_segmentations, gold_seg))
            if self.is_cuda:
                task_loss = task_loss.cuda()
            final_scores += self.task_loss_coef * task_loss

        return final_segmentations, final_scores

    def store_params(self, fpath):
        torch.save(self.state_dict(), fpath)

    def load_params(self, fpath):
        self.load_state_dict(torch.load(fpath))

