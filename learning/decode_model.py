#!/usr/bin/python

import argparse
import sys

import numpy as np
import torch
from torch.autograd import Variable

from model.model import SpeechSegmentor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

plt.rcParams.update({'mathtext.default':  'regular' })
#from mpl_toolkits import mplot3d
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns

sys.path.append('./back_end')
from data_handler import (preaspiration_dataset,
                          switchboard_dataset_after_embeddings, timit_dataset,
                          toy_dataset)

from data_handler import ALL_PHONEMES


def convert_to_batches(data, batch_size, is_cuda, fixed_k):
    """
    Data: list of tuples each from the format: (tensor, label)
          where the tensor should be 2d contatining features
          for each time frame.  

    Output: A list contating tuples from the format: (batch_tensor, lengths, phonemes)      
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

        # Get tensors, phonemes and the segmentation
        tensors  = [ex[0] for ex in data[i:i+cur_batch_size]]
        segments = [ex[1] for ex in data[i:i+cur_batch_size]]
        # phonemes
        phonemes = [ex[2] for ex in data[i:i+cur_batch_size]]

        k = len(segments[0])
        num_of_segments = k+1

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

        # Add it to the batches list along with the real lengths and phonemes
        batches.append((padded_batch, lengths, segments, phonemes))

        # Progress i for the next batch
        i += cur_batch_size

    return batches


def plot_unary_scores(model, batches, position=0, num_of_frames=150):
    # Calc RNN of the first batch
    batch, lengths, segmentations, phonemes = batches[0]
    model.calc_birnn_sums(batch, lengths)

    num_of_frames = min(num_of_frames, batch.size(1))

    # Score for the first position
    scores = model.get_unary_scores(batch, position=position)
    scores = scores[:num_of_frames]

    xposition = [t for t in segmentations[0] if t < num_of_frames]

    # Add scores to plot
    plt.plot(scores, label='scores', linewidth=2.5)

    # Add gold boundaries
    for xc in xposition:
        plt.axvline(x=xc, color='r', linestyle='dotted')

    plt.xlabel('Frame number (i)', fontsize=14)
    if position == 0:
        plt.ylabel('$s_l(i)$', fontsize=14)
    else:
        plt.ylabel('$s_r(i)$', fontsize=14)
    
    # Turn off y labels
    #plt.yticks([])

    fname = "{}.pdf".format("s_l_scores" if position==0 else "s_r_scores")
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_binary_scores(model, batches, num_of_frames=150):

    # Calc RNN of the first batch
    batch, lengths, segmentations, phonemes = batches[0]
    model.calc_birnn_sums(batch, lengths)

    num_of_frames = min(num_of_frames, batch.size(1))

    # get scores for all the possible (i,j) couples
    scores = model.get_binary_scores(batch)
    scores = scores[:num_of_frames, :num_of_frames]
    seg = [t for t in segmentations[0] if t < num_of_frames]

    # Creatre heatmap - ignore uninizilized values (zeros)
    sns.set()
    sns.heatmap(scores, mask=(scores==0))

    # Plot gold segmentation points
    plt.plot(seg[:-1], seg[1:], 'ro',  markersize=5)

    # Info
    plt.title('$s_{lr}(y_i,y_{i+1})$')
    plt.xlabel('$y_i$', fontsize=14)
    plt.ylabel('$y_{i+1}$', fontsize=14)

    plt.savefig('slr_scores_limited.pdf', bbox_inches='tight')
    plt.show()

def plot_sums_tsne(model, batches):
    # Calc RNN of the first batch

    tsne, phonemes = model.get_sums_tsne(batches[:10])

    colors = cm.rainbow(np.linspace(0, 1, 10))

    # Draw t-sne according to the phoneme 
    for i in range(len(tsne)):
        plt.plot(tsne[i, 0], tsne[i, 1], 'ro', color=colors[int(phonemes[i])], markersize=5)

    # Build color map legend
    patches = []
    for j,c in enumerate(colors):
        patches.append(mpatches.Patch(color=c, label=ALL_PHONEMES[j]))

    plt.legend(handles=patches)
    plt.show()


def decode_data(model, dataset_name, dataset, batch_size, is_cuda, use_k):

    labels = []
    predictions = []

    # Convert data to torch batches and loop over them
    batches = convert_to_batches(dataset, batch_size, is_cuda, use_k)

    # Experiments
    #####################################################
    #plot_sums_tsne(model, batches)
    plot_binary_scores(model, batches)
    plot_unary_scores(model, batches, position=0)
    plot_unary_scores(model, batches, position=1)
    exit()
    #####################################################

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


