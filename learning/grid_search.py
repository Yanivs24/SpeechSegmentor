import argparse
import itertools
import os
import time
from train_model import main
import json

parser = argparse.ArgumentParser()
parser.add_argument("train_path", help="A path to the training set")
# parser.add_argument("params_path", help="A path to a file in which the trained model parameters will be stored")
parser.add_argument("--val_path", help="A path to the training set", default=None)
parser.add_argument("--dataset", help="Which dataset to use: sb(switchboard)/pa/toy/timit/vot/word/vowel", default='sb')
parser.add_argument('--learning_rate', help='The learning rate', default=0.0001, type=float)
parser.add_argument('--num_iters', help='Number of iterations (epochs)', default=5000, type=int)
parser.add_argument('--batch_size', help='Size of training batch', default=20, type=int)
parser.add_argument('--patience', help='Num of consecutive epochs to trigger early stopping', default=5, type=int)
parser.add_argument('--use_cuda',  help='disables training with CUDA (GPU)', action='store_true', default=False)
parser.add_argument("--init_params", help="Start training from a set of pretrained parameters", default='')
parser.add_argument('--use_task_loss', help='Train with strucutal loss using task loss (always on when k is known)', action='store_true', default=True)
parser.add_argument('--use_k', help='Apply inference when k (num of segments) is known for each example', action='store_true', default=True)
parser.add_argument('--task_loss_coef', help='Task loss coefficient', default=0.0001, type=float)
parser.add_argument('--grad_clip', help='gradient clipping', default=5, type=float)
parser.add_argument('--max_segment_size', help='Max searched segment size (in indexes)', default=52, type=int)
parser.add_argument('--init_lstm_params', help='Load pretrained LSTM weights and used them as a fixed embedding layer', default='')

args = parser.parse_args()
dargs = vars(args)

# create folder for the new experiment
experiments_folder = 'experiments'
experiment_name = "{}_{}_{}".format(args.dataset, time.strftime("%H_%M_%S"), time.strftime("%d_%m_%Y"))
experiment_folder = os.path.join(experiments_folder, experiment_name)
if not os.path.exists(experiments_folder):
    os.mkdir(experiments_folder)
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

# create grid
grids = {'learning_rate': [0.001, 0.0001],
         'use_k': [True],
         'task_loss_coef': [0.001, 0.0001]}
cartesian_product = (dict(zip(grids, x)) for x in itertools.product(*grids.values()))

# search the grid
for sub_exp_idx, combination in enumerate(cartesian_product):
    sub_exp_name = os.path.join(experiment_folder, "exp_{}".format(sub_exp_idx))
    for k, v in combination.items():
        dargs[k] = v
    dargs['params_path'] = sub_exp_name + ".model"
    metrics_df = main(args)
    metrics_df.to_csv(sub_exp_name + ".csv", sep='\t')
    with open(sub_exp_name + ".txt", 'w') as f:
        f.write(json.dumps(dargs))
    print("==> done")
