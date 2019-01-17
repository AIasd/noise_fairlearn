import numpy as np

import os
import argparse
from random import seed

from util import experiment, plot


SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

log_dir = 'experiment_log/'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='compas', help='dataset to run')
parser.add_argument('--rho-plus', dest='rho_plus', type=float, default=0.2, help='rho+ (default: 0.2)')
parser.add_argument('--rho-minus', dest='rho_minus', type=float, default=0.2, help='rho- (default: 0.2)')
parser.add_argument('--frac', dest='frac', type=float, default=1, help='rho- (default: 1)')
parser.add_argument('--criteria', dest='criteria', type=str, default='DP', help='fairness criteria')
parser.add_argument('--classifier', type=str, default='Agarwal', help='fairness classifier')
parser.add_argument('--trials', type=int, default=3, help='number of trials')

parser.add_argument('--include-sensible', dest='include_sensible', action='store_true', help='include sensible as a training feature')

parser.add_argument('--verbose', dest='verbose', action='store_true', help='print stats at each run')

parser.add_argument('--plot-result', dest='plot_result', action='store_true', help='plot result')

args = parser.parse_args()

dataset = args.dataset
rho = [args.rho_plus, args.rho_minus]
frac = args.frac
learner_name = None
if dataset == 'adult' or dataset == 'adultr':
    eps_list = [0.001 * i for i in range(1, 10)]
    learner_name = 'lsq'
elif dataset == 'compas':
    eps_list = [0.02 * i for i in range(1, 10)]
    learner_name = 'lsq'
elif dataset == 'law':
    eps_list = [0.02 * i for i in range(1, 10)]
    learner_name = 'SVM'
criteria = args.criteria
classifier = args.classifier
trials = args.trials
include_sensible = args.include_sensible
filename = log_dir+'all_data_'+dataset+','+str(rho[0])+','+str(rho[1])+','+str(frac)+','+criteria \
           +','+classifier+','+str(trials)+','+str(include_sensible)+'.pickle'
verbose = args.verbose

all_data = experiment(dataset, rho, frac, eps_list, criteria, classifier, trials, include_sensible, filename, learner_name, verbose)

if args.plot_result:
    plot(filename)
