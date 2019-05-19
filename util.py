'''
This file contains multiple functions that convert data to appropriate format, invoke code of fair classifiers and plot the results.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from cleanlab.classification import LearningWithNoisyLabels

import time
import pickle
import copy
import sys
import re
import os
from collections import namedtuple
from random import seed

from load_data import load_adult, load_compas, load_law, load_german, load_bank
from measures import fair_measure

import seaborn as sns
sns.set(font_scale = 1.5)

sys.path.insert(0, 'fairlearn/')
import classred as red
import moments

sys.path.insert(0, 'fair_classification/')
import utils as ut
import funcs_disp_mist as fdm
import loss_funcs as lf

sys.path.insert(0, 'fairERM/')
from linear_ferm import Linear_FERM


'''
A list of base classifiers that can be used by Agarwal's method.
'''
class LeastSquaresLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, W):
        sqrtW = np.sqrt(W)
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        pred = X.dot(self.weights)
        return 1*(pred > 0.5)


class LR:
    def __init__(self):
        self.clf = LogisticRegression(solver = 'lbfgs')
    def fit(self, X, Y, W):
        try:
            self.clf.fit(X.values, Y.values)
        except ValueError:
            pass

    def predict(self, X):
        try:
            return pd.Series(self.clf.predict(X.values))
        except NotFittedError:
            return pd.Series(np.zeros(X.values.shape[0]))


class SVM:
    def __init__(self):
        self.clf = SVC()
    def fit(self, X, Y, W):
        try:
            self.clf.fit(X.values, Y.values)
        except ValueError:
            pass

    def predict(self, X):
        try:
            return pd.Series(self.clf.predict(X.values))
        except NotFittedError:
            pred = np.random.random(X.values.shape[0])
            pred[pred > 0.5] = 1.0
            pred[pred <= 0.5] = 0.0
            return pd.Series(pred)


SEED = 1122334455
seed(SEED)
np.random.seed(SEED)


keys = ["disp_train", "disp_test", "error_train", "error_test"]




def change_format(dataset_train, dataset_test, sensible_feature, include_sensible):
    '''
    Change data format into that needed for Agarwal classifier. More preprocessing will be needed if Zafar's classifier gets used.
    '''
    d_train = dict()
    d_test = dict()

    for i in range(dataset_train.data.shape[1]):
        if i != sensible_feature or include_sensible:
            d_train[i] = dataset_train.data[:, i]
            d_test[i] = dataset_test.data[:, i]

    dataX = pd.DataFrame(d_train)
    dataY = pd.Series(dataset_train.target)

    dataA = pd.Series(dataset_train.data[:, sensible_feature])
    dataA[dataA>0] = 1
    dataA[dataA<=0] = 0

    dataX_test = pd.DataFrame(d_test)
    dataY_test = pd.Series(dataset_test.target)

    dataA_test = pd.Series(dataset_test.data[:, sensible_feature])
    dataA_test[dataA_test>0] = 1
    dataA_test[dataA_test<=0] = 0

    return [dataX, dataY, dataA, dataX, dataY, dataA, dataX_test, dataY_test, dataA_test]


def permute_and_split(datamat, permute=True, train_ratio=0.8):
    '''
    Permute and split dataset into training and testing.
    '''
    if permute:
        datamat = np.random.permutation(datamat)

    cutoff = int(np.floor(len(datamat)*train_ratio))

    dataset_train = namedtuple('_', 'data, target')(datamat[:cutoff, :-1], datamat[:cutoff, -1])
    dataset_test = namedtuple('_', 'data, target')(datamat[cutoff:, :-1], datamat[cutoff:, -1])

    return dataset_train, dataset_test



def corrupt(dataA, dataY, rho, creteria):
    '''
    Flip values in dataA with probability rho[0] and rho[1] for positive/negative values respectively.
    '''
    rho_a_plus, rho_a_minus = rho

    print('The number of data points belonging to each group:')
    print('before corruption:', np.sum(dataA==0), np.sum(dataA==1))
    for i in range(len(dataA)):
        rand = np.random.random()
        if dataA[i] == 1:
            if rand < rho_a_plus:
                dataA[i] = 0
        else:
            if rand < rho_a_minus:
                dataA[i] = 1

    print('after corruption:', np.sum(dataA==0), np.sum(dataA==1))



def estimate_alpha_beta(cor_dataA, dataY, rho, creteria):
    '''
    Estimate alpha and beta using rho and pi_a_corr.
    '''
    rho_a_plus, rho_a_minus = rho
    if (1 - rho_a_plus - rho_a_minus) < 0:
        print('before', rho_a_plus, rho_a_minus)
        norm = rho_a_plus+rho_a_minus
        rho_a_plus /= norm
        rho_a_minus /= norm
        print('after', rho_a_plus, rho_a_minus)

    pi_a_corr = None
    if creteria == 'EO':
        pi_a_corr = np.sum([1.0 if a > 0 and y > 0 else 0.0 for a, y in zip(cor_dataA, dataY)])/ np.sum([1.0 if y > 0 else 0.0 for y in dataY])
    else:
        pi_a_corr = np.mean([1.0 if a > 0 else 0.0 for a in cor_dataA])


    # To correct wrong estimation. pi_a cannot be negative
    rho_a_minus = np.min([pi_a_corr, rho_a_minus])

    pi_a = (pi_a_corr - rho_a_minus)/(1 - rho_a_plus - rho_a_minus)

    alpha_a = (1-pi_a)*rho_a_minus / pi_a_corr
    beta_a = pi_a*rho_a_plus / (1-pi_a_corr)

    if (1 - alpha_a - beta_a) < 0:
        print('The sum of alpha_a and beta_a is too large.', alpha_a, beta_a)
        print('We scale down them.')
        coeff = alpha_a+beta_a
        alpha_a = 0.95 * alpha_a / coeff
        beta_a = 0.95 * beta_a / coeff

    return alpha_a, beta_a


def _scale_eps(eps, alpha_a, beta_a):
    '''
    Scale down epsilon.
    '''
    return eps * (1 - alpha_a - beta_a)



def denoiseA(data_cor, rho, mode):
    '''
    Denoise the corrupted sensitive attribute using RankPrune.
    '''

    rho_a_plus, rho_a_minus = rho

    dataX = data_cor[0]
    cor_dataA = data_cor[2]
    # dataA = data_cor[5]
    #
    # auc3, auc4 = None, None



    noise_matrix = np.array([[1-rho_a_minus, rho_a_plus],[rho_a_minus, 1-rho_a_plus]])
    # noise_matrix = None

    lnl = LearningWithNoisyLabels(clf=LogisticRegression(random_state=0, solver = 'lbfgs', multi_class = 'auto'))
    lnl.fit(X = dataX.values, s = cor_dataA.values, noise_matrix=noise_matrix)


    # Logistic Regression Baseline
    # lnl = clf=LogisticRegression(random_state=0, solver = 'lbfgs', multi_class = 'auto')
    # lnl.fit(X = dataX.values, y = cor_dataA.values)


    denoised_dataA = pd.Series(lnl.predict(dataX.values))
    data_denoised = copy.deepcopy(data_cor)
    data_denoised[2] = denoised_dataA

    # print(lnl.noise_matrix, rho_a_plus, rho_a_minus)

    # Check recovery accuracy
    # auc1 = np.mean(dataA.values==cor_dataA.values)
    # auc2 = np.mean(dataA.values==denoised_dataA.values)


    # The following is under development.
    rho_est = None
    data_denoised_est = None


    if mode == 'six':

        lnl2 = LearningWithNoisyLabels(LogisticRegression(random_state=0, solver = 'lbfgs', multi_class = 'auto'))
        lnl2.fit(X = dataX.values, s = cor_dataA.values)

        denoised_dataA_est = pd.Series(lnl2.predict(dataX.values))
        data_denoised_est = copy.deepcopy(data_cor)
        data_denoised_est[2] = denoised_dataA_est

        rho_a_plus_est = lnl2.noise_matrix[0][1]
        rho_a_minus_est = lnl2.noise_matrix[1][0]
        rho_est = [rho_a_plus_est, rho_a_minus_est]

        # print(lnl2.noise_matrix, rho_a_plus_est, rho_a_minus_est)


        # lnl3 = LogisticRegression(random_state=0, solver = 'lbfgs', multi_class = 'auto')
        # lnl3.fit(dataX.values, cor_dataA.values)

        # pred_dataA = pd.Series(lnl3.predict(dataX.values))
        # auc3 = np.mean(dataA.values==denoised_dataA_est.values)
        # auc4 = np.mean(dataA.values==pred_dataA.values)

    # print('auc:', auc1, auc2, auc3, auc4)

    return data_denoised, data_denoised_est, rho_est


def experiment(dataset, frac, eval_objective, eps, rho_list, rho, eps_list, criteria, classifier, trials, include_sensible, filename, learner_name='lsq', mode='four', verbose=False):
    '''
    dataset: one of ['compas', 'bank', 'adult', 'law', 'german']. Default is 'compas'.
    frac: real number in interval [0, 1]. The fraction of the data points in chosen dataset to use.

    eval_objective: ['test_rho_est_err', 'test_tau']
    eps: a number specifying the wanted fairness level. Valid when eval_objective='test_rho_est_err'.
    rho_list: a list of (rho_plut, rho_minus) pairs. Valid when eval_objective='test_rho_est_err'.
    rho: [a, b] where a, b in interval [0,0.5].
    eps_list: a list of non-negative real numbers. Valid when eval_objective='test_eps'.

    criteria: one of ['DP','EO']
    classifier: one of ['Agarwal', 'Zafar']. Agarwal is the default.
    trials: the number of trials to run.
    include_sensible: boolean. If to include sensitive attribute as a feature for optimizing the oroginal loss. This is used only for debugging purpose. It is hard-coded to be False now.
    filename: the file name to store the log of experiment(s).
    learner_name: ['lsq', 'LR', 'SVM']. SVM is the slowest. lsq does not work for law school dataset but it works reasonally well on all other datasets.
    mode: ['four']. Currently, we only support four. Valid when eval_objective='test_eps'.
    verbose: boolean. If print out info at each run.
    '''

    # We hard-code mode and classifier.
    mode = 'four'

    # classifier
    if classifier not in ['Agarwal', 'Zafar']:
        classifier = 'Agarwal'

    # We hard-code include_sensible to False.
    include_sensible = False


    sensible_name = None
    sensible_feature = None
    learner = None
    print('input dataset:', dataset)
    if dataset == 'adult':
        datamat = load_adult(frac)
        sensible_name = 'gender'
        sensible_feature = 9
    elif dataset == 'law':
        datamat = load_law(frac)
        sensible_name = 'racetxt'
        sensible_feature = 9
        # lsq does not work for law
        learner_name = 'LR'
    elif dataset == 'german':
        datamat = load_german(frac)
        sensible_name = 'Foreign'
        sensible_feature = 21
    elif dataset == 'bank':
        datamat = load_bank(frac)
        sensible_name = 'Middle_Aged'
        sensible_feature = 7
    else:
        datamat = load_compas(frac)
        sensible_name = 'race'
        sensible_feature = 4


    if learner_name == 'LR':
        learner = LR()
    elif learner_name == 'SVM':
        learner = SVM()
    else:
        learner = LeastSquaresLearner()

    print('eval_objective', eval_objective)
    print('learner_name:', learner_name)

    if eval_objective == 'test_rho_est_err':
        eps_list = [eps for _ in range(len(rho_list))]

    if criteria == 'EO':
        tests = [{"cons_class": moments.EO, "eps": eps} for eps in eps_list]
    else:
        tests = [{"cons_class": moments.DP, "eps": eps} for eps in eps_list]

    if eval_objective == 'test_rho_est_err':
        all_data = _experiment_est_error(datamat, tests, rho, rho_list, trials, sensible_name, sensible_feature, criteria, classifier, include_sensible, learner, mode, verbose)
        _save_all_data(filename, all_data, rho_list)
    else:
        all_data = _experiment(datamat, tests, rho, trials, sensible_name, sensible_feature, criteria, classifier, include_sensible, learner, mode, verbose)
        _save_all_data(filename, all_data, eps_list)

    return all_data

def _experiment_est_error(datamat, tests, real_rho, rho_list, trials, sensible_name, sensible_feature, creteria, classifier, include_sensible, learner, mode, verbose):
    '''
    Internal rountine of running experiment. Run experiments under different settings using different algorithms and collect the results returned by the invoked fair classifiers.
    '''

    n = 4
    all_data = [{k:[[] for _ in range(trials)] for k in keys} for _ in range(n)]

    start = time.time()

    for i in range(trials):
        print('trial:', i, 'time:', time.time()-start)
        dataset_train, dataset_test = permute_and_split(datamat)
        data_nocor = change_format(dataset_train, dataset_test, sensible_feature, include_sensible)

        dataY = data_nocor[1]
        dataA = data_nocor[2]

        res_cor_cache = None
        res_nocor_cache = None

        for j in range(len(tests)):
            rho = rho_list[j]
            test_0 = tests[j]

            cor_dataA = dataA.copy()
            data_cor = copy.deepcopy(data_nocor)

            corrupt(cor_dataA, data_nocor[1], real_rho, creteria)
            data_cor[2] = cor_dataA


            data_denoised, _, _ = denoiseA(data_cor, rho, mode)


            alpha_a, beta_a = estimate_alpha_beta(cor_dataA, dataY, rho, creteria)

            eps_0 = test_0['eps']
            test = copy.deepcopy(test_0)

            if j == 0:
                res_cor = _run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)
                res_cor_cache = res_cor

                res_nocor = _run_test(test, data_nocor, sensible_name, learner,  creteria, verbose, classifier)
                res_nocor_cache = res_nocor
            else:
                res_cor = copy.deepcopy(res_cor_cache)
                res_nocor = copy.deepcopy(res_nocor_cache)

            res_denoised = _run_test(test, data_denoised, sensible_name, learner,  creteria, verbose, classifier)

            test['eps'] = _scale_eps(eps_0, alpha_a, beta_a)
            res_cor_scale = _run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

            results = [res_cor, res_nocor, res_denoised, res_cor_scale]

            for k in keys:
                for j in range(n):
                    all_data[j][k][i].append(results[j][k])

    return all_data

def _experiment(datamat, tests, rho, trials, sensible_name, sensible_feature, creteria, classifier, include_sensible, learner, mode, verbose):
    '''
    Internal rountine of running experiment. Run experiments under different settings using different algorithms and collect the results returned by the invoked fair classifiers.
    '''

    if mode == 'six':
        n = 6
    else:
        n = 4
    all_data = [{k:[[] for _ in range(trials)] for k in keys} for _ in range(n)]

    start = time.time()

    for i in range(trials):
        print('trial:', i, 'time:', time.time()-start)
        dataset_train, dataset_test = permute_and_split(datamat)
        data_nocor = change_format(dataset_train, dataset_test, sensible_feature, include_sensible)

        dataY = data_nocor[1]
        dataA = data_nocor[2]
        cor_dataA = dataA.copy()
        data_cor = copy.deepcopy(data_nocor)

        corrupt(cor_dataA, data_nocor[1], rho, creteria)
        data_cor[2] = cor_dataA


        data_denoised, data_denoised_est, rho_est = denoiseA(data_cor, rho, mode)


        alpha_a, beta_a = estimate_alpha_beta(cor_dataA, dataY, rho, creteria)

        if mode == 'six':
            alpha_a_est, beta_a_est = estimate_alpha_beta(cor_dataA, dataY, rho_est, creteria)

        for test_0 in tests:
            eps_0 = test_0['eps']
            test = copy.deepcopy(test_0)

            res_cor = _run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

            res_nocor = _run_test(test, data_nocor, sensible_name, learner,  creteria, verbose, classifier)

            res_denoised = _run_test(test, data_denoised, sensible_name, learner,  creteria, verbose, classifier)

            test['eps'] = _scale_eps(eps_0, alpha_a, beta_a)
            res_cor_scale = _run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

            results = [res_cor, res_nocor, res_denoised, res_cor_scale]

            if mode == 'six':
                test['eps'] = _scale_eps(eps_0, alpha_a_est, beta_a_est)
                res_cor_scale_est = _run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

                test['eps'] = eps_0
                res_denoised_est = _run_test(test, data_denoised_est, sensible_name, learner, creteria, verbose, classifier)

                results = [res_cor, res_nocor, res_denoised, res_cor_scale, res_denoised_est, res_cor_scale_est]

            for k in keys:
                for j in range(n):
                    all_data[j][k][i].append(results[j][k])

    return all_data


def _run_test(test, data, sensible_name, learner, creteria, verbose, classifier='Zafar'):
    '''
    Run a single trial of experiment using a chosen classifier.
    '''
    res = None
    if classifier == 'Agarwal':
        res = _run_test_Agarwal(test, data, sensible_name, learner, creteria)
    elif classifier == 'Shai':
        res = _run_test_Shai(test, data, sensible_name, learner, creteria)
    else:
        res = _run_test_Zafar(test, data, sensible_name, learner, creteria)

    if verbose:
        print("testing (%s, eps=%.5f)" % (test["cons_class"].short_name, test["eps"]))

        for k in keys:
            print(k+':', res[k], end=' ')
        print()

    return res


def _run_test_Shai(test, data, sensible_name, learner, creteria):
    '''
    Invoking Shai's algorithm.
    '''
    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data

    param_grid = [{'C': [0.01, 0.1, 1.0], 'kernel': ['linear']}]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, n_jobs=1)
    algorithm = Linear_FERM(dataX, dataA, dataY, clf, creteria)
    algorithm.fit()


    def _get_stats(clf, dataX, dataA, dataY):
        pred = algorithm.predict(dataX, dataA)
        disp = fair_measure(pred, dataA, dataY, creteria)
        error = 1 - accuracy_score(dataY, pred)
        return disp, error

    res = dict()
    res["disp_train"], res["error_train"] = _get_stats(algorithm, dataX_train, dataA_train, dataY_train)
    res["disp_test"], res["error_test"] = _get_stats(algorithm, dataX_test, dataA_test, dataY_test)

    return res


def _run_test_Agarwal(test, data, sensible_name, learner, creteria):
    '''
    Invoking Agarwal's algorithm.
    '''
    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data

    res_tuple = red.expgrad(dataX, dataA, dataY, learner,
                            cons=test["cons_class"](), eps=test["eps"], debug=False)

    res = res_tuple._asdict()
    Q = res["best_classifier"]

    def _get_stats(clf, dataX, dataA, dataY):
        pred = clf(dataX)

        disp = fair_measure(pred, dataA, dataY, creteria)
        error = np.mean(np.abs(dataY.values-pred.values))
        return disp, error

    res["disp_train"], res["error_train"] = _get_stats(Q, dataX_train, dataA_train, dataY_train)
    res["disp_test"], res["error_test"] = _get_stats(Q, dataX_test, dataA_test, dataY_test)

    return res




def _run_test_Zafar(test, data, sensible_name, learner, creteria):
    '''
    Invoking Zafar's algorithm.
    '''
    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data
    x, y, x_control, x_train, y_train, x_control_train, x_test, y_test, x_control_test = _convert_data_format_Zafar(data, sensible_name)

    w = None
    if creteria == 'EO':

        loss_function = "logreg" # perform the experiments with logistic regression
        EPS = 1e-6

        cons_type = 1 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
        tau = 5.0
        mu = 1.2
        sensitive_attrs_to_cov_thresh = {sensible_name: {0:{0:0, 1:test['eps']}, 1:{0:0, 1:test['eps']}, 2:{0:0, 1:test['eps']}}} # zero covariance threshold, means try to get the fairest solution
        cons_params = {"cons_type": cons_type,
                        "tau": tau,
                        "mu": mu,
                        "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}


        w = fdm.train_model_disp_mist(x, y, x_control, loss_function, EPS, cons_params)

    else:
        apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
        apply_accuracy_constraint = 0
        sep_constraint = 0

        loss_function = lf._logistic_loss
        sensitive_attrs = [sensible_name]
        # print('eps:',test['eps'])
        sensitive_attrs_to_cov_thresh = {sensible_name:test['eps']}

        gamma = None

        w = ut.train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    y_pred_train = np.sign(np.dot(x_train, w))
    disp_train = fair_measure(y_pred_train, dataA_train, dataY_train, creteria)
    train_score = accuracy_score(y_train, y_pred_train)

    y_pred_test = np.sign(np.dot(x_test, w))
    disp_test = fair_measure(y_pred_test, dataA_test, dataY_test, creteria)
    test_score = accuracy_score(y_test, y_pred_test)

    res = dict()
    res["disp_train"] = disp_train
    res["disp_test"] = disp_test
    res["error_train"] = 1 - train_score
    res["error_test"] = 1 - test_score

    return res



def _convert_data_format_Zafar(data, sensible_name):
    '''
    Convert the data format to that used by Zafar's fair classifier's interface.
    '''
    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data

    x = dataX.values
    y = dataY.values
    x_control = {sensible_name:copy.deepcopy(dataA.values).astype(int)}

    x_train = dataX_train.values
    y_train = dataY_train.values
    x_control_train = {sensible_name:copy.deepcopy(dataA_train.values).astype(int)}


    x_test = dataX_test.values
    y_test = dataY_test.values
    x_control_test = {sensible_name:copy.deepcopy(dataA_test.values).astype(int)}

    for y_tmp in [y, y_train, y_test]:
        y_tmp[y_tmp==0] = -1

    return x, y, x_control, x_train, y_train, x_control_train, x_test, y_test, x_control_test


def _summarize_stats(all_data, r=None):
    '''
    Calculate mean/std over runs for each combination of testing/training error/fairness violation
    '''
    n = len(all_data)
    curves_mean = [{k:None for k in keys} for _ in range(n)]
    curves_std = [{k:None for k in keys} for _ in range(n)]

    if not r:
        r = [0, len(all_data[0])]

    for k in keys:
        for i in range(n):
            c_s = curves_std[i]
            c_m = curves_mean[i]
            a_d = all_data[i]
            c_s[k] = [np.std(l) for l in zip(*a_d[k][r[0]:r[1]])]
            c_m[k] = [np.mean(l) for l in zip(*a_d[k][r[0]:r[1]])]


    return curves_mean, curves_std


def _save_all_data(filename, all_data, eps_list):
    '''
    Save the experiment's data into a file.
    '''
    tmp_all_data = copy.deepcopy(all_data)
    tmp_all_data.extend([eps_list])
    with open(filename, 'wb') as handle:
        pickle.dump(tmp_all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _restore_all_data(filename):
    '''
    Restore the experiment's data
    '''
    with open(filename, 'rb') as handle:
        all_data = pickle.load(handle)
    var_list = all_data[-1]
    all_data = all_data[:-1]

    return all_data, var_list


def plot(filename, eval_objective, ref_end=0.2, ref_line=[False, False, False, False], save=False):
    '''
    filename: str. The name of the file storing data used for plotting.
    eval_objective:
    ref_end: positive real number. the endding point of the ref line. It is only applicable when ref_line contains value True.
    ref_line: a list of booleans with length four. This controls if adding ref line to the generated graphs.
    save: boolean. If to save the plotted graphs.
    '''
    y_label = 'DDP'
    p_eo = re.compile('EO')
    if p_eo.search(filename):
        y_label = 'DEO'

    all_data, var_list = _restore_all_data(filename)


    # var_list is rho_list when eval_objective = 'test_rho_est_err'
    # var_list is eps_list when eval_objective = 'test_tau'
    data = _summarize_stats(all_data)

    xlabels = ['$\\tau$' for _ in range(4)]
    ylabels = [y_label, y_label, 'Error%', 'Error%']
    leg_pos_list = ['upper left', 'upper left', 'lower left', 'lower left']
    if eval_objective == 'test_rho_est_err':
        leg_pos_list = ['lower right', 'lower right', 'upper right', 'upper right']

    if eval_objective == 'test_rho_est_err':
        var_list = np.array(var_list)
        if var_list[:, 1][-1] == 0:
            var_list = np.array(var_list)[:, 0]
        else:
            var_list = np.array(var_list)[:, 1]
        xlabels = ['$\\hat{\\rho}^-$' for _ in range(4)]

    for k, xl, yl, leg_pos, ref in zip(keys, xlabels, ylabels, leg_pos_list , ref_line):
        title_end = ''
        if 'train' in k:
            title_end = '(training)'
        _plot(var_list, data, k, xl, yl, leg_pos, filename, ref, ref_end, save, title_end)


def _plot(var_list, data, k, xl, yl, leg_pos, filename, ref, ref_end, save, title_end):
    '''
    Plot four graphs. Internal routine for plot
    '''
    curves_mean, curves_std = data
    labels = ['cor', 'nocor', 'denoise', 'cor_scale', 'cor_scale_est', 'denoise_est']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = [':', ':', '--', '-']

    fig, ax = plt.subplots()

    # labels = ['cor', 'nocor', 'cor_scale']
    for stat, err, label, color, linestyle in zip(curves_mean, curves_std, labels, colors, linestyles):
            ax.errorbar(var_list, stat[k], yerr=err[k], label=label.replace('_', ' '), color=color, linestyle=linestyle)
    if ref:
        ax.plot([0, ref_end], [0, ref_end], 'k-', alpha=0.75, zorder=0, color='grey', linestyle='dashed')

    title_content = None
    try:
        p0 = re.compile('all\_data\_(.*)\.pickle')
        save_file_content = p0.search(filename).group(1)

        p = re.compile('all\_data\_([a-zA-Z]+),([0-9\.]+),([0-9\.]+),[0-9\.]+,([A-Z]{2}),[a-zA-Z]+,[0-9]+,[a-zA-Z]+,[\_a-zA-Z]+\.pickle')
        res = p.search(filename)
        dataset, rho_a_plus, rho_a_minus, creteria = res.group(1), res.group(2), res.group(3), res.group(4)

        # Modify a bit name shown on the title for the following two datasets
        if dataset == 'law':
            dataset = 'Law'
        elif dataset == 'compas':
            dataset = 'COMPAS'

        title_content = dataset+' '+creteria+', '+'$\\rho^+=$'+rho_a_plus+', $\\rho^-=$'+rho_a_minus+title_end

    except TypeError:
        pass

    ax.legend(loc=leg_pos, framealpha=0.1)
    ax.set_title(title_content)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    plt.show()

    save_dir = 'new_imgs/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if save:
        fig.savefig(save_dir+k+'_'+save_file_content+'.pdf', bbox_inches='tight')
