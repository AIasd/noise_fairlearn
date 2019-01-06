'''
This file contains multiple functions that convert data to appropriate format, invoke code in Agarwal classifier/Zafar classifier and plot according to results.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import time
import pickle
import copy
import sys
import re
from collections import namedtuple
from random import seed

from measures import fair_measure

sys.path.insert(0, 'fairlearn/')
import classred as red

sys.path.insert(0, 'fair_classification/')
import utils as ut
import funcs_disp_mist as fdm
import loss_funcs as lf

sys.path.insert(0, 'fairERM/')
from linear_ferm import Linear_FERM



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

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

learner = LeastSquaresLearner()

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


def corrupt(dataA, rho):
    '''
    Flip values in dataA with probability rho[0] and rho[1] for positive/negative values respectively.
    '''

    pi_a = np.mean([1.0 if a > 0 else 0.0 for a in dataA])
    rho_a_plus = rho[0]
    rho_a_minus = rho[1]

    pi_a_corr = (1-rho_a_plus)*pi_a + rho_a_minus*(1-pi_a)

    alpha_a = (1-pi_a)*rho_a_minus / pi_a_corr
    beta_a = pi_a*rho_a_plus / (1-pi_a_corr)
    c = 0
    # print(np.sum(dataA==0), np.sum(dataA==1))
    for i in range(len(dataA)):
        rand = np.random.random()
        if dataA[i] == 1:
            if rand < rho_a_plus:
                c += 1
                dataA[i] = 0
        else:
            if rand < rho_a_minus:
                c -= 1
                dataA[i] = 1
    # print(c, len(dataA))
    # print(np.sum(dataA==0), np.sum(dataA==1))
    return alpha_a, beta_a





def _experiment(datamat, tests, rho, trials, sensible_name, sensible_feature, creteria, classifier, include_sensible, verbose):
    all_stats_cor = {k:[[] for _ in range(trials)] for k in keys}
    all_stats_nocor = {k:[[] for _ in range(trials)] for k in keys}
    all_stats_cor_scale = {k:[[] for _ in range(trials)] for k in keys}


    start = time.time()

    for i in range(trials):
        print('trial:', i, 'time:', time.time()-start)
        dataset_train, dataset_test = permute_and_split(datamat)

        data_nocor = change_format(dataset_train, dataset_test, sensible_feature, include_sensible)
        cor_dataA = copy.deepcopy(data_nocor[2])
        alpha_a, beta_a = corrupt(cor_dataA, rho=rho)

        data_cor = copy.deepcopy(data_nocor)
        data_cor[2] = cor_dataA


        for test in tests:
            res_cor = run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

            res_nocor = run_test(test, data_nocor, sensible_name, learner,  creteria, verbose, classifier)

            test['eps'] *= (1-alpha_a-beta_a)
            res_cor_scale = run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

            for k in keys:
                all_stats_cor[k][i].append(res_cor[k])
                all_stats_nocor[k][i].append(res_nocor[k])
                all_stats_cor_scale[k][i].append(res_cor_scale[k])

    all_data = [all_stats_cor, all_stats_nocor, all_stats_cor_scale]


    return all_data


def run_test(test, data, sensible_name, learner, creteria, verbose, classifier='Zafar'):
    res = None
    if classifier == 'Agarwal':
        res = run_test_Agarwal(test, data, sensible_name, learner, creteria)
    elif classifier == 'Shai':
        res = run_test_Shai(test, data, sensible_name, learner, creteria)
    else:
        res = run_test_Zafar(test, data, sensible_name, learner, creteria)

    if verbose:
        print("testing (%s, eps=%.5f)" % (test["cons_class"].short_name, test["eps"]))

        for k in keys:
            print(k+':', res[k], end=' ')
        print()

    return res


def run_test_Shai(test, data, sensible_name, learner, creteria):
    # Standard SVM -  Train an SVM using the training set
    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data

    param_grid = [{'C': [0.01, 0.1, 1.0], 'kernel': ['linear']}]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid, n_jobs=1)
    algorithm = Linear_FERM(dataX, dataA, dataY, clf, creteria)
    algorithm.fit()


    def _get_stats(clf, dataX, dataA, dataY):
        pred = algorithm.predict(dataX, dataA)
        disp = fair_measure(pred, dataX, dataA, dataY, creteria)
        error = 1 - accuracy_score(dataY, pred)
        return disp, error

    res = dict()
    res["disp_train"], res["error_train"] = _get_stats(algorithm, dataX_train, dataA_train, dataY_train)
    res["disp_test"], res["error_test"] = _get_stats(algorithm, dataX_test, dataA_test, dataY_test)

    return res


def run_test_Agarwal(test, data, sensible_name, learner, creteria):

    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data

    res_tuple = red.expgrad(dataX, dataA, dataY, learner,
                            cons=test["cons_class"](), eps=test["eps"])

    res = res_tuple._asdict()
    Q = res["best_classifier"]

    def _get_stats(clf, dataX, dataA, dataY):
        pred = clf(dataX).values
        for i in range(len(pred)):
            if np.random.random() < pred[i]:
                pred[i] = 1.0
            else:
                pred[i] = 0.0
        pred = pd.Series(pred)
        disp = fair_measure(pred, dataX, dataA, dataY, creteria)

        error = 1 - accuracy_score(dataY, pred)
        return disp, error

    res["disp_train"], res["error_train"] = _get_stats(Q, dataX_train, dataA_train, dataY_train)
    res["disp_test"], res["error_test"] = _get_stats(Q, dataX_test, dataA_test, dataY_test)

    return res

# The following measure
# def run_test_Agarwal(test, data, sensible_name, learner, creteria):
#
#     dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data
#
#     res_tuple = red.expgrad(dataX, dataA, dataY, learner,
#                             cons=test["cons_class"](), eps=test["eps"])
#
#     res = res_tuple._asdict()
#
#     Q = res["best_classifier"]
#     res["n_classifiers"] = len(res["classifiers"])
#
#     disp = test["cons_class"]()
#     disp.init(dataX_train, dataA_train, dataY_train)
#
#     disp_test = test["cons_class"]()
#     disp_test.init(dataX_test, dataA_test, dataY_test)
#
#     error = moments.MisclassError()
#     error.init(dataX_train, dataA_train, dataY_train)
#
#     error_test = moments.MisclassError()
#     error_test.init(dataX_test, dataA_test, dataY_test)
#
#     res["disp_train"] = disp.gamma(Q).max()
#     res["disp_test"] = disp_test.gamma(Q).max()
#     res["error_train"] = error.gamma(Q)[0]
#     res["error_test"] = error_test.gamma(Q)[0]
#
#     return res


def run_test_Zafar(test, data, sensible_name, learner, creteria):
    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data
    x, y, x_control, x_train, y_train, x_control_train, x_test, y_test, x_control_test = convert_data_format_Zafar(data, sensible_name)

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
    disp_train = fair_measure(y_pred_train, dataX_train, dataA_train, dataY_train, creteria)
    train_score = accuracy_score(y_train, y_pred_train)

    y_pred_test = np.sign(np.dot(x_test, w))
    disp_test = fair_measure(y_pred_test, dataX_test, dataA_test, dataY_test, creteria)
    test_score = accuracy_score(y_test, y_pred_test)

    res = dict()
    res["disp_train"] = disp_train
    res["disp_test"] = disp_test
    res["error_train"] = 1 - train_score
    res["error_test"] = 1 - test_score

    return res


# def run_test_Zafar(test, data, sensible_name, learner, creteria):
#
#     x, y, x_control, x_train, y_train, x_control_train, x_test, y_test, x_control_test = convert_data_format_Zafar(data, sensible_name)
#     sensitive_attrs = [sensible_name]
#
#     loss_function = "logreg" # perform the experiments with logistic regression
#     EPS = 1e-6
#
#     cons_type = 1 # FPR constraint -- just change the cons_type, the rest of parameters should stay the same
#     tau = 5.0
#     mu = 1.2
#     sensitive_attrs_to_cov_thresh = {sensible_name: {0:{0:0, 1:test['eps']}, 1:{0:0, 1:test['eps']}, 2:{0:0, 1:test['eps']}}} # zero covariance threshold, means try to get the fairest solution
#     cons_params = {"cons_type": cons_type,
#                     "tau": tau,
#                     "mu": mu,
#                     "sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}
#
#
#     def _train_test_classifier():
#         w = fdm.train_model_disp_mist(x, y, x_control, loss_function, EPS, cons_params)
#
#
#
#         train_score, test_score, cov_all_train, cov_all_test, s_attr_to_fp_fn_train, s_attr_to_fp_fn_test = fdm.get_clf_stats(w, x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attrs)
#         disp_test = np.abs(s_attr_to_fp_fn_test[sensible_name][0]["fpr"] - s_attr_to_fp_fn_test[sensible_name][0]["fpr"])
#         disp_train = np.abs(s_attr_to_fp_fn_train[sensible_name][0]["fpr"] - s_attr_to_fp_fn_train[sensible_name][0]["fpr"])
#
#
#         # accuracy and FPR are for the test because we need of for plotting
#         return disp_test, disp_train, test_score, train_score
#
#
#     disp_test, disp_train, test_score, train_score  = _train_test_classifier()
#     res = dict()
#     res["disp_train"] = disp_train
#     res["disp_test"] = disp_test
#     res["error_train"] = 1-train_score
#     res["error_test"] = 1-test_score
#
#     return res



def convert_data_format_Zafar(data, sensible_name):
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








def summarize_stats(all_data, r=None):
    all_stats_cor, all_stats_nocor, all_stats_cor_scale = all_data

    if not r:
        r = [0, len(all_stats_cor)]

    std_cor = {k:None for k in keys}
    std_nocor = {k:None for k in keys}
    std_cor_scale = {k:None for k in keys}
    stats_cor = {k:None for k in keys}
    stats_nocor = {k:None for k in keys}
    stats_cor_scale = {k:None for k in keys}

    for k in keys:
        std_cor[k] = [np.std(l) for l in zip(*all_stats_cor[k][r[0]:r[1]])]
        std_nocor[k] = [np.std(l) for l in zip(*all_stats_nocor[k][r[0]:r[1]])]
        std_cor_scale[k] = [np.std(l) for l in zip(*all_stats_cor_scale[k][r[0]:r[1]])]

        stats_cor[k] = [np.mean(l) for l in zip(*all_stats_cor[k][r[0]:r[1]])]
        stats_nocor[k] = [np.mean(l) for l in zip(*all_stats_nocor[k][r[0]:r[1]])]
        stats_cor_scale[k] = [np.mean(l) for l in zip(*all_stats_cor_scale[k][r[0]:r[1]])]

    return stats_cor, stats_nocor, stats_cor_scale, std_cor, std_nocor, std_cor_scale


def save_all_data(filename, all_data, eps_list):
    tmp_all_data = copy.deepcopy(all_data)
    tmp_all_data.extend([eps_list])
    with open(filename, 'wb') as handle:
        pickle.dump(tmp_all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restore_all_data(filename):
    with open(filename, 'rb') as handle:
        all_data = pickle.load(handle)
    eps_list = all_data[-1]
    all_data = all_data[:-1]

    return all_data, eps_list

def _plot(eps_list, data, k, xl, yl, filename):
    '''
    Plot four graphs. Internal routine for plot
    '''
    stats_cor, stats_nocor, stats_cor_scale, std_cor, std_nocor, std_cor_scale = data
    labels = ['cor', 'nocor', 'cor_scale']
    stats = [stats_cor, stats_nocor, stats_cor_scale]
    err_bars = [std_cor, std_nocor, std_cor_scale]
    for stat, err, label in zip(stats, err_bars, labels):
        plt.errorbar(eps_list, stat[k], yerr=err[k], label=k+','+label)

    title_content = filename
    try:
        p = re.compile('\(.*\)')
        title_content = p.search(filename)[0]
    except TypeError:
        pass

    plt.legend(loc='upper left')
    plt.title('epsilon VS '+k+title_content)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()


def estimate(dataX, dataA, dataY):
    eta_max = 0
    eta_min = 0
    pi_cor = np.mean([1.0 if a > 0 else 0.0 for a in dataA])
    alpha = eta_min * (eta_max - pi_cor) / pi_cor * (eta_max - eta_min)
    beta = (1 - eta_max) * (pi_cor - eta_min) / (1 - pi_cor) * (eta_max - eta_min)
