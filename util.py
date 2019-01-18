'''
This file contains multiple functions that convert data to appropriate format, invoke code in Agarwal classifier/Zafar classifier and plot according to results.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from cleanlab.classification import LearningWithNoisyLabels

import time
import pickle
import copy
import sys
import re
import os
from collections import namedtuple
from random import seed

from load_data import load_adult, load_compas, load_law, load_german
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


class SVC_u:
    def __init__(self):
        self.clf = SVC(gamma=2, C=1, probability=True, random_state=0)

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

SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
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


    # To correct wrong estimation
    rho_a_minus = np.min([pi_a_corr, rho_a_minus])

    pi_a = (pi_a_corr - rho_a_minus)/(1 - rho_a_plus - rho_a_minus)

    alpha_a = (1-pi_a)*rho_a_minus / pi_a_corr
    beta_a = pi_a*rho_a_plus / (1-pi_a_corr)

    if (1 - alpha_a - beta_a) < 0:
        print('the sum of alpha_a and beta_a is too large.', alpha_a, beta_a)
        alpha_a, beta_a = 0.49, 0.49

    return alpha_a, beta_a





def scale_eps(eps, alpha_a, beta_a):

    # new_eps = None
    # if creteria == 'DP':
    #     new_eps = eps * (1-alpha_a-beta_a)
    # elif creteria == 'EO':
    #     alpha_a_p = alpha_a*p_01 / ((1-alpha_a)*p_11 + alpha_a*p_01)
    #     beta_a_p = beta_a*p_11 / ((1-beta_a)*p_01 + beta_a*p_11)
    #     # print('alpha_a_p:', alpha_a_p, 'beta_a_p:',  beta_a_p)
    #     new_eps =  eps * (1-alpha_a_p-beta_a_p)

    return eps * (1 - alpha_a - beta_a)



def denoiseA(data_cor, rho):
    # classifiers = [
    #     GaussianNB(),
    #     LogisticRegression(random_state=0, solver = 'lbfgs', multi_class = 'auto'),
    #     KNeighborsClassifier(n_neighbors=3),
    #     SVC(kernel="linear", C=0.025, probability=True, random_state=0),
    #     SVC(gamma=2, C=1, probability=True, random_state=0),
    #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #     MLPClassifier(alpha=1, random_state=0, ),
    #     AdaBoostClassifier(random_state=0),
    #     QuadraticDiscriminantAnalysis()
    # ]

    rho_a_plus, rho_a_minus = rho

    dataX = data_cor[0]
    cor_dataA = data_cor[2]
    dataA = data_cor[5]

    lnl2 = LearningWithNoisyLabels(clf=SVC(gamma=2, C=1, probability=True, random_state=0))
    noise_matrix = np.array([[1-rho_a_plus, rho_a_plus],[rho_a_minus, 1-rho_a_minus]])
    # noise_matrix = np.array([[1-rho_a_plus, rho_a_minus],[rho_a_plus, 1-rho_a_minus]])
    lnl2.fit(X = dataX.values, s = cor_dataA.values, noise_matrix=noise_matrix)

    denoised_dataA = pd.Series(lnl2.predict(dataX.values))
    data_denoised = copy.deepcopy(data_cor)
    data_denoised[2] = denoised_dataA

    print('after denoised:', np.sum(denoised_dataA==0), np.sum(denoised_dataA==1))

    # Check recovery accuracy
    auc1 = np.mean(dataA.values==cor_dataA.values)
    auc2 = np.mean(dataA.values==denoised_dataA.values)

    print('auc:', auc1, auc2)
    print(lnl2.noise_matrix, rho_a_plus, rho_a_minus)


    # lnl = LearningWithNoisyLabels(clf=SVC(gamma=2, C=1, probability=True, random_state=0))
    # lnl.fit(X = dataX.values, s = cor_dataA.values)
    #
    # rho_a_plus_est = lnl.noise_matrix[0][1]
    # rho_a_minus_est = lnl.noise_matrix[1][0]
    # rho_est = [rho_a_plus_est, rho_a_minus_est]
    #
    # print(lnl.noise_matrix, rho_a_plus_est, rho_a_minus_est)


    return data_denoised


def experiment(dataset, rho, frac, eps_list, criteria, classifier, trials, include_sensible, filename, learner_name='lsq', verbose=False):
    '''
    dataset: one of ['compas', 'adult', 'adultr']
    rho: [a, b] where a, b in interval [0,0.5]
    frac: real number in interval [0,1]. The fraction of the data points in chosen dataset to use.
    eps_list: a list of non-negative real numbers
    criteria: one of ['DP','EO']
    classifier: one of ['Agarwal', 'Zafar', 'Shai']. Zafar is the fastest.
                Shai will ignore eps_list since eps=0 is inherent in this implementation.
    trials: the number of trials to run
    include_sensible: boolean. If to include sensitive attribute as a feature for optimizing the oroginal loss. Note that even
                      if this is set to False, sensitive attribute will still be used for constraint(s).
    filename: the file name to store the log of experiment(s).
    learner_name: ['lsq', 'SVM']
    verbose: boolean. If print out info at each run.
    '''
    sensible_name = None
    sensible_feature = None
    learner = None

    if dataset == 'adultr':
        datamat = load_adult(frac)
        sensible_name = 'race'
        sensible_feature = 8
    if dataset == 'adult':
        datamat = load_adult(frac)
        sensible_name = 'gender'
        sensible_feature = 9
    elif dataset == 'law':
        datamat = load_law(frac)
        sensible_name = 'racetxt'
        sensible_feature = 9

        # lsq does not work for law
        learner_name = 'SVM'
    elif dataset == 'german':
        datamat = load_german(frac)
        sensible_name = 'Foreign'
        sensible_feature = 21
    else:
        datamat = load_compas(frac)
        sensible_name = 'race'
        sensible_feature = 4

    if learner_name == 'SVM':
        learner = SVC_u()
    else:
        learner = LeastSquaresLearner()

    print('learner_name', learner_name)


    if criteria == 'EO':
        tests = [{"cons_class": moments.EO, "eps": eps} for eps in eps_list]
    else:
        tests = [{"cons_class": moments.DP, "eps": eps} for eps in eps_list]

    all_data = _experiment(datamat, tests, rho, trials, sensible_name, sensible_feature, criteria, classifier, include_sensible, learner, verbose)
    save_all_data(filename, all_data, eps_list)

    return all_data

def _experiment(datamat, tests, rho, trials, sensible_name, sensible_feature, creteria, classifier, include_sensible, learner, verbose):
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



        # p_01, p_11 = get_eta(data_nocor)
        # p_01_cor, p_11_cor = get_eta(data_cor)

        data_denoised = denoiseA(data_cor, rho)

        # p_01_est, p_11_est = est_p(p_01_cor, p_11_cor, rho_est)


        alpha_a, beta_a = estimate_alpha_beta(cor_dataA, dataY, rho, creteria)
        # alpha_a_est, beta_a_est = estimate_alpha_beta(cor_dataA, dataY, rho_est, creteria)

        # print('data_cor', np.mean(data_cor[1].values), np.mean(data_cor[2].values))
        #
        # print('data_nocor', np.mean(data_nocor[1].values), np.mean(data_nocor[2].values))
        #
        # print('data_denoised', np.mean(data_denoised[1].values), np.mean(data_denoised[2].values))

        for test_0 in tests:
            eps_0 = test_0['eps']
            test = copy.deepcopy(test_0)

            res_cor = run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

            res_nocor = run_test(test, data_nocor, sensible_name, learner,  creteria, verbose, classifier)

            res_denoised = run_test(test, data_denoised, sensible_name, learner,  creteria, verbose, classifier)

            test['eps'] = scale_eps(eps_0, alpha_a, beta_a)
            res_cor_scale = run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)

            # test['eps'] = scale_eps(eps_0, alpha_a_est, beta_a_est, p_01_est, p_11_est, creteria)
            # res_cor_scale_est = run_test(test, data_cor, sensible_name, learner, creteria, verbose, classifier)


            # results = [res_cor, res_nocor, res_denoised, res_cor_scale, res_cor_scale_est]
            results = [res_cor, res_nocor, res_denoised, res_cor_scale]

            for k in keys:
                for j in range(n):
                    all_data[j][k][i].append(results[j][k])

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
        disp = fair_measure(pred, dataA, dataY, creteria)
        error = 1 - accuracy_score(dataY, pred)
        return disp, error

    res = dict()
    res["disp_train"], res["error_train"] = _get_stats(algorithm, dataX_train, dataA_train, dataY_train)
    res["disp_test"], res["error_test"] = _get_stats(algorithm, dataX_test, dataA_test, dataY_test)

    return res


def run_test_Agarwal(test, data, sensible_name, learner, creteria):

    dataX, dataY, dataA, dataX_train, dataY_train, dataA_train, dataX_test, dataY_test, dataA_test = data

    res_tuple = red.expgrad(dataX, dataA, dataY, learner,
                            cons=test["cons_class"](), eps=test["eps"], debug=False)

    res = res_tuple._asdict()
    Q = res["best_classifier"]

    def _get_stats(clf, dataX, dataA, dataY):
        pred = clf(dataX)
        disp = fair_measure(pred, dataA, dataY, creteria)
        # print('dataX', dataX.values.shape, dataX.values[3, :])
        # print('pred', pred.values[:20])
        # print('dataY', dataY.values[:20])
        # print(np.mean(dataY.values), np.mean(pred))
        error = np.mean(np.abs(dataY.values-pred.values))
        # print('disp, error', disp, error)
        return disp, error

    res["disp_train"], res["error_train"] = _get_stats(Q, dataX_train, dataA_train, dataY_train)
    res["disp_test"], res["error_test"] = _get_stats(Q, dataX_test, dataA_test, dataY_test)

    return res




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


def plot(filename, ref_end=0.2, ref_line=[False, False, False, False], mode='four', save=False):
    y_label = 'DDP'

    p_eo = re.compile('EO')
    if p_eo.search(filename):
        y_label = 'DEO'


    all_data, eps_list = restore_all_data(filename)
    data = summarize_stats(all_data)

    keys = ["disp_train", "disp_test", "error_train", "error_test"]
    xlabels = ['$\\tau$' for _ in range(4)]
    ylabels = [y_label, y_label, 'Error%', 'Error%']
    leg_pos_list = ['upper left', 'upper left', 'lower left', 'lower left']

    for k, xl, yl, leg_pos, ref in zip(keys, xlabels, ylabels, leg_pos_list , ref_line):
        title_end = ''
        if 'train' in k:
            title_end = '(training)'
        _plot(eps_list, data, k, xl, yl, leg_pos, filename, ref, ref_end, mode, save, title_end)


def _plot(eps_list, data, k, xl, yl, leg_pos, filename, ref, ref_end, mode, save, title_end):
    '''
    Plot four graphs. Internal routine for plot
    '''
    curves_mean, curves_std = data
    labels = ['cor', 'nocor', 'denoise', 'cor_scale']

    fig, ax = plt.subplots()

    # labels = ['cor', 'nocor', 'cor_scale']
    for stat, err, label in zip(curves_mean, curves_std, labels):
        if mode == 'three':
            if label != 'denoise':
                ax.errorbar(eps_list, stat[k], yerr=err[k], label=label.replace('_', ' '))
        else:
            ax.errorbar(eps_list, stat[k], yerr=err[k], label=label.replace('_', ' '))
    if ref:
        # lims = [
        #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        #     np.max([ax.get_xlim(), ax.get_ylim()])   # max of both axes
        # ]
        # lims2 = copy.deepcopy(lims)
        lims = [0, ref_end]
        lims2 = [0, ref_end]

        # lims[1] /= 2
        ax.plot(lims, lims2, 'k-', alpha=0.75, zorder=0, color='grey', linestyle='dashed')

    title_content = None
    try:
        p0 = re.compile('all\_data\_(.*)\.pickle')
        save_file_content = p0.search(filename).group(1)

        p = re.compile('all\_data\_([a-zA-Z]+),([0-9\.]+),([0-9\.]+),[0-9\.]+,([A-Z]{2}),[a-zA-Z]+,[0-9]+,[a-zA-Z]+\.pickle')
        res = p.search(filename)
        dataset, rho_a_plus, rho_a_minus, creteria = res.group(1), res.group(2), res.group(3), res.group(4)

        if dataset == 'law':
            dataset = 'Law'
        elif dataset == 'adult':
            dataset = 'adult'
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

    save_dir = 'imgs/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if save:
        fig.savefig(save_dir+k+'_'+save_file_content+'.pdf', bbox_inches='tight')




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
#     res["disp_train"] = disp.gamma(Q, True).max()
#     res["disp_test"] = disp_test.gamma(Q, True).max()
#     res["error_train"] = error.gamma(Q, True)[0]
#     res["error_test"] = error_test.gamma(Q, True)[0]
#
#     return res

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

# def get_eta(data):
#     '''
#     Calculate Pr[Y=1|A=0] and Pr[Y=1|A=1] in training set.
#     '''
#     dataY, dataA = data[1], data[2]
#     c_a_0, c_a_1, c_01, c_11 = 0, 0, 0, 0
#
#     for y, a in zip(dataY, dataA):
#         if a == 0:
#             c_a_0 += 1
#             if y == 1:
#                 c_01 += 1
#         else:
#             c_a_1 += 1
#             if y == 1:
#                 c_11 += 1
#     p_01 = c_01 / c_a_0
#     p_11 = c_11 / c_a_1
#     # print('p_01:', p_01, 'p_11:', p_11)
#     return p_01, p_11
#
# def est_p(p_01_cor, p_11_cor, rho_est):
#     '''
#     estimate p_01 and p_11 using p_01_cor and rho_est
#     '''
#
#     rho_a_plus_est, rho_a_minus_est = rho_est
#     p_01_est = (p_01_cor - rho_a_minus_est)/(1 - rho_a_plus_est - rho_a_minus_est)
#     p_11_est = (p_11_cor - rho_a_minus_est)/(1 - rho_a_plus_est - rho_a_minus_est)
#
#     return p_01_est, p_11_est


  # rho_a_plus, rho_a_minus = rho
  #
  # pi_a_corr = np.sum([1.0 if a > 0 and y > 0 else 0.0 for a, y in zip(cor_dataA, dataY)])/ np.sum([1.0 if y > 0 else 0.0 for y in dataY])
  #
  # pi_a = (pi_a_corr - rho_a_minus)/(1 - rho_a_plus - rho_a_minus)
  #
  # alpha_a = (1-pi_a)*rho_a_minus / pi_a_corr
  # beta_a = pi_a*rho_a_plus / (1-pi_a_corr)
  # print('alpha_a', alpha_a, beta_a)
  #
  # p_01, p_11 = get_eta(data_nocor)
  #
  #
  # pi_a_corr = np.mean([1.0 if a > 0 else 0.0 for a in cor_dataA])
  # pi_a = (pi_a_corr - rho_a_minus)/(1 - rho_a_plus - rho_a_minus)
  # alpha_a = (1-pi_a)*rho_a_minus / pi_a_corr
  # beta_a = pi_a*rho_a_plus / (1-pi_a_corr)
  #
  # alpha_a_p = alpha_a*p_01 / ((1-alpha_a)*p_11 + alpha_a*p_01)
  # beta_a_p = beta_a*p_11 / ((1-beta_a)*p_01 + beta_a*p_11)
  # print('alpha_a_p', alpha_a_p, beta_a_p)
