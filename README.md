# noise_fairlearn

## Introduction

This repo consists of code used for reproducing the results shown in the paper [**Noise-tolerant fair classification**](https://arxiv.org/abs/1901.10837) (Neurips 2019) by Alexandre Louis Lamy, Ziyuan Zhong, Aditya Krishna Menon, Nakul Verma.
It implements a general interface for noisy fair binary classification.
It currently supports the following:

### fairness criteria:

- Demographic Parity
- Equality of Opportunity

### classifiers:

- Agarwal

### datasets:

- UCI Adult
- UCI German
- UCI Bank
- COMPAS
- Law School

## Usage

Directly use demo.ipynb or run

```
python3 run_experiment.py --eval_objective test_tau --dataset compas --rho-plus 0.2 --rho-minus 0.2 --frac 1 --criteria DP --classifier Agarwal --trials 3 --plot-result
```

## Reference

Code is modified on top of code in the following repos:

files in fair_classification are modified from:

<https://github.com/mbilalzafar/fair-classification>

files in fairlearn are modified from:

<https://github.com/Microsoft/fairlearn>

files in fairERM are modified from:
<https://github.com/jmikko/fair_ERM>
