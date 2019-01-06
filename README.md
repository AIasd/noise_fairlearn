# noise_fairlearn
## Introduction
This repo implements a general interface for noisy fair binary classification. It currently supports the following:

### fairness criteria:

* Demographic Parity
* Equality of Opportunity

### classifiers:

* Zafar
* Agarwal
* Shai (will ignore the epsilon value passed because epsilon=0 is inherent in the implementation)

### datasets:

 * UCI Adult
 * COMPAS


## Reference
Code is modified on top of code in the following repos:

files in fair_classification are modified from:

<https://github.com/mbilalzafar/fair-classification>

files in fairlearn are modified from:

<https://github.com/Microsoft/fairlearn>

<https://github.com/mbilalzafar/fair-classification>
