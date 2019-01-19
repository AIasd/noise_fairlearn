'''
This file contains code for preprocessing/loading adult/compas dataset


Code for loading adult dataset is modified from:
https://github.com/jmikko/fair_ERM/blob/master/load_data.py

TBD: Better preprocessing(apply more one-hot-encoding) can be used. See
https://github.com/mbilalzafar/fair-classification/blob/master/disparate_impact/adult_data_demo/prepare_adult_data.py


Code for loading COMPAS dataset is modified from:
https://github.com/mbilalzafar/fair-classification/blob/master/disparate_mistreatment/propublica_compas_data_demo/load_compas_data.py

'''
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler

import urllib.request as url

from random import seed, shuffle
import os


SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


# def check_data_file(fname, addr):
#     '''
#     The code will look for the data file in the present directory, if it is not found, it will download them from GitHub.
#     '''
#     print("Looking for file '%s' in the current directory..." % fname)
#
#     if not os.path.isfile(fname):
#         print("'%s' not found! Downloading from GitHub..." % fname)
#         response = url.urlopen(addr)
#         data = response.read()
#         with open(fname, "wb") as fileOut:
#             fileOut.write(data)
#         print("'%s' download and saved locally.." % fname)
#     else:
#         print("File found in current directory..")

def load_german(frac=1, scaler=True):
    '''
    22 features and 1 target
    Job                            1000 non-null int64
    Credit amount                  1000 non-null int64
    Duration                       1000 non-null int64
    Purpose_car                    1000 non-null uint8
    Purpose_domestic appliances    1000 non-null uint8
    Purpose_education              1000 non-null uint8
    Purpose_furniture/equipment    1000 non-null uint8
    Purpose_radio/TV               1000 non-null uint8
    Purpose_repairs                1000 non-null uint8
    Purpose_vacation/others        1000 non-null uint8
    Sex_male                       1000 non-null uint8
    Housing_own                    1000 non-null uint8
    Housing_rent                   1000 non-null uint8
    Savings_moderate               1000 non-null uint8
    Savings_no_inf                 1000 non-null uint8
    Savings_quite rich             1000 non-null uint8
    Savings_rich                   1000 non-null uint8
    Check_moderate                 1000 non-null uint8
    Check_no_inf                   1000 non-null uint8
    Check_rich                     1000 non-null uint8
    Age_cat_Old                    1000 non-null uint8
    Foreign                        1000 non-null uint8
    Risk_bad                       1000 non-null uint8
    '''


    df_credit = pd.read_csv("datasets/new_german_credit_data.csv", index_col=0)


    #Let's look the Credit Amount column
    interval = (18, 25, 120)

    cats = ['Young', 'Old']
    df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)


    df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
    df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

    #Purpose to Dummies Variable
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
    #Sex feature in dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
    # Housing get dummies
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    # Housing get Saving Accounts
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    # Housing get Risk
    df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
    # Housing get Checking Account
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    # Housing get Age categorical
    df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)


    #Excluding the missing columns
    del df_credit["Age"]
    del df_credit["Saving accounts"]
    del df_credit["Checking account"]
    del df_credit["Purpose"]
    del df_credit["Sex"]
    del df_credit["Housing"]
    del df_credit["Age_cat"]
    del df_credit["Risk"]
    del df_credit['Risk_good']


    df_credit['Credit amount'] = np.log(df_credit['Credit amount'])



    datamat = df_credit.drop('Risk_bad', 1).values
    target = df_credit["Risk_bad"].values
    A = np.copy(datamat[:, 11])

    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)

    datamat[:, 11] = A


    datamat = np.concatenate([datamat, target[:, np.newaxis]], axis=1)
    datamat = np.random.permutation(datamat)
    datamat = datamat[:int(np.floor(len(datamat)*frac)), :]

    return datamat




def load_law(frac=1, scaler=True):
    '''
    ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime',
      'fam_inc', 'male', 'pass_bar', 'tier', 'racetxt']
    'racetxt' can have values {'Hispanic', 'American Indian / Alaskan Native', 'Black', 'White', 'Other', 'Asian'}
    '''
    LAW_FILE = 'datasets/law_data_clean.csv'
    # names = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime', 'fam_inc', 'male', 'pass_bar', 'tier', 'racetxt']
    data = pd.read_csv(LAW_FILE)

    data = data.loc[data['racetxt'].isin(['White', 'Black'])]
    # Here we apply discretisation on column marital_status
    # data.replace(['Hispanic', 'American Indian / Alaskan Native',
    #               'Black', 'Other', 'Asian'],
    #              ['Minority', 'Minority', 'Minority', 'Minority',
    #               'Minority'], inplace=True)


    # categorical fields
    category_col = ['racetxt']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values

    # datamat[:, 9], datamat[:, 11] = datamat[:, 11], datamat[:, 9]

    temp = np.copy(datamat[:, 11])
    datamat[:, 11] = datamat[:, 9]
    datamat[:, 9] = temp




    # datamat = datamat[datamat[:,11].argsort()]
    # datamat = datamat[:int(len(datamat)/7)]


    datamat = np.random.permutation(datamat)
    A = np.copy(datamat[:, 9])


    target = datamat[:, -1]
    datamat = datamat[:, :-1]


    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)

    datamat[:, 9] = A

    datamat = np.concatenate([datamat, target[:, np.newaxis]], axis=1)

    print('The dataset is loaded...')

    datamat = datamat[:int(np.floor(len(datamat)*frac)), :]


    return datamat


# def load_adult(frac=1, scaler=True):
#     '''
#     ['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain', 'capital.loss']
#     '''
#
#     ADULT_TRAIN_FILE = 'datasets/adult.data'
#     ADULT_TEST_FILE = 'datasets/adult.test'
#
#     names = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship',
#      'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', 'income']
#
#     data = pd.read_csv(
#             ADULT_TRAIN_FILE,
#             names = names, skipinitialspace=True
#                 )
#
#     data_test = pd.read_csv(
#             ADULT_TEST_FILE,
#             names=names, skipinitialspace=True
#         )
#     # print(data_test.head())
#     data = pd.concat([data, data_test])
#     data = data[data["occupation"] != '?']
#     data = data[data["native.country"] != '?']
#     data = data[data["workclass"] != '?']
#     data = data[data["race"] != 'Asian-Pac-Islander']
#     data = data[data["race"] != 'Amer-Indian-Eskimo']
#     data = data[data["race"] != 'Other']
#
#
#     # create numerical columns representing the categorical data
#     data['workclass_num'] = data.workclass.map({'Private':0.0, 'State-gov':1.0, 'Federal-gov':2.0, 'Self-emp-not-inc':3.0, 'Self-emp-inc':4.0, 'Local-gov':5.0, 'Without-pay':6.0})
#     data['over50K'] = np.where(data.income == '<=50K', 0.0, 1.0)
#     data['marital_num'] = data['marital.status'].map({'Widowed':0.0, 'Divorced':1.0, 'Separated':2.0, 'Never-married':3.0, 'Married-civ-spouse':4.0, 'Married-AF-spouse':4.0, 'Married-spouse-absent':5.0})
#
#     data['race_num'] = data.race.map({'Black':0.0, 'White':1.0, 'Asian-Pac-Islander':2.0, 'Amer-Indian-Eskimo':3.0, 'Other':4.0})
#     data['sex_num'] = np.where(data.sex == 'Female', 0.0, 1.0)
#     data['rel_num'] = data.relationship.map({'Not-in-family':0.0, 'Unmarried':0.0, 'Own-child':0.0, 'Other-relative':0.0, 'Husband':1.0, 'Wife':1.0})
#
#     data = data[data['race_num'] < 2]
#
#     print(data.values.shape)
#
#
#     X = data[['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain', 'capital.loss']]
#     y = data.over50K
#
#     datamat = np.concatenate([X, y[:, np.newaxis]], axis=1)
#     datamat = np.random.permutation(datamat)
#     datamat = datamat[:int(np.floor(len(datamat)*frac)), :]
#
#     return datamat



def load_adult(frac=1, scaler=True):
    '''
    :param frac: the fraction of data to return
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    dir = 'datasets'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    # train_addr = 'https://github.com/jmikko/fair_ERM/blob/master/datasets/adult/adult.data'
    # test_addr = 'https://github.com/jmikko/fair_ERM/blob/master/datasets/adult/adult.test'

    ADULT_TRAIN_FILE = dir+os.sep+'adult.data'
    ADULT_TEST_FILE = dir+os.sep+'adult.test'

    # check_data_file(ADULT_TRAIN_FILE, train_addr)
    # check_data_file(ADULT_TEST_FILE, test_addr)

    names=[
        "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
        "hours per week", "native-country", "income"]

    data = pd.read_csv(
        ADULT_TRAIN_FILE,
        names=names, skipinitialspace=True
            )

    data_test = pd.read_csv(
        ADULT_TEST_FILE,
        names=names, skipinitialspace=True
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data


    data = data[data["workclass"] != '?']
    data = data[data["occupation"] != '?']
    data = data[data["native-country"] != '?']
    data = data[data["race"] != 'Asian-Pac-Islander']
    data = data[data["race"] != 'Amer-Indian-Eskimo']
    data = data[data["race"] != 'Other']
    print(len(data))

    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)

    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation', 'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values

    datamat = np.random.permutation(datamat)

    target = datamat[:, -1]
    datamat = datamat[:, :-1]

    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)

    datamat = np.concatenate([datamat, target[:, np.newaxis]], axis=1)

    print('The dataset is loaded...')
    datamat = datamat[:int(np.floor(len(datamat)*frac)), :]

    print(datamat[0])

    return datamat



def load_compas(frac=1):
    '''
    Feature list:
    'intercept', 'age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'race', 'sex', 'priors_count', 'c_charge_degree'
    '''
    def add_intercept(x):

        """ Add intercept to the data before linear classification """
        m,n = x.shape
        intercept = np.ones(m).reshape(m, 1) # the constant b
        return np.concatenate((intercept, x), axis = 1)

    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] #features to be used for classification
    CONT_VARIABLES = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid" # the decision variable


    dir = 'datasets'
    if not os.path.isdir(dir):
        os.mkdir(dir)


    COMPAS_INPUT_FILE = dir+os.sep+"compas-scores-two-years.csv"
    # addr = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

    # check_data_file(COMPAS_INPUT_FILE, addr)

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])



    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)


    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]



    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    # y[y==0] = -1



    print("\nNumber of people recidivating within two years")
    print(pd.Series(y).value_counts())
    print("\n")


    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it


    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

        else: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)




        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES: # continuous feature, just append the name
            feature_names.append(attr)
        else: # categorical features
            if vals.shape[1] == 1: # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))



    """permute the data randomly"""
    perm = list(range(X.shape[0]))
    shuffle(perm)
    X = np.array([X[i] for i in perm])
    y = np.array([y[i] for i in perm])


    X = add_intercept(X)

    feature_names = ["intercept"] + feature_names
    assert(len(feature_names) == X.shape[1])
    print("Features we will be using for classification are:", feature_names, "\n")

    datamat = np.concatenate([X, y[:, np.newaxis]], axis=1)

    # Take fraction
    cutoff = int(np.floor(len(datamat)*frac))
    datamat = datamat[:cutoff, :]
    import copy
    ref = datamat[:, 4]
    tmp = copy.deepcopy(datamat[:, 4])

    ref[tmp==0.0] = 1.0
    ref[tmp==1.0] = 0.0

    return datamat
