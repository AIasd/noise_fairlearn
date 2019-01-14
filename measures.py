import numpy as np

def fair_measure(predictions, dataA, dataY, creteria):
    eq_sensible_feature = dict()
    sensible_feature_values = list(set(dataA.values.tolist()))

    for val in sensible_feature_values:
        positive_sensitive = 0
        eq_tmp = 0
        for i in range(len(predictions)):
            if dataA[i] == val and (dataY[i] == 1 or creteria=='DP'):
                positive_sensitive += 1
                if predictions[i] == 1:
                    eq_tmp += 1
        eq_sensible_feature[val] = eq_tmp / positive_sensitive

    disp = np.abs(eq_sensible_feature[sensible_feature_values[0]] -
                               eq_sensible_feature[sensible_feature_values[1]])

    return disp
