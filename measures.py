'''
Fair measure(DP/EO) for probabilistic predictions.
'''

import numpy as np

def fair_measure(predictions, dataA, dataY, creteria):
    l_0 = []
    l_1 = []
    for i, p in enumerate(predictions):
        if dataA[i] == 0 and (dataY[i] == 1 or creteria=='DP'):
            l_0.append(p)
        if dataA[i] == 1 and (dataY[i] == 1 or creteria=='DP'):
            l_1.append(p)

    m_0 = np.mean(l_0)
    m_1 = np.mean(l_1)

    # m = np.mean(predictions)
    # return np.max([np.abs(m-m_0), np.abs(m-m_1)])

    return np.abs(m_0 - m_1)
