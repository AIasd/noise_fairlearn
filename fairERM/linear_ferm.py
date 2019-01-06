import numpy as np
import copy

class Linear_FERM:
    # The linear FERM algorithm
    def __init__(self, dataX, dataA, dataY, model, creteria):
        self.dataX = dataX.values
        self.dataA = dataA.values
        self.dataY = dataY.values
        self.values_of_sensible_feature = list(set(dataA))
        self.val0 = np.min(self.values_of_sensible_feature)
        self.val1 = np.max(self.values_of_sensible_feature)
        self.model = model
        self.u = None
        self.creteria = creteria

    def predict(self, dataX, dataA):
        dataX = dataX.values
        dataA = dataA.values
        if self.u is None:
            print('Model not trained yet!')
            return 0
        newdataX = np.array([ex if dataA[idx] == self.val0 else ex + self.u for idx, ex in enumerate(dataX)])

        prediction = self.model.predict(newdataX)
        return prediction

    def fit(self):
        # Evaluation of the empirical averages among the groups
        average_A_1 = None
        if self.creteria == 'EO':
            tmp = [ex for idx, ex in enumerate(self.dataX)
                   if self.dataY[idx] == 1 and self.dataA[idx] == self.val1]
            average_A_1 = np.mean(tmp, 0)
            tmp = [ex for idx, ex in enumerate(self.dataX)
                   if self.dataY[idx] == 1 and self.dataA[idx] == self.val0]
        else:
            tmp = [ex for idx, ex in enumerate(self.dataX)
                   if self.dataA[idx] == self.val1]
            average_A_1 = np.mean(tmp, 0)
            tmp = [ex for idx, ex in enumerate(self.dataX)
                   if self.dataA[idx] == self.val0]
        average_not_A_1 = np.mean(tmp, 0)

        # Evaluation of the vector u (difference among the two averages)
        self.u = -(average_A_1 - average_not_A_1)

        # Application of the new representation
        newdataX = copy.deepcopy(self.dataX)

        for idx in range(newdataX.shape[0]):
            if self.dataA[idx] == self.val0:
                newdataX[idx] += self.u


        # Fitting the linear model by using the new data
        self.model.fit(newdataX, self.dataY)
