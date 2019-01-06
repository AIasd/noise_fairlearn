def fair_measure(model, dataX, dataA, dataY, creteria):
    predictions = model.predict(dataX, dataA)
    truth = dataY
    eq_sensible_feature = dict()

    for val in list(set(dataA.values.tolist())):

        positive_sensitive = 0
        eq_tmp = 0
        for i in range(len(predictions)):
            if dataA[i] == val and (truth[i] == 1 or creteria=='DP'):
                positive_sensitive += 1
                if predictions[i] == 1:
                    eq_tmp += 1
        eq_sensible_feature[val] = eq_tmp / positive_sensitive

    return eq_sensible_feature
