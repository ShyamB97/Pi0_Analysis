from enum import Enum
from numpy.core.fromnumeric import var
import xgboost as xgb
import numpy as np

class HYPERPARAM(Enum):
    LEARNING_RATE = 0
    REGULARIZATION = 1
    MAX_TREE_DEPTH = 2
    SUBSAMPLE = 3
    MAX_FEATURES = 4
    ITERATIONS = 5


def DefaultParam(objective):
    """
    Creates default hyper-parameters to use as a starting point.
    returns the hyoerparameters and number of iterations.
    """
    param = {}

    # Booster parameters
    param['eta']              = 0.3 # learning rate
    param['gamma']            = 0.5
    param['max_depth']        = 5 # maximum depth of a tree
    param['subsample']        = 1 # fraction of events to train tree on
    param['colsample_bytree'] = 0.2 # fraction of features to train tree on

    # Learning task parameters
    param['objective']   = objective # objective function
    param['eval_metric'] = 'error'           # evaluation metric for cross validation
    param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]

    return param, 1000


def DMatrix(training, test, predict, features):
    # Assign variable to predict and features
    target_train = training[predict]
    train_DM = xgb.DMatrix(training[features], target_train)
    
    target_test = test[predict]
    test_DM = xgb.DMatrix(test[features], target_test)

    return target_train, target_test, train_DM, test_DM


def Run(data, training, objective, predict, features, n=1000, param=None):
    if param == None:
        param, _ = DefaultParam(objective)

    # Assign variable to predict and features
    target_train, target_test, train_DM, test_DM = DMatrix(training, data, predict, features)

    # create watchlist to track evaluation metrics per iteration
    watchlist = [(test_DM,'test'), (train_DM,'train')]
    results = {}

    model = xgb.train(param, train_DM, num_boost_round=n, evals=watchlist, evals_result=results, verbose_eval=False) # train model
    prediction = model.predict(test_DM) # use model on data

    return prediction, target_test, target_train, results, model


def RunNSample(data, training, objective, predict, features, vary : HYPERPARAM, sample, vRange=[], n=1000, param=None):
    """
    RUN the BDT training and prediction multiple times to see how a hyperparameter
    affects the prediction/evaulation metric.
    """
    if param == None:
        param, _ = DefaultParam(objective)

    # Assign variable to predict and features
    target_train, target_test, train_DM, test_DM = DMatrix(training, data, predict, features)

    if len(vRange) < 2:
        if vary == HYPERPARAM.ITERATIONS:
            vRange = [10, int(1E6)]
        if vary == HYPERPARAM.LEARNING_RATE:
            vRange = [0.01, 1]
        if vary == HYPERPARAM.MAX_FEATURES:
            vRange = [0.01, 1]
        if vary == HYPERPARAM.MAX_TREE_DEPTH:
            vRange = [1, 10]
        if vary == HYPERPARAM.REGULARIZATION:
            vRange = [0, 10]
        if vary == HYPERPARAM.SUBSAMPLE:
            vRange = [0.1, 1]
    x = np.linspace(vRange[0], vRange[1], sample, endpoint=True)

    output = {
        "results"       : [],
        "prediction"   : [],
        "target_test"  : target_test,
        "target_train" : target_train
    }

    for i in range(sample):
        # create watchlist to track evaluation metrics per iteration
        watchlist = [(test_DM,'test'), (train_DM,'train')]
        results = {}

        if vary == HYPERPARAM.LEARNING_RATE:
            param[0] = ('eta', x[i])
        if vary == HYPERPARAM.REGULARIZATION:
            param[1] = ('gamma', x[i])
        if vary == HYPERPARAM.MAX_TREE_DEPTH:
            param[2] = ('max_depth', x[i])
        if vary == HYPERPARAM.SUBSAMPLE:
            param[3] = ('subsample', x[i])
        if vary == HYPERPARAM.MAX_FEATURES:
            param[4] = ('colsample_bytree', x[i])
        if vary == HYPERPARAM.ITERATIONS:
            n = x[i]

        model = xgb.train(param, train_DM, num_boost_round=n, evals=watchlist, evals_result=results, verbose_eval=False) # train model
        prediction = model.predict(test_DM) # use model on data
        output['results'].append(results)
        output['prediction'].append(prediction)
    return x, output

    
