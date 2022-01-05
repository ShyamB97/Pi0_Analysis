### Place this in a script that is inteded to be ran ###
import sys

from xgboost.core import Objective
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_base_libs') # need this to import custom scripts in different directories
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_bdt') # need this to import custom scripts in different directories
######
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

import Plots
import BDT_libs

def BinaryEvaluation(true, prediction, _print=False):
    """
    Calculate Binary Evaluation metrics after testing model.
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true = np.array(true, int)
    for i in range(len(true)):
        if true[i] == 1 and prediction[i] == 1:
            true_positive += 1
        if true[i] == 0 and prediction[i] == 0:
            true_negative += 1
        if true[i] == 0 and prediction[i] == 1:
            false_positive += 1
        if true[i] == 1 and prediction[i] == 0:
            false_negative += 1
    true_positive_rate = true_positive / (true_positive + false_negative)
    true_negative_rate = true_negative / (true_negative + false_positive)
    false_positive_rate = false_positive / (false_positive + true_negative)
    false_negative_rate = false_negative / (false_negative + true_positive)
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_negative + false_positive)
    
    if _print is True:
        print("true positive: %i" % true_positive)
        print("true negative: %i" % true_negative)
        print("false positive: %i" % false_positive)
        print("false negative: %i" % false_negative)
        print("true positive rate: %.3f" % true_positive_rate)
        print("true negative rate: %.3f" % true_negative_rate)
        print("false positive rate: %.3f" % false_positive_rate)
        print("false negative rate: %.3f" % false_negative_rate)
        print("accuracy: %.3f" % accuracy)
    return true_positive, true_negative, false_positive, false_negative, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, accuracy


def RunSingle(test, training, features, n, param, threshold=0.5):
    #* Run BDT
    prediction, true, target_train, results, model = BDT_libs.Run(test, training, 'binary:logisitic', "true pi0 event", features, n, param)
    trained_logloss = results['train']['logloss']
    test_logloss = results['test']['logloss']

    #* apply a threshold cut to probability
    # set threshold to 0.5 by default
    pi0_prediction = []
    for p in prediction:
        if p > threshold:
            pi0_prediction.append(1)
        else:
            pi0_prediction.append(0)

    #* Plots
    # feature importance
    plt.rcParams["figure.figsize"] = (8, 5)
    xgb.plot_importance(model, grid=True)
    plt.tight_layout()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    # learning curves
    x = np.linspace(1, n, n)
    Plots.Plot(x, trained_logloss, "iteration", "logloss", label="training")
    Plots.Plot(x, test_logloss, "iteration", "logloss", label="test", newFigure=False)

    # signal probability
    Plots.PlotHist(prediction, 100, xlabel="prediction")
    Plots.PlotHist2D(prediction, np.array(true, int), (100,2) , xlabel="predicted signal probability", ylabel="true pi0 event")

    # *print stats of run
    BinaryEvaluation(true, pi0_prediction, True)


def RunMultiple(test, training, features, n, param, threshold=0.5):
    """
    train and test BDT multiple times to see how performance varies with a hyperparamater.
    """
    x, output = BDT_libs.RunNSample(test, training, param[5][1], 'true pi0 event', features, BDT_libs.HYPERPARAM.LEARNING_RATE, 10, [0.03, 0.3], n=1000, param=param)
    y = np.linspace(1, n, n)
    for i in range(len(x)):
        newFig = False
        if i == 0: newFig = True
        Plots.Plot(y, output['results'][i]['train']['logloss'], "iteration", "logloss", label= f'{x[i]:.3f}', newFigure=newFig)
    for i in range(len(x)):
        newFig = False
        if i == 0: newFig = True
        Plots.Plot(y, output['results'][i]['test']['logloss'], "iteration", "logloss", label=f'{x[i]:.3f}', newFigure=newFig)

    binary_results = []
    xlabel = "learning rate"
    labels = [
        "true positive",
        "true negative",
        "false positive",
        "false negative",
        "true positive rate",
        "true negative rate",
        "false positive rate",
        "false negative rate",
        "accuracy"
    ]

    for i in range(len(output['prediction'])):
        #* apply a threshold cut to probability
        # set threshold to 0.5 by default
        pi0_prediction = []
        for p in output['prediction'][i]:
            if p > threshold:
                pi0_prediction.append(1)
            else:
                pi0_prediction.append(0)
        binary_results.append(list(BinaryEvaluation(output['target_test'], pi0_prediction)))

    binary_results = np.array(binary_results)
    for i in range(len(labels)):
        Plots.Plot(x, binary_results[:, i], xlabel, labels[i])


#* load data and apply cuts
data = pd.read_csv("../BDT_input/Prod4a_6GeV_BeamSim_00_allshower.csv", index_col=False)
data = data[data["leading cnn score"] > 0.5]
data = data[data["secondary cnn score"] > 0.5]
data = data[data["average pandora tag"] != 13]
data = data[data["average pandora tag"] != -999]
data.reset_index(drop=True, inplace=True)

#* split data into training/testing set
# Training data is a subset of the signal only, background is the remaining data.
training_index = data[:int(len(data)/2)].index
training = data.iloc[training_index]
test = data.drop(index=training_index)

#* Create feature list 
features = list(data.columns) # parameters to train on
exclude = ['true', 'position', 'direction', 'event ID', 'run', 'invariant mass'] # regular expressions in parameters we don't want to include in features
for f in data.columns:
    for regex in exclude:
        if regex in f:
            features.remove(f)
            break

#* Booster parameters
param = {}
param['eta']              = 0.05 # learning rate
param['gamma']            = 0
param['max_depth']        = 4 # maximum depth of a tree
param['subsample']        = 1 # fraction of events to train tree on
param['colsample_bytree'] = 1 # fraction of features to train tree on

#* Learning task parameters
# for objective function choose:
# binary:logistic for probability output
# binary:hinge for a binary score (0 or 1) output
param['objective']   = 'binary:logistic'
param['eval_metric'] = 'error' # evaluation metric for cross validation
param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]
n = 1000 # number

threshold = 0.5
RunSingle(test, training, features, n, param)
RunMultiple(test, training, features, n, param, threshold)
