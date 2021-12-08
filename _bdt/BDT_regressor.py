#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:53:52 2021

@author: sb16165
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from base_libs import Plots
from typing import Tuple

from scipy.special import gamma, polygamma

### OBJECTIVE FUCNTIONS ###
def MSE(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the first derivitive'''
        y = dtrain.get_label()
        return predt - y

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the second derivitive'''
        y = dtrain.get_label()
        return np.ones(y.shape)

    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def custom(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    def gradient() -> np.ndarray:
        '''Compute the first derivitive'''
        #return 1 - (y * np.exp(-predt))
        #return (polygamma(0, predt)/gamma(predt)) - predt*np.log(y)
        return (x - y) / x**2

    def hessian() -> np.ndarray:
        '''Compute the second derivitive'''
        y = dtrain.get_label()
        #return y * np.exp(-predt)
        #return (polygamma(1, predt)/gamma(predt)) - (polygamma(0, predt)/gamma(predt))**2 - np.log(y)
        return -(x - 2*y) / x**3

    y = dtrain.get_label()
    x = predt
    #x[x < 0] = 0 + 1e-6
    grad = gradient()
    hess = hessian()
    return grad, hess


def GLM(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''GLM using gamma devience and log link
    '''
    predt[predt < 0] = 1e-6
    y = dtrain.get_label()
    x = np.log(predt)
    den = (x/y) + (y/x) - 1

    def gradient() -> np.ndarray:
        '''Compute the first derivitive'''
        num = 1/y - (y/x**2)
        return num / den

    def hessian() -> np.ndarray:
        '''Compute the second derivitive'''
        return 2 * y / ( den * x**3)

    grad = gradient()
    hess = hessian() - grad**2
    return grad, hess
### OBJECTIVE FUCNTIONS ###


def MakePlots(prediction, data, training, model, subDirectory="", plot=True, save=False, plot_true_mass = True):

    def InvMass(_range=None, bins=20, name="prediction"):
        pred = prediction
        mass = data["invariant mass (GeV)"]
        true_mass = data["true invariant mass (GeV)"]
        if _range != None:
            pred = prediction[prediction < _range]
            mass = mass[mass < _range]
            true_mass = true_mass[true_mass < _range]

        
        h_1, h_2, edges = Plots.PlotHistComparison(pred, mass, bins, "invariant mass (GeV)", label_1="predicted", label_2="calculated", alpha=0.5, sf=3)
        if plot_true_mass is True:
            plt.hist(true_mass, edges, label="true Invariant mass", alpha=0.5, density=False, color="C2")
        else:
            plt.axvline(0.13497, ymax=max(h_1), label="$m_{\pi_{0}}$", color="black")
        plt.legend()
        if save is True:
            os.makedirs(subDirectory, exist_ok=True)
            Plots.Save(name, subDirectory, "BDT_regressor")

    ### INVARIANT MASS ###
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    InvMass(bins=100)
    InvMass(0.5, name="prediction_zoom")
    
    ### PREDICTION VS CALCULATED ###
    Plots.PlotHist2D(prediction, data["invariant mass (GeV)"], 100, xlabel="predicted invariant mass (GeV)", ylabel="calculated invariant mass (GeV)")
    if save is True:
        os.makedirs(subDirectory, exist_ok=True)
        Plots.Save("2D", subDirectory, "BDT_regressor")

    residual = prediction - data["invariant mass (GeV)"]
    energy_residual = data["leading energy (GeV)"] + data["secondary energy (GeV)"] - data["total true energy (GeV)"]
    angle_residual = data["opening angle (rad)"] - data["true opening angle (rad)"]

    ### PREDICTION - CALCULATED ###
    Plots.PlotHist(residual, 100, "prediction - calculated (GeV)")
    if save is True:
        Plots.Save("difference", subDirectory, "BDT_regressor")

    ### TRAINING SET INVARIANT MASS ###
    Plots.PlotHist(training["invariant mass (GeV)"], 100, "training set true invariant mass (GeV)")
    if save is True:
        Plots.Save("training", subDirectory, "BDT_regressor")

    ### ENERGY RESIDUAL ###
    Plots.PlotHist(energy_residual, 100, xlabel="energy residual (GeV)")
    if save is True:
        Plots.Save("energy_res", subDirectory, "BDT_regressor")

    ### ANGLE RESIDUAL ###
    Plots.PlotHist(angle_residual, 100, xlabel="opening angle residual (rad)")
    if save is True:
        Plots.Save("angle_res", subDirectory, "BDT_regressor")

    ### RESIDUAL VS ENERGY RESIDUAL ###
    Plots.PlotHist2D(residual, energy_residual, 100, xlabel="prediction - calculated (GeV)", ylabel="energy residual (GeV)")
    if save is True:
        Plots.Save("energy_res_2D", subDirectory, "BDT_regressor")

    ### RESIDUAL VS ANGLE RESIDUAL  ###
    Plots.PlotHist2D(residual, angle_residual, 100, xlabel="prediction - calculated (GeV)", ylabel="opening angle residual (rad)")
    if save is True:
        Plots.Save("angle_res_2D", subDirectory, "BDT_regressor")

    ### INVARIANT MASS RESIDUALS ###
    h_1, h_2, _ = Plots.PlotHistComparison(prediction - data["true invariant mass (GeV)"], data["invariant mass (GeV)"] - data["true invariant mass (GeV)"], 100, "invariant mass residual (GeV)", label_1="predicted", label_2="calculated", alpha=0.5, sf=3)
    plt.legend()
    if save is True:
        os.makedirs(subDirectory, exist_ok=True)
        Plots.Save("residual", subDirectory, "BDT_regressor")

    ### FEATURE IMPORTANCE ###
    plt.rcParams["figure.figsize"] = (8, 5)
    xgb.plot_importance(model, grid=False)
    plt.tight_layout()
    if save is True:
        Plots.Save("importance", subDirectory, "BDT_regressor")
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def PlotResiduals(prediction, calculated, true, bins=100, _range=[], subDirectory="", plot=True, save=False):
    l_p = "prediction - true mass (GeV)"
    l_c = "calculated mass - true mass (GeV)"
    res_p = prediction - true
    res_c = calculated - true

    if len(_range) > 1:
        res_p = res_p[res_p > _range[0]]
        res_p = res_p[res_p < _range[1]]
        res_c = res_c[res_c < _range[1]]
        res_c = res_c[res_c > _range[0]]

    Plots.PlotHist(res_p, bins, l_p)
    if save is True: Plots.Save("res_p.png", subDirectory)
    Plots.PlotHist(res_c, bins, l_c)
    if save is True: Plots.Save("res_c.png", subDirectory)
    Plots.PlotHistComparison(res_p, res_c, bins, "residuals (GeV)", label_1=l_p, label_2=l_c, alpha=0.5)
    if save is True: Plots.Save("res_both.png", subDirectory)


def PlotCorrelations(data, df, x_label="x label", subDirectory="", outName="correlations"):
    fig = plt.figure(figsize=(32, 18))
    for i in range(len(df.columns)):
        name = df.columns[i]
        plt.subplot(6, 8, i+1)
        Plots.PlotHist2D(data, df[name], 25, xlabel=None, ylabel=name)
    fig.text(0.5, 0.005, x_label, ha='center')
    out = subDirectory + outName + ".png"
    plt.savefig(out, dpi=500)
    plt.close()


def Run(data, training, n=1000, predict="invariant mass", features=None, param=None, _print=True):

    if features == None:
        features = data.columns[1:12]
    
    
    if param == None:
        param = {}

        # Booster parameters
        param['eta']              = 0.3 # learning rate
        param['gamma']            = 0.5
        param['max_depth']        = 5 # maximum depth of a tree
        param['subsample']        = 1 # fraction of events to train tree on
        param['colsample_bytree'] = 0.2 # fraction of features to train tree on

        # Learning task parameters
        param['objective']   = 'reg:gamma' # objective function
        param['eval_metric'] = 'error'           # evaluation metric for cross validation
        param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]


    # Assign variable to predict and features
    target_train = training[predict]
    train_DM = xgb.DMatrix(training[features], target_train)
    
    true = data[predict]
    test_DM = xgb.DMatrix(data[features], true)

    model = xgb.train(param, train_DM, num_boost_round=n) # train model
    prediction = model.predict(test_DM) # use model on data
    
    # print stats of run
    if _print == True:
        print(model.eval(test_DM))
        print("training mean and st.dev")
        print(np.mean(target_train))
        print(np.std(target_train))
        print("target mean and st.dev")
        print(np.mean(true))
        print(np.std(true))
        print("predicted mean and st.dev")
        print(np.mean(prediction))
        print(np.std(prediction))

    return prediction, true, target_train, model


#xgb.plot_tree(booster, num_trees=booster.best_iteration)

