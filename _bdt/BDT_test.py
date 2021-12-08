#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:58:42 2021

@author: sb16165
"""

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import training
from xgboost.training import train
import xgboost as xgb
import BDT_regressor
import base_libs.Plots


def ApplyEventSelection(prediction, test):
    prediction = prediction[test["leading pandora tag"] == 11]
    test = test[test["leading pandora tag"] == 11]
    prediction = prediction[test["secondary pandora tag"] == 11]
    test = test[test["secondary pandora tag"] == 11]

    prediction = prediction[test["leading CNN score"] > 0.6]
    test = test[test["leading CNN score"] > 0.6]
    prediction = prediction[test["secondary CNN score"] > 0.6]
    test = test[test["secondary CNN score"] > 0.6]

    prediction = prediction[test["leading nHits"] > 50]
    test = test[test["leading nHits"] > 50]
    prediction = prediction[test["secondary nHits"] > 50]
    test = test[test["secondary nHits"] > 50]

    prediction = prediction[test["leading start hits"] > 1]
    test = test[test["leading start hits"] > 1]
    prediction = prediction[test["secondary start hits"] > 1]
    test = test[test["secondary start hits"] > 1]

    prediction = prediction[test["opening angle"] < 1]
    test = test[test["opening angle"] < 1]
    return prediction, test


#training = pd.read_csv("BDT_input/features_new/diphoton_truth_hits.txt")
#test = pd.read_csv("BDT_input/Prod4a_1GeV_BeamSim_00_allshower.csv")


#test = pd.read_csv("BDT_input/features_new/pi0_0p5GeV_hits.txt")
#training = pd.read_csv("BDT_input/diphoton_truth_hits.txt")
#test = pd.read_csv("BDT_input/pi0_0p5GeV_hits.txt")

#del training["Unnamed: 0"]
del test["Unnamed: 0"]

features = [
    "average CNN score",
    "collection plane hits of shower pairs",
    "pair seperation (cm)",
    "opening angle (rad)",
    "leading energy (GeV)",
    "secondary energy (GeV)",
    "pair energy difference (GeV)",
    "decay vertex tangential distance (cm)",
    "decay vertex x position (cm)",
    "decay vertex y position (cm)",
    "decay vertex z position (cm)"
]

features_no_vertex = [
    "average CNN score",
    "collection plane hits of shower pairs",
    "pair seperation (cm)",
    "opening angle (rad)",
    "leading energy (GeV)",
    "secondary energy (GeV)",
    "pair energy difference (GeV)",
    "decay vertex tangential distance (cm)"
]

# Booster parameter
param = [
    ('eta', 0.1),
    ('gamma', 0.5),
    ('max_depth', 3),
    ('subsample', 1),
    ('colsample_bytree', 0.8),
    ('objective', 'reg:gamma'),
    ('eval_metric', 'error'),
    ('eval_metric', 'logloss'),
    ('eval_metric', 'rmse')
]

#test = test[test["true pi0 event"] == True]
prediction, true, target_train, model = BDT_regressor.Run(test, training, predict="invariant mass (GeV)", features=features)

outDir = "BDT_output/test/"
BDT_regressor.MakePlots(prediction, test, training, model, plot=True, save=True, subDirectory=outDir)
BDT_regressor.PlotResiduals(prediction, test["invariant mass (GeV)"], test["true invariant mass (GeV)"], _range=[-1, 1], subDirectory=outDir, save=True)
BDT_regressor.PlotCorrelations(prediction, test, "predicted invariant mass (GeV)", outDir, outName="correlations_p")
BDT_regressor.PlotCorrelations(test["invariant mass (GeV)"], test, "calculated invariant mass (GeV)", outDir, outName="correlations_c")
BDT_regressor.PlotCorrelations(test["true invariant mass (GeV)"], test, "true invariant mass (GeV)", outDir, outName="correlations_t")
BDT_regressor.PlotCorrelations(prediction - test["true invariant mass (GeV)"], test, "predicted - calculated invariant mass (GeV)", outDir, outName="correlations_res_p")
BDT_regressor.PlotCorrelations(test["invariant mass (GeV)"] - test["true invariant mass (GeV)"], test, "predicted - true invariant mass (GeV)", outDir, outName="correlations_res_c")
