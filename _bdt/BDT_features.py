#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:07:50 2021

@author: sb16165
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from base_libs import Plots
from base_libs import Master
import Analyser

def Plot(features):
    Plots.PlotHist(features["invariant mass (GeV)"], xlabel="Shower pair invariant mass (GeV)")
    print(np.mean(features["invariant mass (GeV)"]))
    plt.axvline(0.13497, ymax=5000, label="$m_{\pi_{0}}$", color="black")
    #Plots.PlotHist(features["pair leading energy"] + features["pair secondary energy"] - features["total true energy"], xlabel="total energy residual (GeV)")
    #sPlots.PlotHist(features["opening angle"] - features["true opening angle"], xlabel="true opening angle (rad)")


file = "ROOTFiles/pi0_0p5GeV_100K_5_7_21.root"
#file = "ROOTFiles/diphoton_100K.root"
#file = "ROOTFiles/diphoton_test_50K.root"
config_file = "selection_full.txt"

data = Master.Data(file)

#features_0 = Analyser.CalculateFeatures(data, [2], [Master.Conditional.GREATER], [50], mc=False)
#features_1 = Analyser.CalculateDataFrame(data, [2], [Master.Conditional.GREATER], [50])
features = Analyser.CalculateDataFrameNew(data, [10], [Master.Conditional.GREATER], [50], mc=False)

features.to_csv("BDT_input/model_new/pi0_0p5GeV_hits.csv")
#features.to_csv("BDT_input/model_new/diphoton_train_hits.csv")
#features.to_csv("BDT_input/model_new/diphoton_valid_hits.csv")


plot = False
if plot is True:
    Plot(features)
    plt.show()
