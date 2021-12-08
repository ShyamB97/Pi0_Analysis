#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:08:52 2021

@author: sb16165
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

import Plots
import Master
import Analyser
import numpy as np


file = "ROOTFiles/pi0_0p5GeV_100K_5_7_21.root"
config_file = "selection_full.txt"
#file = "ROOTFiles/diphoton_10K.root"
subDirectory = "diphoton/BDT_preselection/"
os.makedirs(subDirectory, exist_ok=True)

data = Master.Data(file)


features = Analyser.CalculateFeatures(data, [2], [Master.Conditional.GREATER], [50], mc=False)
#features = Analyser.CalculateFeatures(data, *Master.ParseConfig(config_file), mc=False)
#features = features[features["pair seperation"] > 3]
#features = features[features["pair angle"] < 1]
#features = Analyser.CalculateFeatures(data, mc=True)

#del features["pair leading"]
#del features["pair second"]
#del features["pair energy difference"]

#features.to_csv("BDT_input/pi0_0p5GeV_no_selection.txt")


#inv_mass_signal = features[features["signal"] == 1]["invariant mass"]
#inv_mass_background = features[features["signal"] == 0]["invariant mass"]



plt.figure()
Plots.PlotHist(features["invariant mass"], xlabel="Shower pair invariant mass (GeV)")
print(np.mean(features["invariant mass"]))
plt.axvline(0.13497, ymax=5000, label="$m_{\pi_{0}}$", color="black")
#Plots.PlotHistComparison(inv_mass_signal, inv_mass_background, 100, Master.paramAxesLabels[13], label_1="signal", label_2="background", alpha=0.5)

#Plots.Save("inv_mass_signal_check", subDirectory, "pi0_0p5GeV_100K_6_8_21_selection")