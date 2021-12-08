#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:08:52 2021

@author: sb16165
"""
import sys
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# custom imports
import Master
from Master import Conditional, Unwrap, GetCondString, paramString, paramAxesLabels
from Plots import PlotHist2D, Save, LeastSqrFit
import Analyser


start_time = time.time()

# handle command line imports
args = sys.argv

# we don't care about the plot type, we only care about the selected data
root_filename, reference_filename, subDirectory, beamData, _, cutData, config_file, param, conditional, cut = Master.ParseCommandLine(args)

print("Opening root file: " + root_filename)
os.makedirs(subDirectory, exist_ok=True)
data = Master.Data(root_filename)

# if a config file is specified, read the contents
if config_file != None:
    print("we have a config file!")
    param, conditional, cut = Master.ParseConfig(config_file)
    
    # we only care about events that reconstruct the pion
    param.append(13)
    conditional.append(Conditional.GREATER)
    cut.append(0)

if cutData is False:
    param = [13]
    conditional = [Conditional.GREATER]
    cut = [0]
    cutData = True


# start hit constants
r = 1 # cm
l_min = -1 # cm
l_max = 4 # cm

#interesting_quantity = list(paramString.keys())[list(paramString.values()).index("invariant mass")]
q = list(Analyser.CalculateParameters(data, param, conditional, cut, cutData, beamData, [r, l_min, l_max]))

q[0] = Analyser.SortPairsByEnergy(q[0])
q = Unwrap(q)

q_interesting = q[param[-1]] # last cut is the parameter to plot against
pair_len = len(q_interesting)


print("Data selected:")
for i in range(len(param)):
    print(paramString[param[i]] + GetCondString(conditional[i]) + str(cut[i]))


print("Saving Plots:")
nbins = 100

label = paramAxesLabels[param[-1]]
for i in range(len(q)):
    if param[-1] == i:
        continue
    print(paramString[i])

    if(len(q[i]) == pair_len):
        PlotHist2D(q_interesting, q[i], nbins, xlabel=label, ylabel=paramAxesLabels[i])
        Save(paramString[i], subDirectory, reference_filename)
    elif(len(q[i]) == 2):
        plt.figure(2, (12, 5))
        plt.subplot(121)
        PlotHist2D(q_interesting, q[i][0], nbins, xlabel=label, ylabel=paramAxesLabels[i], title="leading shower")
        plt.subplot(122)
        PlotHist2D(q_interesting, q[i][1], nbins, xlabel=label, ylabel=paramAxesLabels[i], title="sub-leading shower")
        Save(paramString[i], subDirectory, reference_filename)

#PlotHist(q_interesting, nbins, label)
LeastSqrFit(q_interesting, 10, xlabel=label)
Save(paramString[param[-1]], subDirectory, reference_filename)

print("done!")
print("ran in %s seconds. " % (time.time() - start_time) )