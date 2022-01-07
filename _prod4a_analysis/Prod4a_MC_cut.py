### Place this in a script that is inteded to be ran ###
import sys
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_base_libs') # need this to import custom scripts in different directories
######
import Master 
from Master import ITEM, QUANTITY, Conditional, SelectionMask, Unwrap
from SelectionQuantities import CalculateQuantities, GetShowerPairValues
from Analyser import SortPairsByEnergy
import MC_lib
import Plots

import matplotlib.pyplot as plt
import numpy as np
import os


print("loading data...")
data = Master.DataList(filename="../ROOTFiles/Prod4a_6GeV_BeamSim_00.root")
outDir = "test/"

print("computing quantities...")
selection = [
    [QUANTITY.CNN_SCORE],
    [Conditional.GREATER],
    [0.4]
]

null_selection = [None, None, None]

quantities = CalculateQuantities(data, True, *null_selection)
custom_mask = MC_lib.AdvancedCNNScoreMask(quantities[QUANTITY.CNN_SCORE], quantities[QUANTITY.SHOWER_PAIRS], data[ITEM.ENERGY])

print("finding signal...")
mask = MC_lib.FindPi0Signal(quantities, data)

print("filtering events...")
signal_q, background_q = MC_lib.Filter(quantities.copy(), mask, quantities[QUANTITY.SHOWER_PAIRS]) # apply filter to calculated quantities
signal_d, background_d = MC_lib.Filter(data.copy(), mask, quantities[QUANTITY.SHOWER_PAIRS]) # apply filter to data

performance = MC_lib.CutEfficiency((data.copy(), quantities.copy()), (signal_d, signal_q), (background_d, background_q), selection=None, mask=custom_mask)

data = Master.CutDict(data, custom_mask=custom_mask)
quantities = CalculateQuantities(data, True, *null_selection)
mask = MC_lib.FindPi0Signal(quantities, data)
signal_q, background_q = MC_lib.Filter(quantities.copy(), mask, quantities[QUANTITY.SHOWER_PAIRS]) # apply filter to calculated quantities
label = MC_lib.plotLabels[QUANTITY.CNN_SCORE]


cnn = Unwrap(quantities[QUANTITY.CNN_SCORE])
Plots.PlotHist( cnn[cnn > -999] , bins=50, xlabel=label)

cnn_signal = Unwrap(signal_q[QUANTITY.CNN_SCORE])
cnn_background = Unwrap(background_q[QUANTITY.CNN_SCORE])
Plots.PlotHistComparison(cnn_signal[cnn_signal > -999], cnn_background[cnn_background > -999], bins=50, xlabel=label, label_1="signal", label_2="background", alpha=0.5)


single_quantities = [quantities[QUANTITY.CNN_SCORE], data[ITEM.ENERGY]]
single_quantities = [GetShowerPairValues(q, quantities[QUANTITY.SHOWER_PAIRS]) for q in single_quantities]
single_quantities = [Unwrap(q) for q in single_quantities]
single_quantities = SortPairsByEnergy(single_quantities, index=1)

Plots.PlotHist2D(single_quantities[0][0], single_quantities[0][1], 50, [-998, max(single_quantities[0][0])], [-998, max(single_quantities[0][1])], "leading " + label, "secondary " + label)
