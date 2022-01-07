### Place this in a script that is inteded to be ran ###
import sys
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_base_libs') # need this to import custom scripts in different directories
######

import os
import numpy as np

from Master import Unwrap, DataList, QUANTITY, ITEM, Conditional
from SelectionQuantities import CalculateQuantities
import Plots
from MC_lib import GetShowerPairValues


def Residual(x, true_x):
    """
    Calculate residuals in suitable format for plotting.
    """
    x = Unwrap(x)
    true_x = Unwrap(true_x)
    res = x - true_x
    return res, x, true_x


def PlotSingle(data, ranges, labels, bins, subDirectory, save):
    """
    Plot 1D histograms.
    """
    if save is True: os.makedirs(subDirectory, exist_ok=True)
    for i in range(len(data)):
        if len(ranges[i]) == 2:
            data[i] = data[i][data[i] > ranges[i][0]]
            data[i] = data[i][data[i] < ranges[i][1]]
        Plots.PlotHist(data[i], bins, labels[i])
        if save is True: Plots.Save(str(i), subDirectory)


outDir = "diphoton_100K/"
os.makedirs(outDir, exist_ok=True)
b = 50 # number of bins
save = True

#* Loading data
data = DataList(filename="ROOTFiles/diphoton_100K.root")
quantities = CalculateQuantities(data, param=[QUANTITY.TRUE_INVARIANT_MASS, QUANTITY.TRUE_OPENING_ANGLE], conditional=[Conditional.GREATER, Conditional.GREATER], cut=[0, 0])

#* Residuals
res_inv_mass, inv_mass, true_inv_mass = Residual(quantities[QUANTITY.INVARIANT_MASS], quantities[QUANTITY.TRUE_INVARIANT_MASS])
res_angle, angle, true_angle = Residual(quantities[QUANTITY.OPENING_ANGLE], quantities[QUANTITY.TRUE_OPENING_ANGLE])

paired_energy = Unwrap(GetShowerPairValues(data[ITEM.ENERGY], quantities[QUANTITY.SHOWER_PAIRS], returnPaired=True))
paired_true_energy = Unwrap(GetShowerPairValues(data[ITEM.TRUE_ENERGY], quantities[QUANTITY.SHOWER_PAIRS], returnPaired=True))
ind = np.argsort(paired_energy, axis=1)
paired_energy = np.take_along_axis(paired_energy, ind, axis=1)
paired_true_energy = np.take_along_axis(paired_true_energy, ind, axis=1)

res_l_energy, l_energy, true_l_energy = Residual(paired_energy[:, 1], paired_true_energy[:, 1])
res_s_energy, s_energy, true_s_energy = Residual(paired_energy[:, 0], paired_true_energy[:, 0])

true_sum_momentum = paired_true_energy[:, 0] + paired_true_energy[:, 1]

#* Plot truths vs residuals
x = [true_inv_mass, true_angle, true_l_energy, true_s_energy, true_sum_momentum]
x_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True secondary shower energy (GeV)", "True system momentum (GeV)"]
x_r = [ [], [min(true_angle), 2.5], [], [], [] ]
#x_r = [ [-998, 0.2], [min(true_angle), 2], [min(true_l_energy), 0.5], [min(true_s_energy), 0.5], [min(true_sum_momentum), 1.1] ]

y = [res_inv_mass, res_angle, res_l_energy, res_s_energy]
y_l = ["Invariant mass residual (GeV)", "Opening angle residual (rad)", "Leading shower energy residual (GeV)", "Secondary shower energy residual (GeV)"]
y_r = [ [-900, max(res_inv_mass)], [], [-2, max(res_l_energy)], [] ]
#y_r = [[-0.15, max(res_inv_mass)], [-1, max(res_angle)], [-0.5, max(res_l_energy)], [-0.6, max(res_s_energy)]]

if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
for j in range(len(y)):
    for i in range(len(x)):
        Plots.PlotHist2D(x[i], y[j], b, x_range=x_r[i], y_range=y_r[j], xlabel=x_l[i], ylabel=y_l[j])
        if save is True: Plots.Save(str(j)+str(i), outDir + "2D/")

#* Plot residuals
PlotSingle(y, y_r, y_l, b, outDir + "residuals/", save)

#* plot truths
PlotSingle(x, x_r, x_l, b, outDir + "truths/", save)