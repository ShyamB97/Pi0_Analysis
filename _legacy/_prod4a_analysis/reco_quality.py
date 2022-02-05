"""
Created on: 19/01/2022 08:54

Author: Shyam Bhuller

Description: Make plots of reco - true kinematics.
Plots MC truth, errors (reco - true) and errors as a function of true kinematics.
"""
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


def Error(x, true_x):
    """Calculate errors (reco - true).

    Args:
        x (Array): reconstructed quantity
        true_x (Array): true quantity

    Returns:
        tuple: error, x and true_x
    """
    x = Unwrap(x)
    true_x = Unwrap(true_x)
    err = x - true_x
    return err, x, true_x


def PlotSingle(data, ranges, labels, bins, subDirectory, names, save):
    """Plot 1D histograms.

    Args:
        data (list[Array]): data to plot (should be flattened)
        ranges (list of length 2): plot range
        labels (list[string]): list of x labels
        bins (int/Array): number of bins or specific bins
        subDirectory (string): directory to save plots
        names (list[string]): name of plots to save
        save (bool): whether to save plots or not
    """
    if save is True: os.makedirs(subDirectory, exist_ok=True)
    for i in range(len(data)):
        if len(ranges[i]) == 2:
            data[i] = data[i][data[i] > ranges[i][0]]
            data[i] = data[i][data[i] < ranges[i][1]]
        Plots.PlotHist(data[i], bins, labels[i])
        if save is True: Plots.Save(names[i], subDirectory)


outDir = "/home/sb16165/MEGAsync/Pi0 Reconstruction/Plots/MC study/pi0_0p5GeV_100K/reco_quality/"
os.makedirs(outDir, exist_ok=True)
b = 50 # number of bins
save = False

#* Loading data
data = DataList(filename="../ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
quantities = CalculateQuantities(data, allPairs=False)

tags = quantities[ITEM.PANDORA_TAG]

#* errors
res_inv_mass, inv_mass, true_inv_mass = Error(quantities[QUANTITY.INVARIANT_MASS], quantities[QUANTITY.TRUE_INVARIANT_MASS])
res_angle, angle, true_angle = Error(quantities[QUANTITY.OPENING_ANGLE], quantities[QUANTITY.TRUE_OPENING_ANGLE])

paired_energy = Unwrap(GetShowerPairValues(data[ITEM.ENERGY], quantities[QUANTITY.SHOWER_PAIRS], returnPaired=True))
paired_true_energy = Unwrap(GetShowerPairValues(data[ITEM.TRUE_ENERGY], quantities[QUANTITY.SHOWER_PAIRS], returnPaired=True))
ind = np.argsort(paired_energy, axis=1)
paired_energy = np.take_along_axis(paired_energy, ind, axis=1)
paired_true_energy = np.take_along_axis(paired_true_energy, ind, axis=1)

res_l_energy, l_energy, true_l_energy = Error(paired_energy[:, 1], paired_true_energy[:, 1])
res_s_energy, s_energy, true_s_energy = Error(paired_energy[:, 0], paired_true_energy[:, 0])

true_pi0_energy = paired_true_energy[:, 0] + paired_true_energy[:, 1]
true_sum_momentum = (true_pi0_energy**2 - true_inv_mass**2)**0.5

pi0_energy = paired_energy[:, 0] + paired_energy[:, 1]
sum_momentum = (pi0_energy**2 - inv_mass**2)**0.5
sum_momentum[sum_momentum!=sum_momentum] = -999 

res_pi0_mom, pi0_mom, true_pi0_mom = Error(sum_momentum, true_sum_momentum )

#* Plot truths vs errors
names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]

x = [true_inv_mass, true_angle, true_l_energy, true_s_energy, true_sum_momentum]
x_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True secondary shower energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
#x_r = [ [] ] * 5
#x_r = [ [], [min(true_angle), 2.5], [], [], [] ]
x_r = [ [-998, 0.2], [min(true_angle), 2], [min(true_l_energy), 0.5], [min(true_s_energy), 0.5], [min(true_pi0_mom), 1.1] ]

y = [res_inv_mass, res_angle, res_l_energy, res_s_energy, res_pi0_mom]
y_l = ["Invariant mass error (GeV)", "Opening angle error (rad)", "Leading shower energy error (GeV)", "Secondary shower energy error (GeV)", "$\pi^{0}$ momentum error (GeV)"]
#y_r = [ [] ] * 4
#y_r = [ [-900, max(res_inv_mass)], [], [-2, max(res_l_energy)], [] ]
y_r = [[-0.15, max(res_inv_mass)], [-1, max(res_angle)], [-0.5, max(res_l_energy)], [-0.6, max(res_s_energy)], [-990, max(res_pi0_mom)]]

if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
for j in range(len(y)):
    for i in range(len(x)):
        Plots.PlotHist2D(x[i], y[j], b, x_range=x_r[i], y_range=y_r[j], xlabel=x_l[i], ylabel=y_l[j])
        if save is True: Plots.Save( names[j]+"_"+names[i] , outDir + "2D/")

#* Plot errors
PlotSingle(y, y_r, y_l, b, outDir + "errors/", names, save)

#* plot truths
PlotSingle(x, x_r, x_l, b, outDir + "truths/", names, save)

#? make more usable in the terminal? i.e. args and main func