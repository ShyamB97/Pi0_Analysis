### Place this in a script that is inteded to be ran ###
import sys
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_base_libs') # need this to import custom scripts in different directories
######
import Master 
from Master import ITEM, QUANTITY, Conditional, Unwrap
from Plots import PlotHist, PlotHist2D, PlotHistComparison, Save, PlotBar
from SelectionQuantities import CalculateQuantities, GetShowerPairValues
from Analyser import SortPairsByEnergy
import MC_lib

import matplotlib.pyplot as plt
import numpy as np
import os

def PlotAllCorrelations(data, quantities, out_directory):
    residual = quantities[QUANTITY.INVARIANT_MASS] - quantities[QUANTITY.TRUE_INVARIANT_MASS]
    # add a custom quantity to the dictionary
    quantities.update({"RESIDUAL": residual})
    MC_lib.plotLabels.update(
        {"RESIDUAL": "invaraint mass - true invariant mass (GeV)"})

    # don't want to mutate existing data when plotting so pass soft copies instead
    print("reco invariant mass")
    MC_lib.PlotCorrelations(data.copy(), quantities.copy(), QUANTITY.INVARIANT_MASS, out_directory, "correlations_inv_mass.png")
    print("true invariant mass")
    MC_lib.PlotCorrelations(data.copy(), quantities.copy(), QUANTITY.TRUE_INVARIANT_MASS, out_directory, "correlations_true_inv_mass.png")
    print("residual")
    MC_lib.PlotCorrelations(data.copy(), quantities.copy(), "RESIDUAL", out_directory, "correlations_res.png", x_range=[-2, 2])


def CompareShowerPairs(data, save=False):
    quantities_all = CalculateQuantities(data, allPairs=True)
    quantities = CalculateQuantities(data, allPairs=False)

    mask_all = MC_lib.FindPi0Signal(quantities_all, data)
    mask = MC_lib.FindPi0Signal(quantities, data)

    signal_all, _ = MC_lib.Filter(quantities_all, mask_all)
    signal, _ = MC_lib.Filter(quantities, mask)

    MC_lib.PlotQuantities(signal, signal_all, save, "prod4a_MC_1GeV_00_pairComparison/", "pair by hits", "all shower pairs")



def MakePlots(directory, sub_directory, data_1, data_2, single=True):
    """
    Main Plotting function, will plot all data, residuals and correlations of invariant mass and it's residuals vs all the data.
    Capable of plotting signal, background or both either in one data set or side-by-side.
    """
    out = directory + sub_directory
    os.makedirs(out, exist_ok=True)
    if single is True:
        MC_lib.PlotQuantities(data_2, None, True, out)
        MC_lib.PlotShowerPairCorrelations(data_1, data_2, True, out)
        MC_lib.PlotResiduals(data_1, data_2, True, out)
        PlotAllCorrelations(data_1, data_2, out)
    else:
        MC_lib.PlotQuantities(data_1, data_2, True, out)


print("loading data...")
data = Master.DataList(filename="ROOTFiles/Prod4a_6GeV_BeamSim_00.root")
outDir = "Prod4a_6GeV_BeamSim_00_allshower/"

print("computing quantities...")
selection = [
    [QUANTITY.CNN_SCORE, QUANTITY.SHOWER_PAIR_PANDORA_TAGS, QUANTITY.SHOWER_SEPERATION],
    [Conditional.GREATER, Conditional.NOT_EQUAL, Conditional.LESS],
    [0.5, 13, 200]
]

null_selection = [None, None, None]

quantities = CalculateQuantities(data, True, *null_selection)

print("finding signal...")
mask = MC_lib.FindPi0Signal(quantities, data)

print("filtering events...")
signal_q, background_q = MC_lib.Filter(quantities.copy(), mask, quantities[QUANTITY.SHOWER_PAIRS]) # apply filter to calculated quantities

signal_d, background_d = MC_lib.Filter(data, mask, quantities[QUANTITY.SHOWER_PAIRS]) # apply filter to data

print("plotting...")
print("all events")
MakePlots(outDir, "", dict(data), dict(quantities))

print("signal events")
MakePlots(outDir, "signal/", dict(signal_d), dict(signal_q))

print("background events")
MakePlots(outDir, "background/", dict(background_d), dict(background_q))

print("signal + shower")
MakePlots(outDir, "both/", dict(signal_q), dict(background_q), single=False)
