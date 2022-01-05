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


def Stats(data, quantities):
    """
    return general Stats of the data.
    """
    number_of_events = len(Unwrap(data[ITEM.EVENT_ID]))
    number_of_showers = len(Unwrap(data[ITEM.HITS]))
    number_of_shower_pairs = len(Unwrap(quantities[QUANTITY.SHOWER_SEPERATION]))
    return number_of_events, number_of_showers, number_of_shower_pairs 


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


def Plot():
    """
    Plots for data, signal and background shower pairs.
    """
    print("plotting...")
    print("all events")
    MakePlots(outDir, "", dict(data), dict(quantities))

    print("signal events")
    MakePlots(outDir, "signal/", dict(signal_d), dict(signal_q))

    print("background events")
    MakePlots(outDir, "background/", dict(background_d), dict(background_q))

    print("signal + shower")
    MakePlots(outDir, "both/", dict(signal_q), dict(background_q), single=False)


def CutEfficiency(_all, signal, background, selection=None, mask=None, allPairs=True):
    global new_data, new_quantities, dummy, dummy_1
    #nEvt, nShower, nPair = Stats(*_all)
    signal_nEvt, signal_nShower, signal_nPair = Stats(*signal)
    background_nEvt, background_nShower, background_nPair = Stats(*background)

    if mask == None and selection != None:
        new_data = Master.CutDict(_all[0], *selection, None)
        new_quantities = CalculateQuantities(new_data, allPairs, *selection)
    elif mask != None and selection == None:
        new_data = Master.CutDict(_all[0], custom_mask=mask)
        new_quantities = CalculateQuantities(new_data, allPairs=allPairs)
    else:
        print("no selection specified")
        return

    signal_id = MC_lib.FindPi0Signal(new_quantities, new_data)
    new_quantities_f = MC_lib.Filter(new_quantities.copy(), signal_id, new_quantities[QUANTITY.SHOWER_PAIRS])
    new_data_f = MC_lib.Filter(new_data.copy(), signal_id, new_quantities[QUANTITY.SHOWER_PAIRS])

    #new_nEvt, new_nShower, new_nPair = Stats(new_data, new_quantities)
    new_signal_nEvt, new_signal_nShower, new_signal_nPair = Stats(new_data_f[0], new_quantities_f[0])
    new_background_nEvt, new_background_nShower, new_background_nPair = Stats(new_data_f[1], new_quantities_f[1])

    #diff_nEvt = abs(new_nEvt - nEvt)
    #diff_nShower = abs(new_nShower - nShower)
    #diff_nPair = abs(new_nPair - nPair)
    
    #diff_signal_nEvt = abs(new_signal_nEvt - signal_nEvt)
    #diff_signal_nShower = abs(new_signal_nShower - signal_nShower)
    diff_signal_nPair = abs(new_signal_nPair - signal_nPair)
    
    #diff_background_nEvt = abs(new_background_nEvt - background_nEvt)
    #diff_background_nShower = abs(new_background_nShower - background_nShower)
    diff_background_nPair = abs(new_background_nPair - background_nPair)

    signalPercentageLost = (diff_signal_nPair / signal_nPair)
    backgroundPercentageLost = (diff_background_nPair / background_nPair)

    print("percentage of signal shower pairs lost: % .3f" % (diff_signal_nPair / signal_nPair) )
    print("percentage of background shower pairs lost: % .3f" % (diff_background_nPair / background_nPair) )
    return signalPercentageLost, backgroundPercentageLost


print("loading data...")
data = Master.DataList(filename="ROOTFiles/Prod4a_6GeV_BeamSim_00.root")
outDir = "Prod4a_6GeV_BeamSim_00_allshower/"

print("computing quantities...")
selection = [
    [QUANTITY.CNN_SCORE],
    [Conditional.GREATER],
    [0.4]
]

null_selection = [None, None, None]

quantities = CalculateQuantities(data, True, *null_selection)

print("finding signal...")
mask = MC_lib.FindPi0Signal(quantities, data)

print("filtering events...")
signal_q, background_q = MC_lib.Filter(quantities.copy(), mask, quantities[QUANTITY.SHOWER_PAIRS]) # apply filter to calculated quantities
signal_d, background_d = MC_lib.Filter(data.copy(), mask, quantities[QUANTITY.SHOWER_PAIRS]) # apply filter to data

def AnalyseCuts():
    """
    see how 1D cuts affect the signal/background lost 
    """
    x = np.linspace(0.1, 1, 10, endpoint=True)
    y = []
    for i in range(len(x)):
        selection[2][0] = x[i]
        y.append(CutEfficiency( (data, quantities), (signal_d, signal_q), (background_d, background_q), selection ))

    y = np.array(y)

    Plots.Plot(x, y, "cnn score cut", "percentage lost", label=["signal", "background"])
