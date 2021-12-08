import numpy as np
from base_libs import Master
from base_libs.Master import ITEM, QUANTITIY, Unwrap
from base_libs.Plots import PlotHist, PlotHist2D, PlotHistComparison, Save
from base_libs.SelectionQuantities import CalculateQuantities
import MC_lib
import os


def PlotAllCorrelations(data, quantities, out_directory):
    residual = quantities[QUANTITIY.INVARIANT_MASS] - quantities[QUANTITIY.TRUE_INVARIANT_MASS]
    # add a custom quantity to the dictionary
    quantities.update({"RESIDUAL": residual})
    MC_lib.plotLabels.update(
        {"RESIDUAL": "invaraint mass - true invariant mass (GeV)"})

    # don't want to mutate existing data when plotting so pass soft copies instead
    print("reco invariant mass")
    MC_lib.PlotCorrelations(data.copy(), quantities.copy(), QUANTITIY.INVARIANT_MASS, out_directory, "correlations_inv_mass.png")
    print("true invariant mass")
    MC_lib.PlotCorrelations(data.copy(), quantities.copy(), QUANTITIY.TRUE_INVARIANT_MASS, out_directory, "correlations_true_inv_mass.png")
    print("residual")
    MC_lib.PlotCorrelations(data.copy(), quantities.copy(), "RESIDUAL", out_directory, "correlations_res.png", x_range=[-1, 5])


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
        MC_lib.PlotResiduals(data_1, data_2, True, out)
        PlotAllCorrelations(data_1, data_2, out)
    else:
        MC_lib.PlotQuantities(data_1, data_2, True, out)



print("loading data...")
data = Master.DataList(filename="ROOTFiles/Prod4a_6GeV_BeamSim_00.root")
outDir = "prod4a_MC_6GeV_00_allshower/"

print("computing quantities...")
quantities = CalculateQuantities(data, allPairs=True, param=None, conditional=None, cut=None)

print("finding signal...")
mask = MC_lib.FindPi0Signal(quantities, data)

print("filtering events...")
signal_q, background_q = MC_lib.Filter(quantities.copy(), mask, quantities[QUANTITIY.SHOWER_PAIRS]) # apply filter to calculated quantities

signal_d, background_d = MC_lib.Filter(data, mask, quantities[QUANTITIY.SHOWER_PAIRS]) # apply filter to data

print("plotting...")
print("all events")
MakePlots(outDir, "", dict(data), dict(quantities))

print("signal events")
MakePlots(outDir, "signal/", dict(signal_d), dict(signal_q))

print("background events")
MakePlots(outDir, "background/", dict(background_d), dict(background_q))

print("signal + shower")
MakePlots(outDir, "both/", dict(signal_q), dict(background_q), single=False)