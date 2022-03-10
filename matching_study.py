"""
Created on: 08/02/2022 17:00

Author: Shyam Bhuller

Description: Compare results of different filters for Pi0MC. 
"""

import argparse
import os
import awkward as ak
import matplotlib.pyplot as plt

# custom modules
import Plots
import Master


def CreateFilteredEvents(events : Master.Data, nDaughters=None):
    valid = Master.Pi0MCMask(events, nDaughters)

    filtered = events.Filter([valid], [valid])

    print(f"Number of shower events: {ak.num(filtered.recoParticles.direction, 0)}")
    showers, _, selection_mask, angles = filtered.GetMCMatchingFilters(returnAngles=True)

    Plots.PlotHist(ak.ravel(angles[:, 0]), 50, "Angle between matched shower and true photon 1 (rad)")
    Plots.PlotHist(ak.ravel(angles[:, 1]), 50, "Angle between matched shower and true photon 2 (rad)")
    
    reco_filters = [showers, selection_mask]
    true_filters = [selection_mask]

    return filtered.Filter(reco_filters, true_filters)


def Plot1D(data : ak.Array, xlabels : list, subDir : str, plot_range = []):
    """ 1D histograms of data for each sample

    Args:
        data (ak.Array): list of samples data to plot
        xlabels (list): x labels
        subDir (str): subdirectiory to save in
        plot_range (list, optional): range to plot. Defaults to [].
    """
    if save is True: os.makedirs(outDir + subDir, exist_ok=True)
    for j in range(len(names)):
        Plots.PlotHistComparison(data[:, j], plot_range, bins=bins, xlabel=xlabels[j], histtype="step", labels=s_l, density=True)
        if save is True: Plots.Save( names[j] , outDir + subDir)


@Master.timer
def main():
    events = Master.Data(file)
    events.ApplyBeamFilter() # apply beam filter if possible
    ts = []
    rs = []
    es = []

    for i in range(len(filters)):
        t, r, e = Master.CalculateQuantities(CreateFilteredEvents(events, filters[i]), names)
        ts.append(t)
        rs.append(r)
        es.append(e)

    ts = ak.Array(ts)
    rs = ak.Array(rs)
    es = ak.Array(es)
    Plot1D(ts, t_l, "truth/")
    Plot1D(rs, r_l, "reco/")
    Plot1D(es, e_l, "fractional_error/", fe_range)

    if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
    plt.rcParams["figure.figsize"] = (6.4*3,4.8*1)
    for j in range(len(names)):
        plt.figure()
        for i in range(len(filters)):
            plt.subplot(1, 3, i+1)
            if i == 0:
                _, edges = Plots.PlotHist2D(ts[i][j], es[i][j], bins, y_range=fe_range, xlabel=t_l[j], ylabel=e_l[j], title=s_l[i], newFigure=False)
            else:
                Plots.PlotHist2D(ts[i][j], es[i][j], edges, y_range=fe_range, xlabel=t_l[j], ylabel=e_l[j], title=s_l[i], newFigure=False)
        if save is True: Plots.Save( names[j] , outDir + "2D/")
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


if __name__ == "__main__":
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True secondary shower energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error (GeV)", "Opening angle fractional error (rad)", "Leading shower energy fractional error (GeV)", "Secondary shower energy fractional error (GeV)", "$\pi^{0}$ momentum fractional error (GeV)"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]
    fe_range = [-1, 1]

    filters = [2, 3, -3]
    s_l = ["2 daughters", "3 daughters", "> 3 daughters"]

    parser = argparse.ArgumentParser(description="Study em shower merging for pi0 decays.")
    parser.add_argument("-f", "--file", dest="file", type=str, default="ROOTFiles/pi0_0p5GeV_100K_5_7_21.root", help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms.")
    parser.add_argument("-s", "--save", dest="save", type=bool, default=False, help="whether to save the plots.")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/match_MC_compare/", help="directory to save plots.")
    #args = parser.parse_args("-f ROOTFiles/pi0_multi_9_3_22.root".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    main()
