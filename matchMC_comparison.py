"""
Created on: 08/02/2022 17:00

Author: Shyam Bhuller

Description: Compare results of different filters for Pi0MC. 
"""

import os
import awkward as ak
import time
import numpy as np
import matplotlib.pyplot as plt

# custom modules
import Plots
import Master
import vector


def CreateFilteredEvents(events : Master.Event, nDaughters=None):
    valid, photons = Master.Pi0MCFilter(events, nDaughters)

    shower_dir = events.recoParticles.direction[valid]
    print(f"Number of showers events: {ak.num(shower_dir, 0)}")
    photon_dir = vector.normalize(events.trueParticles.momentum)[photons][valid]

    showers, _, selection_mask, angles = events.MatchMC(photon_dir, shower_dir, returnAngles=True)

    Plots.PlotHist(ak.ravel(angles[:, 0]), 50, "Angle between matched shower and true photon 1 (rad)")
    Plots.PlotHist(ak.ravel(angles[:, 1]), 50, "Angle between matched shower and true photon 2 (rad)")
    

    reco_filters = [valid, showers, selection_mask]
    true_filters = [valid, selection_mask]

    return events.Filter(reco_filters, true_filters), photons[valid][selection_mask]

@Master.timer
def CalculateQuantities(events : Master.Event, nDaughter : int = None):
    global names
    filtered, photons = CreateFilteredEvents(events, nDaughter)
    print(f"Number of events after filtering: {ak.num(filtered.recoParticles.nHits, 0)}")
    mct = Master.MCTruth(filtered, filtered.SortByTrueEnergy(), photons)
    rmc = Master.RecoQuantities(filtered, filtered.SortByTrueEnergy())

    # keep track of events with no shower pairs
    null = ak.flatten(rmc[-1], -1)
    null = ak.num(null, 1) > 0

    error = []
    reco = []
    true = []
    for i in range(len(names)):
        print(names[i])
        e, r, t = Master.Error(rmc[i], mct[i], null)
        error.append(e)
        reco.append(r)
        true.append(t)

    error = np.nan_to_num(error, nan=-999)
    reco = np.nan_to_num(reco, nan=-999)
    true = np.nan_to_num(true, nan=-999)
    return true, reco, error


#* user parameters
save = True
outDir = "pi0_0p5GeV_100K/match_MC_compare/"
bins = 50

names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True secondary shower energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
e_l = ["Invariant mass fractional error (GeV)", "Opening angle fractional error (rad)", "Leading shower energy fractional error (GeV)", "Secondary shower energy fractional error (GeV)", "$\pi^{0}$ momentum fractional error (GeV)"]
r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]

s = time.time()

events = Master.Event("ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
filters = [2, 3, -3]
f_l = ["2 daughters", "3 daughters", "> 3 daughters"]

null = events.recoParticles.direction.x == -999 # exclude events where direction couldn't be calculated

fe_range = [-1, 1]

ts = []
rs = []
es = []

for i in range(len(filters)):
    t, r, e = CalculateQuantities(events, filters[i])
    ts.append(t)
    rs.append(r)
    es.append(e)

if save is True: os.makedirs(outDir + "truths/", exist_ok=True)
for j in range(len(names)):
    plt.figure()
    for i in range(len(filters)):
        data = ts[i][j]
        data = data[data > -900]
        if i == 0:
            _, edges = Plots.PlotHist(data, bins=bins, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
        else:
            Plots.PlotHist(data, bins=edges, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)

    if save is True: Plots.Save( names[j] , outDir + "truths/")

if save is True: os.makedirs(outDir + "reco/", exist_ok=True)
for j in range(len(names)):
    plt.figure()
    for i in range(len(filters)):
        data = rs[i][j]
        data = data[data > -900]
        if i == 0:
            _, edges = Plots.PlotHist(data, bins=bins, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
        else:
            Plots.PlotHist(data, bins=edges, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)

    if save is True: Plots.Save( names[j] , outDir + "reco/")

if save is True: os.makedirs(outDir + "fractional_error/", exist_ok=True)
for j in range(len(names)):
    plt.figure()
    for i in range(len(filters)):
        data = es[i][j]
        data = data[data > fe_range[0]]
        data = data[data < fe_range[1]]
        if i == 0:
            _, edges = Plots.PlotHist(data, bins=bins, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
        else:
            Plots.PlotHist(data, bins=edges, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)

    if save is True: Plots.Save( names[j] , outDir + "fractional_error/")

if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
plt.rcParams["figure.figsize"] = (6.4*3,4.8*1)
for j in range(len(names)):
    plt.figure()
    for i in range(len(filters)):
        plt.subplot(1, 3, i+1)
        if i == 0:
            _, edges = Plots.PlotHist2D(ts[i][j], es[i][j], bins, y_range=fe_range, xlabel=t_l[j], ylabel=e_l[j], title=f_l[i], newFigure=False)
        else:
            Plots.PlotHist2D(ts[i][j], es[i][j], edges, y_range=fe_range, xlabel=t_l[j], ylabel=e_l[j], title=f_l[i], newFigure=False)
    if save is True: Plots.Save( names[j] , outDir + "2D/")
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


#plt.show()

print(f'time taken: {(time.time()-s):.4f}' )