"""
Created on: 08/02/2022 17:00

Author: Shyam Bhuller

Description: Compare results of different filters for Pi0MC. 
"""

import os
import awkward as ak
import time
import numpy as np
# custom modules
import Plots
import Master
import vector

import matplotlib.pyplot as plt


def MCTruth(events, sortEnergy, photons):
    """Calculate true shower pair quantities.

    Args:
        events (Master.Event): events to process
        sortEnergy (ak.array): mask to sort shower pairs by true energy
        photons (ak.Array): mask of photons in MC truth

    Returns:
        tuple of ak.Array: calculated quantities
    """
    #* get the primary pi0
    mask_pi0 = np.logical_and(events.trueParticles.number == 1, events.trueParticles.pdg == 111)

    #* get pi0 -> two photons only
    #mask_daughters = ak.all(events.trueParticles.pdg != 11, axis=1)
    #mask_daughters = np.logical_and(events.trueParticles.mother == 1, mask_daughters)

    #* compute start momentum of dauhters
    p_daughter = events.trueParticles.momentum[photons]
    sum_p = ak.sum(p_daughter, axis=1)
    sum_p = vector.magntiude(sum_p)
    p_daughter_mag = vector.magntiude(p_daughter)
    p_daughter_mag = p_daughter_mag[sortEnergy]

    #* compute true opening angle
    angle = np.arccos(vector.dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

    #* compute invariant mass
    e_daughter = events.trueParticles.energy[photons]
    inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    p_pi0 = events.trueParticles.momentum[mask_pi0]
    p_pi0 = vector.magntiude(p_pi0)
    return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0


def RecoMC(events, sortEnergy):
    """Calculate reconstructed shower pair quantities.

    Args:
        events(Master.Event): events to process
        sortEnergy (ak.array): mask to sort shower pairs by true energy

    Returns:
        tuple of ak.Array: calculated quantities + array which masks null shower pairs
    """
    #! not needed if we match to true photons
    sortedPairs = ak.unflatten(events.recoParticles.energy[sortEnergy], 1, 0)
    leading = sortedPairs[:, :, 1:]
    secondary = sortedPairs[:, :, :-1]

    #* opening angle
    #direction_pair = GetPairValues(pairs, r_dir)
    direction_pair = ak.unflatten(events.recoParticles.direction[sortEnergy], 1, 0)
    direction_pair_mag = vector.magntiude(direction_pair)
    angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

    #* Invariant Mass
    inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    # create momentum vectors assuming these are photon showers (not neccessarily true) -> E * dir
    shower_mom = vector.prod(sortedPairs, direction_pair)
    pi0_momentum = vector.magntiude(ak.sum(shower_mom, axis=2))/1000

    null_dir = np.logical_or(direction_pair[:, :, 1:].x == -999, direction_pair[:, :, :-1].x == -999) # mask shower pairs with invalid direction vectors
    null = np.logical_or(leading < 0, secondary < 0) # mask of shower pairs with invalid energy
    
    #* filter null data
    pi0_momentum = np.where(null_dir, -999, pi0_momentum)
    pi0_momentum = np.where(null, -999, pi0_momentum)

    leading = leading/1000
    secondary = secondary/1000

    leading = np.where(null, -999, leading)
    leading = np.where(null_dir, -999, leading)
    secondary = np.where(null, -999, secondary)
    secondary = np.where(null_dir, -999, secondary)

    angle = np.where(null, -999, angle)
    angle = np.where(null_dir, -999, angle)

    inv_mass = inv_mass/1000
    inv_mass = np.where(null, -999,inv_mass)
    inv_mass = np.where(null_dir, -999,inv_mass)

    return inv_mass, angle, leading, secondary, pi0_momentum, ak.unflatten(null, 1, 0)


def Error(reco, true, null):
    """Calcuate fractional error, filter null data and format data for plotting.

    Args:
        reco (ak.array): reconstructed quantity
        true (ak.array): true quantity
        null (ak.array): mask for events without shower pairs, reco direction or energy

    Returns:
        tuple of np.array: flattened numpy array of errors, reco and truth
    """
    true = true[null]
    true = ak.where( ak.num(true, 1) > 0, true, [np.nan]*len(true) )
    reco = ak.flatten(reco, 1)[null]
    print(f"reco pairs: {len(reco)}")
    print(f"true pairs: {len(true)}")
    error = (reco / true) - 1
    return ak.to_numpy(ak.ravel(error)), ak.to_numpy(ak.ravel(reco)), ak.to_numpy(ak.ravel(true))


@Master.timer
def CreateFilteredEvents(events : Master.Event, nDaughters=None):
    valid, photons = Master.Pi0MCFilter(events, nDaughters)

    shower_dir = events.recoParticles.direction[valid]
    print(f"Number of showers events: {ak.num(shower_dir, 0)}")
    photon_dir = vector.normalize(events.trueParticles.momentum)[photons][valid]

    showers, selection_mask, angles = events.MatchMC(photon_dir, shower_dir, returnAngles=True)

    Plots.PlotHist(ak.ravel(angles[:, 0]), 50, "Angle between matched shower and true photon 1 (rad)")
    Plots.PlotHist(ak.ravel(angles[:, 1]), 50, "Angle between matched shower and true photon 2 (rad)")
    

    reco_filters = [valid, showers, selection_mask]
    true_filters = [valid, selection_mask]

    return events.Filter(reco_filters, true_filters), photons[valid][selection_mask]


def CalculateQuantities(events, nDaughter : int = None):
    global names
    filtered, photons = CreateFilteredEvents(events, nDaughter)
    print(f"Number of events after filtering: {ak.num(filtered.recoParticles.nHits, 0)}")
    mct = MCTruth(filtered, filtered.SortByTrueEnergy(), photons)
    rmc = RecoMC(filtered, filtered.SortByTrueEnergy())

    # keep track of events with no shower pairs
    null = ak.flatten(rmc[-1], -1)
    null = ak.num(null, 1) > 0

    error = []
    reco = []
    true = []
    for i in range(len(names)):
        print(names[i])
        e, r, t = Error(rmc[i], mct[i], null)
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
filters = [None, 2, 3]
f_l = [">= 2 daughters", "2 daughters", "3 daughters"]

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
        data = rs[i][j]
        data = data[data > -900]
        Plots.PlotHist(ts[i][j], bins=bins, xlabel=t_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
    if save is True: Plots.Save( names[j] , outDir + "truths/")

if save is True: os.makedirs(outDir + "reco/", exist_ok=True)
for j in range(len(names)):
    plt.figure()
    for i in range(len(filters)):
        data = rs[i][j]
        data = data[data > -900]
        Plots.PlotHist(data, bins=bins, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
    if save is True: Plots.Save( names[j] , outDir + "reco/")

if save is True: os.makedirs(outDir + "fractional_error/", exist_ok=True)
for j in range(len(names)):
    plt.figure()
    for i in range(len(filters)):
        data = es[i][j]
        data = data[data > fe_range[0]]
        data = data[data < fe_range[1]]
        Plots.PlotHist(data, bins=bins, xlabel=e_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
    if save is True: Plots.Save( names[j] , outDir + "fractional_error/")

#plt.show()

print(f'time taken: {(time.time()-s):.4f}' )