"""
Created on: 22/01/2022 15:41

Author: Shyam Bhuller

Description: compare recontructed MC to MC truth.
"""
### Place this in a script that is inteded to be ran ###
import sys
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_base_libs') # need this to import custom scripts in different directories
######
import os
import uproot
import awkward as ak
import time
import Plots
import itertools
import numpy as np
import matplotlib.pyplot as plt
from Master import timer
from vector import *


@timer
def MatchMC(true, reco):
    """ Matches Reconstructed showers to true photons and selected the best events
        i.e. ones which have both errors of less than 0.25 radians. Only works for
        events with two reconstructed showers and two true photons per event.

    Args:
        true (ak.Record created by vector): direction of true photons
        reco (ak.Record created by vector): direction of reco showers

    Returns:
        ak.Array: shower indices in order of true photon number
        ak.Array: mask which indicates which pi0 decays are "good"
    """
    # angle of all reco showers wrt to each true photon per event i.e. error
    angle_error_0 = angle(reco, true[:, 0])
    angle_error_1 = angle(reco, true[:, 1])

    # get smallest angle wrt to each true photon
    m_0 = ak.unflatten(ak.min(angle_error_0, -1), 1)
    m_1 = ak.unflatten(ak.min(angle_error_1, -1), 1)
    angles = ak.concatenate([m_0, m_1], -1)
    
    # get shower which had the smallest angle
    #NOTE reco showers are now sorted by true photon number essentially. 
    m_0 = ak.unflatten(ak.argmin(angle_error_0, -1), 1)
    m_1 = ak.unflatten(ak.argmin(angle_error_1, -1), 1)
    showers = ak.concatenate([m_0, m_1], -1)
    
    # get events where both reco MC angles are less than 0.25 radians
    mask = np.logical_and(angles[:, 0] < 0.25, angles[:, 1] < 0.25)
    
    # check how many showers had the same reco match to both true particles
    same_match = showers[:, 0][mask] == showers[:, 1][mask]
    print(f"number of events where both photons match to the same shower: {ak.count(ak.mask(same_match, same_match) )}")
    
    return showers, mask


@timer
def GetPairValues(pairs, value):
    """get shower pair values, in pairs

    Args:
        pairs (list): shower pairs per event
        value (ak.Array): values to retrieve

    Returns:
        ak.Array: paired showers values per event
    """
    paired = []
    for i in range(len(pairs)):
        pair = pairs[i]
        evt = []
        for j in range(len(pair)):
            if len(pair[j]) > 0:
                evt.append( [value[i][pair[j][0]], value[i][pair[j][1]]] )
            else:
                evt.append([])
        paired.append(evt)
    return ak.Array(paired)


@timer
def ShowerPairsByHits(hits):
    """pair reconstructed showers in an event by the number of hits.
    pairs the two largest showers per event.

    Args:
        hits (Array): number of collection plane hits of daughters per event

    Returns:
        [list]: shower pairs (maximum of one per event), note lists are easier to iterate through than np or ak arrays, hence the conversion
    """
    showers = ak.argsort(hits, ascending=False) # shower number sorted by nHits
    mask = ak.count(showers, 1) > 1
    showers = ak.pad_none(showers, 2, clip=True) # only keep two largest showers
    showers = ak.where( mask, showers, [[]]*len(mask) )
    pairs = ak.unflatten(showers, 1)
    pairs = ak.argsort(pairs, -1)
    return ak.to_list(pairs)

@timer
def MCTruth():
    global sortindex #! come up with better way!
    #* get the primary pi0
    mask_pi0 = np.logical_and(g4_num == 1, pdg == 111)

    #* get pi0 -> two photons only
    mask_daughters = ak.all(pdg != 11, axis=1)
    mask_daughters = np.logical_and(g4_mother == 1, mask_daughters)

    #* compute start momentum of dauhters
    p_daughter = momentum[mask_daughters]
    sum_p = ak.sum(p_daughter, axis=1)
    sum_p = magntiude(sum_p)
    p_daughter_mag = magntiude(p_daughter)
    sortindex = ak.argsort(p_daughter_mag, ascending=True) # hacked way to sort reco showers by true energy
    #p_daughter_mag = ak.sort(p_daughter_mag, ascending=True) # sort by leading photon
    p_daughter_mag = p_daughter_mag[sortindex]

    #* compute true opening angle
    angle = np.arccos(dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

    #* compute invariant mass
    e_daughter = startE[mask_daughters]
    inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    p_pi0 = momentum[mask_pi0]
    p_pi0 = magntiude(p_pi0)
    return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0

@timer
def RecoMC():
    #* leading + subleading energies
    # get shower pairs
    #pairs = ShowerPairsByHits(nHits)
    #global energy_pair, sortindex
    #energy_pair = ak.flatten(GetPairValues(pairs, energy), -1)
    #sortedPairs = ak.sort(energy_pair, ascending=True)
    #sortedPairs = ak.unflatten(energy_pair[sortindex], 1, 0) # sort showers by true photon energy
    #leading = sortedPairs[:, :, 1:]
    #secondary = sortedPairs[:, :, :-1]
    #! not needed if we match to true photons
    sortedPairs = ak.unflatten(energy[sortindex], 1, 0)
    leading = sortedPairs[:, :, 1:]
    secondary = sortedPairs[:, :, :-1]

    #* opening angle
    #direction_pair = GetPairValues(pairs, r_dir)
    direction_pair = ak.unflatten(r_dir[sortindex], 1, 0)
    direction_pair_mag = magntiude(direction_pair)
    angle = np.arccos(dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

    #* Invariant Mass
    inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

    #* sum energy
    #total_energy = leading + secondary # assuming the daughters are photon showers

    #* pi0 momentum
    # create momentum vectors assuming these are photon showers (not neccessarily true) -> E * dir
    shower_mom = prod(sortedPairs, direction_pair)
    pi0_momentum = magntiude(ak.sum(shower_mom, axis=2))/1000

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
    true = true[null]
    true = ak.where( ak.num(true, 1) > 0, true, [np.nan]*len(true) )
    reco = ak.flatten(reco, 1)[null]
    print(f"reco pairs: {len(reco)}")
    print(f"true pairs: {len(true)}")
    error = reco - true
    return ak.to_numpy(ak.ravel(error)), ak.to_numpy(ak.ravel(reco)), ak.to_numpy(ak.ravel(true))


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
            d = data[i][data[i] > ranges[i][0]]
            d = d[d < ranges[i][1]]
        Plots.PlotHist(d, bins, labels[i])
        if save is True: Plots.Save(names[i], subDirectory)


def PlotReco(reco):
    #Plots.PlotBar(ak.to_numpy(nDaughter), xlabel="Number of reconstructed daughter objects per event")
    #if save is True: Plots.Save("nDaughters", outDir)
    #Plots.PlotHist(total_energy/1000, bins, "Sum shower energy (GeV))")
    #if save is True: Plots.Save("sum_energy", outDir)
    outDirReco = outDir + "reco/"
    reco[reco==-999]=None
    if save is True: os.makedirs(outDirReco, exist_ok=True)
    Plots.PlotHist(reco[2], bins, "Leading shower energy (GeV)")
    if save is True: Plots.Save("leading_energy", outDirReco)
    Plots.PlotHist(reco[3], bins, "Subleading shower energy (GeV)")
    if save is True: Plots.Save("subleading_energy", outDirReco)
    Plots.PlotHist(reco[1], bins, "Opening angle (rad)")
    if save is True: Plots.Save("angle", outDirReco)
    Plots.PlotHist(reco[0], bins, "Invariant mass (GeV)")
    if save is True: Plots.Save("mass", outDirReco)
    Plots.PlotHist(reco[4], bins, "$\pi^{0}$ momentum (GeV)")
    if save is True: Plots.Save("pi0_momentum", outDirReco)


save = True
outDir = "pi0_0p5GeV_100K/match_MC/quality/"
bins = 50
s = time.time()
file = uproot.open("ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
events = file["pduneana/beamana"]
#* truth information
momentum = vector(events["g4_pX"].array(), events["g4_pY"].array(), events["g4_pZ"].array())
t_dir = normalize(momentum)

pdg = events["g4_Pdg"].array()
startE = events["g4_startE"].array()
g4_num = events["g4_num"].array()
g4_mother = events["g4_mother"].array()

#* reco information
nHits = events["reco_daughter_PFP_nHits_collection"].array()
r_dir = vector( events["reco_daughter_allShower_dirX"].array(),
                events["reco_daughter_allShower_dirY"].array(),
                events["reco_daughter_allShower_dirZ"].array() )

energy = events["reco_daughter_allShower_energy"].array()

#* filter
nDaughter = ak.count(nHits, 1)

r_mask = nDaughter == 2
photons = g4_mother == 1 # get only primary daughters
photons = pdg == 22 # get only photons
t_mask = ak.num(photons[photons], -1) == 2 # exclude pi0 -> e+ + e- + photons

# Plots.PlotBar(ak.to_list(t_pdg[t_mask]))

valid = np.logical_and(r_mask, t_mask) # events which have 2 reco daughters and correct pi0 decay

null = ak.any(r_dir.x == -999, 1) # exclude events where direction couldn't be calculated

valid = np.logical_and(valid, np.logical_not(null))


r_dir = r_dir[valid]
t_dir = t_dir[photons][valid]

showers, selection_mask = MatchMC(t_dir, r_dir)

r_dir = r_dir[showers][selection_mask]
nHits = nHits[valid][showers][selection_mask]
energy = energy[valid][showers][selection_mask]

momentum = momentum[valid][selection_mask]
pdg = pdg[valid][selection_mask]
energy_photon = startE[photons][valid][selection_mask]
startE = startE[valid][selection_mask]
g4_num = g4_num[valid][selection_mask]
g4_mother = g4_mother[valid][selection_mask]


mct = MCTruth()
rmc = RecoMC()

# keep track of events with no shower pairs
null = ak.flatten(rmc[-1], -1)
null = ak.num(null, 1) > 0

# plot names and labels
names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
x_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True secondary shower energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
y_l = ["Invariant mass error (GeV)", "Opening angle error (rad)", "Leading shower energy error (GeV)", "Secondary shower energy error (GeV)", "$\pi^{0}$ momentum error (GeV)"]

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

x_r = [[-998, max(true[0])], [-998, max(true[1])], [-998, max(true[2])], [-998, max(true[3])], [-998, max(true[4])]]
y_r = [[-998, max(error[0])], [-998, max(error[1])], [-998, max(error[2])], [-998, max(error[3])], [-998, max(error[4])]]


if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
for j in range(len(error)):
    for i in range(len(true)):
        Plots.PlotHist2D(true[i], error[j], bins, x_range=x_r[i], y_range=y_r[j], xlabel=x_l[i], ylabel=y_l[j])
        if save is True: Plots.Save( names[j]+"_"+names[i] , outDir + "2D/")


#* Plot errors
PlotSingle(error, y_r, y_l, bins, outDir + "errors/", names, save)

Plots.PlotHist2D(true[2], error[2]/true[2], 50, y_range=[-999, 1], xlabel=x_l[2], ylabel="Leading energy fractional error")
if save is True: Plots.Save( "fractional_leading" , outDir + "2D/")
Plots.PlotHist2D(true[3], error[3]/true[3], 50, y_range=[-999, 1], xlabel=x_l[3],ylabel="Secondary energy fractional error")
if save is True: Plots.Save( "fractional_subleading" , outDir + "2D/")

PlotReco(reco)

#? plot truths not needed
PlotSingle(true, x_r, x_l, bins, outDir + "truths/", names, save)
print(f'time taken: {(time.time()-s):.4f}' )