"""
Created on: 19/01/2022 19:00

Author: Shyam Bhuller

Description: Plot certain reconstructed quantities from ROOT files produced from the Analyser.
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
from Master import timer
import matplotlib.pyplot as plt
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
def AllShowerPairs(nd):
    """Get all shower pair combinations, excluding duplicates and self paired.

    Args:
        nd (Array): number of daughters in an event

    Returns:
        list: Jagged array of pairs per event
    """
    pairs = []
    for i in range(len(nd)):
        comb = itertools.combinations(range(nd[i]), 2)
        pairs.append(list(comb))
    return pairs


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
    return ak.to_list(pairs)


save = True
outDir = "pi0_0p5GeV_100K/match_MC/reco_info/"
bins = 50
allPairs = False
s = time.time()
file = uproot.open("ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
events = file["pduneana/beamana"]

nHits = events["reco_daughter_PFP_nHits_collection"].array()
direction = vector( events["reco_daughter_allShower_dirX"].array(),
                    events["reco_daughter_allShower_dirY"].array(),
                    events["reco_daughter_allShower_dirZ"].array() )


#* number of showers
nDaughter = ak.count(nHits, 1)

#* filter events
r_mask = nDaughter == 2

t_mom = vector(events["g4_pX"].array(), events["g4_pY"].array(), events["g4_pZ"].array())
t_dir = normalize(t_mom)

t_num = events["g4_num"].array()
t_mother =events["g4_mother"].array()
t_pdg = events["g4_Pdg"].array()


photons = t_mother == 1 # get only primary daughters
photons = t_pdg == 22 # get only photons
t_mask = ak.num(photons[photons], -1) == 2 # exclude pi0 -> e+ + e- + photons
# Plots.PlotBar(ak.to_list(t_pdg[t_mask]))

valid = np.logical_and(r_mask, t_mask) # events which have 2 reco daughters and correct pi0 decay

null = ak.any(direction.x == -999, 1) # exclude events where direction couldn't be calculated

valid = np.logical_and(valid, np.logical_not(null))

direction = direction[valid]
t_dir = t_dir[photons][valid]

showers, selection_mask = MatchMC(t_dir, direction)

direction = direction[showers][selection_mask]
nHits = nHits[valid][showers][selection_mask]

#* leading + subleading energies
# get shower pairs
if allPairs is True:
    pairs = AllShowerPairs(nDaughter)
else:
    pairs = ShowerPairsByHits(nHits)

energy = events["reco_daughter_allShower_energy"].array()[valid][showers][selection_mask]
energy_pair = GetPairValues(pairs, energy)
sortedPairs = ak.sort(energy_pair, ascending=True)
leading = sortedPairs[:, :, 1:]
secondary = sortedPairs[:, :, :-1]

#* opening angle

direction_pair = GetPairValues(pairs, direction)
direction_pair_mag = magntiude(direction_pair)
angle = np.arccos(dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

#* Invariant Mass
inv_mass = (2 * energy_pair[:, :, 1:] * energy_pair[:, :, :-1] * (1 - np.cos(angle)))**0.5

#* sum energy
total_energy = leading + secondary # assuming the daughters are photon showers

#* pi0 momentum
# create momentum vectors assuming these are photon showers (not neccessarily true) -> E * dir
shower_mom = prod(energy_pair, direction_pair)
pi0_momentum = magntiude(ak.sum(shower_mom, axis=2))
null = np.invert(np.logical_or(leading < 0, secondary < 0)) # mask of shower pairs with invalid energy


#* plots

if allPairs is False:
    pi0_momentum = ak.unflatten(pi0_momentum, 1)[null]
    pi0_momentum = ak.ravel(pi0_momentum)
# flatten data and remove invalid data i.e. -999
null = ak.ravel(null)
if allPairs is True:
    pi0_momentum = ak.ravel(pi0_momentum)[null] # pi0_momentum has different nested shape than the rest

total_energy = ak.ravel(total_energy)[null]
leading = ak.ravel(leading)[null]
secondary = ak.ravel(secondary)[null]
angle = ak.ravel(angle)[null]
inv_mass = ak.ravel(inv_mass)[null]

if save is True: os.makedirs(outDir, exist_ok=True)

Plots.PlotBar(ak.to_numpy(nDaughter), xlabel="Number of reconstructed daughter objects per event")
if save is True: Plots.Save("nDaughters", outDir)
Plots.PlotHist(total_energy/1000, bins, "Sum shower energy (GeV))")
if save is True: Plots.Save("sum_energy", outDir)
Plots.PlotHist(leading/1000, bins, "Leading shower energy (GeV)")
if save is True: Plots.Save("leading_energy", outDir)
Plots.PlotHist(secondary/1000, bins, "Subleading shower energy (GeV)")
if save is True: Plots.Save("subleading_energy", outDir)
Plots.PlotHist(angle, bins, "Opening angle (rad)")
if save is True: Plots.Save("angle", outDir)
Plots.PlotHist(inv_mass/1000, bins, "Invariant mass (GeV)")
if save is True: Plots.Save("mass", outDir)
Plots.PlotHist(pi0_momentum/1000, bins, "$\pi^{0}$ momentum (GeV)")
if save is True: Plots.Save("pi0_momentum", outDir)
print(f'time taken: {(time.time()-s):.4f}' )
if save is False:
    plt.show()

#? make more usable in the terminal? i.e. args and main func