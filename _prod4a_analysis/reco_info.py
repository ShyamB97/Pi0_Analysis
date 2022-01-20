"""
Created on: 19/01/2022 19:00

Author: Shyam Bhuller

Description: Plot certain reconstructed quantities from ROOT files produced from the Analyser.
"""
### Place this in a script that is inteded to be ran ###
import sys

from sqlalchemy import true
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


def vector(x, y, z):
    """Creates a vector like record

    Args:
        x (Any): x component
        y (Any): y component
        z (Any): z component

    Returns:
        ak.Record: record structurd like a 3-vector
    """
    return ak.zip({"x" : x, "y" : y, "z" : z})


def magntiude(vec):
    """magnitude of 3-vector

    Args:
        vec (ak.Record created by vector): vector

    Returns:
        ak.Array: array of magnitudes
    """
    return (vec.x**2 + vec.y**2 + vec.z**2)**0.5


def dot(a, b):
    """dot product of 3-vector

    Args:
        a (ak.Record created by vector): first vector
        b (ak.Record created by vector): second vector

    Returns:
        ak.Array: array of dot products
    """
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z)


def prod(s, v):
    """product of scalar and vector

    Args:
        s (ak.Array or single number): scalar
        v (ak.Record created by vector): vector

    Returns:
        ak.Record created by vector: s * v
    """
    return vector(s * v.x, s * v.y, s * v.z)


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


save = False
outDir = "pi0_0p5GeV_100K/reco_info/"
bins = 50
allPairs = False
s = time.time()
file = uproot.open("ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
events = file["pduneana/beamana"]

#* number of showers
nHits = events["reco_daughter_PFP_nHits_collection"].array()
nDaughter = ak.count(nHits, 1)
#* leading + subleading energies
# get shower pairs
if allPairs is True:
    pairs = AllShowerPairs(nDaughter)
else:
    pairs = ShowerPairsByHits(nHits)

energy = events["reco_daughter_allShower_energy"].array()
energy_pair = GetPairValues(pairs, energy)
#flattened = ak.flatten(energy_pair)
sortedPairs = ak.sort(energy_pair, ascending=False)
leading = sortedPairs[:, :, 1:]
secondary = sortedPairs[:, :, :-1]

#mask = leading > 0
#leading = ak.where(mask, leading, [[]]*len(leading))

#* opeining angle
direction = vector( events["reco_daughter_allShower_dirX"].array(),
                    events["reco_daughter_allShower_dirY"].array(),
                    events["reco_daughter_allShower_dirZ"].array() )

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