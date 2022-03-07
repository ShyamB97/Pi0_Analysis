"""
Created on: 19/01/2022 19:00

Author: Shyam Bhuller

Description: Plot certain reconstructed quantities from ROOT files produced from the Analyser.
"""
import os
import awkward as ak
import time
import Plots
import itertools
import numpy as np
import Master
from Master import timer
import matplotlib.pyplot as plt
import vector


save = False
outDir = "pi0_0p5GeV_100K/match_MC/reco_info/"
bins = 50
allPairs = False
s = time.time()
events = Master.Data("../ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")


#* number of showers
nDaughter = ak.count(events.recoParticles.nHits, 1)

#* filter events
r_mask = nDaughter == 2

t_dir = vector.normalize(events.trueParticles.momentum)

photons = events.trueParticles.mother == 1 # get only primary daughters
photons = events.trueParticles.pdg == 22 # get only photons
t_mask = ak.num(photons[photons], -1) == 2 # exclude pi0 -> e+ + e- + photons

valid = np.logical_and(r_mask, t_mask) # events which have 2 reco daughters and correct pi0 decay

null = ak.any(events.recoParticles.direction.x == -999, 1) # exclude events where direction couldn't be calculated

valid = np.logical_and(valid, np.logical_not(null))

direction = events.recoParticles.direction[valid]

showers, selection_mask = Master.MatchMC()

direction = direction[showers][selection_mask]
nHits = events.recoParticles.nHits[valid][showers][selection_mask]

#* leading + subleading energies
# get shower pairs
if allPairs is True:
    pairs = Master.AllShowerPairs(nDaughter)
else:
    pairs = Master.ShowerPairsByHits(nHits)

energy = events.recoParticles.energy[valid][showers][selection_mask]
energy_pair = Master.GetPairValues(pairs, energy)
sortedPairs = ak.sort(energy_pair, ascending=True)
leading = sortedPairs[:, :, 1:]
secondary = sortedPairs[:, :, :-1]

#* opening angle

direction_pair = Master.GetPairValues(pairs, direction)
direction_pair_mag = vector.magntiude(direction_pair)
angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

#* Invariant Mass
inv_mass = (2 * energy_pair[:, :, 1:] * energy_pair[:, :, :-1] * (1 - np.cos(angle)))**0.5

#* sum energy
total_energy = leading + secondary # assuming the daughters are photon showers

#* pi0 momentum
# create momentum vectors assuming these are photon showers (not neccessarily true) -> E * dir
shower_mom = vector.prod(energy_pair, direction_pair)
pi0_momentum = vector.magntiude(ak.sum(shower_mom, axis=2))
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