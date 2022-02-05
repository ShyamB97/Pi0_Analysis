"""
Created on: 19/01/2022 19:14

Author: Shyam Bhuller

Description: Plot certain MC truth quantities from ROOT files produced from the Analyser.
MC is not backtracked from the reconstructed particle
"""

import Master
import numpy as np
import awkward as ak
import time
import os
import Plots
import vector


save = False
outDir = "pi0_0p5GeV_100K/match_MC/truth_info/"
bins = 50
start = time.time()

events = Master.Event("../ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
direction = vector.normalize(events.trueParticles.momentum)


#* filter events
valid, photons = Master.Pi0MCFilter(events, 2)

r_dir = events.recoParticles.direction[valid]
direction = direction[photons][valid]

_, selection_mask = events.MatchMC(direction, r_dir) # showers are not needed for truth info

momentum = events.trueParticles.momentum[valid][selection_mask]
pdg = events.trueParticles.pdg[valid][selection_mask]
startE = events.trueParticles.energy[valid][selection_mask]
g4_num = events.trueParticles.number[valid][selection_mask]
g4_mother = events.trueParticles.mother[valid][selection_mask]

#* get the primary pi0
mask_pi0 = np.logical_and(g4_num == 1, pdg == 111)

#* get pi0 -> two photons only
mask_daughters = ak.all(pdg != 11, axis=1)
mask_daughters = np.logical_and(g4_mother == 1, mask_daughters)

#* pi0 momentum
p_pi0 = momentum[mask_pi0]
p_pi0 = vector.magntiude(p_pi0)


#* compute start momentum of dauhters
p_daughter = momentum[mask_daughters]
sum_p = ak.sum(p_daughter, axis=1)
sum_p = vector.magntiude(sum_p)
p_daughter_mag = vector.magntiude(p_daughter)
p_daughter_mag = ak.sort(p_daughter_mag, ascending=True) # sort by leading photon

#* compute true opening angle
angle = np.arccos(vector.dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

#* compute invariant mass
e_daughter = startE[mask_daughters]
inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5


#* plots
#Plots.PlotBar( ak.to_list(pdg[mask_daughters]), xlabel="Pdg codes" ) # for debugging
if save is True: os.makedirs(outDir, exist_ok=True)

Plots.PlotHist(ak.flatten(p_daughter_mag[:, 1:]), bins, "Leading photon energy (GeV)")
if save is True: Plots.Save("lead_photon_energy", outDir)
Plots.PlotHist(ak.flatten(p_daughter_mag[:, :-1]), bins, "Subleading photon energy (GeV)")
if save is True: Plots.Save("sub_photon_energy", outDir)
sum_p = sum_p[sum_p != 0] # exlude events where no data was captured
man_bins = np.linspace(0.4, ak.max(sum_p)+0.1, bins, True) # override default binning for correct plot view
Plots.PlotHist(sum_p, man_bins, "Sum photon energies (GeV)")
if save is True: Plots.Save("sum_photon_energy", outDir)
Plots.PlotHist(ak.flatten(angle), bins, "Opening angle (rad)")
if save is True: Plots.Save("angle", outDir)
man_bins = np.linspace(0, ak.mean(inv_mass)+0.1, bins, True)
Plots.PlotHist(ak.flatten(inv_mass), man_bins, "Invariant mass (GeV)")
if save is True: Plots.Save("inv_mass", outDir)
man_bins = np.linspace(0.4, ak.mean(p_pi0)+0.1, bins, True)
Plots.PlotHist(ak.flatten(p_pi0), man_bins, "$\pi^{0}$ momentum (GeV)")
if save is True: Plots.Save("pi0_mom", outDir)
print("time taken: " + str(time.time()-start))


#* Diphoton sample, plot true momentum distribution of initial photons, check it matches the generated distribution, plus positions
#? make more usable in the terminal? i.e. args and main func
#? add capability of plotting backtracked MC as well? (separate script??)