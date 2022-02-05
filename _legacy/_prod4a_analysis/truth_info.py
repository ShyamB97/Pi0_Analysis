"""
Created on: 19/01/2022 19:14

Author: Shyam Bhuller

Description: Plot certain MC truth quantities from ROOT files produced from the Analyser.
MC is not backtracked from the reconstructed particle
"""
### Place this in a script that is inteded to be ran ###
import sys
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_base_libs') # need this to import custom scripts from _base_libs
######
import numpy as np

#! Can't use Master.DataList to retrieve G4 values.
import uproot
import awkward as ak
import time
import os
import Plots
import matplotlib.pyplot as plt
from vector import *


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


save = False
outDir = "pi0_0p5GeV_100K/match_MC/truth_info/"
bins = 50
start = time.time()
file = uproot.open("../ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
events = file["pduneana/beamana"]
momentum = vector(events["g4_pX"].array(), events["g4_pY"].array(), events["g4_pZ"].array())
direction = normalize(momentum)
pdg = events["g4_Pdg"].array()
startE = events["g4_startE"].array()
g4_num = events["g4_num"].array()
g4_mother = events["g4_mother"].array()

#* filter events
nHits = events["reco_daughter_PFP_nHits_collection"].array()
nDaughter = ak.count(nHits, 1)

r_mask = nDaughter == 2

r_dir = vector( events["reco_daughter_allShower_dirX"].array(),
                    events["reco_daughter_allShower_dirY"].array(),
                    events["reco_daughter_allShower_dirZ"].array() )

photons = g4_mother == 1 # get only primary daughters
photons = pdg == 22 # get only photons
t_mask = ak.num(photons[photons], -1) == 2 # exclude pi0 -> e+ + e- + photons

valid = np.logical_and(r_mask, t_mask) # events which have 2 reco daughters and correct pi0 decay

null = ak.any(r_dir.x == -999, 1) # exclude events where direction couldn't be calculated

valid = np.logical_and(valid, np.logical_not(null))

r_dir = r_dir[valid]
direction = direction[photons][valid]

_, selection_mask = MatchMC(direction, r_dir) # showers are not needed for truth info

momentum = momentum[valid][selection_mask]
pdg = pdg[valid][selection_mask]
startE = startE[valid][selection_mask]
g4_num = g4_num[valid][selection_mask]
g4_mother = g4_mother[valid][selection_mask]

#* get the primary pi0
mask_pi0 = np.logical_and(g4_num == 1, pdg == 111)

#* get pi0 -> two photons only
mask_daughters = ak.all(pdg != 11, axis=1)
mask_daughters = np.logical_and(g4_mother == 1, mask_daughters)

#* pi0 momentum
p_pi0 = momentum[mask_pi0]
p_pi0 = magntiude(p_pi0)


#* compute start momentum of dauhters
p_daughter = momentum[mask_daughters]
sum_p = ak.sum(p_daughter, axis=1)
sum_p = magntiude(sum_p)
p_daughter_mag = magntiude(p_daughter)
p_daughter_mag = ak.sort(p_daughter_mag, ascending=True) # sort by leading photon

#* compute true opening angle
angle = np.arccos(dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

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