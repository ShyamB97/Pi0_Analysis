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


save = True
outDir = "pi0_0p5GeV_100K/truth_info/"
bins = 50
start = time.time()
file = uproot.open("ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
events = file["pduneana/beamana"]
momentum = vector(events["g4_pX"].array(), events["g4_pY"].array(), events["g4_pZ"].array())

pdg = events["g4_Pdg"].array()
startE = events["g4_startE"].array()
g4_num = events["g4_num"].array()
g4_mother = events["g4_mother"].array()

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
p_daughter_mag = ak.sort(p_daughter_mag, ascending=True) # sort by leading photon
#sum_p = ak.sum(p_daughter_mag, axis=1)

#* compute true opening angle
angle = np.arccos(dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

#* compute invariant mass
e_daughter = startE[mask_daughters]
inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

#* pi0 momentum
p_pi0 = momentum[mask_pi0]
p_pi0 = magntiude(p_pi0)

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
#! incorrect g4 information in existing root files
#TODO Edit analyser to get all particles in the truth tables, recreate data
#? make more usable in the terminal? i.e. args and main func
#? add capability of plotting backtracked MC as well? (separate script??)