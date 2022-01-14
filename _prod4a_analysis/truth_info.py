### Place this in a script that is inteded to be ran ###
import sys
sys.path.append('../_base_libs') # need this to import custom scripts in different directories
######

import os
import numpy as np

from Master import Unwrap, DataList, QUANTITY, ITEM, Conditional
from SelectionQuantities import CalculateQuantities
import Plots
from MC_lib import GetShowerPairValues

#! Can't use Master.DataList to retrieve G4 values.
import uproot
import awkward as ak
from enum import Enum
import itertools
import matplotlib.pyplot as plt

def vector(a, b, c):
    return ak.zip({"x" : a, "y" : b, "z" : c})

def magntiude(vec):
    return (vec.x**2 + vec.y**2 + vec.z**2)**0.5

def dot(a, b):
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z)

file = uproot.open("../ROOTFiles/diphoton_10K.root")
events = file["pduneana/beamana"]

start_pos = vector(events["g4_startX"].array(), events["g4_startY"].array(), events["g4_startZ"].array())
end_pos = vector(events["g4_endX"].array(), events["g4_endY"].array(), events["g4_endZ"].array())
momentum = vector(events["g4_pX"].array(), events["g4_pY"].array(), events["g4_pZ"].array())

pdg = events["g4_Pdg"].array()
startE = events["g4_startE"].array()
mass = events["g4_mass"].array()
g4_num = events["g4_num"].array()
g4_mother = events["g4_mother"].array()

#* get the primary pi0
mask_pi0 = np.logical_and(g4_num == 1, pdg == 111)

#* get pi0 -> two photons only
mask_daughters = ak.all(pdg != 11, axis=1)
mask_daughters = np.logical_and(g4_mother == 1, mask_daughters)

#* compute start momentum of dauhters
p_daughter = momentum[mask_daughters]
p_daughter_mag = magntiude(p_daughter)
sum_p = ak.sum(p_daughter_mag, axis=1)

#* compute true opening angle
angle = np.arccos(dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

#* compute invariant mass
e_daughter = startE[mask_daughters]
inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

#* pi0 momentum
p_pi0 = momentum[mask_pi0]
p_pi0 = ( p_pi0.x**2 + p_pi0.y**2 + p_pi0.z**2 )**0.5

#* plots
Plots.PlotBar( ak.to_list(pdg[mask_daughters]), xlabel="Pdg codes" )

Plots.PlotHist(sum_p[sum_p != 0], xlabel="Sum photon energies (GeV)")
Plots.PlotHist(ak.flatten(angle), xlabel="Opening angle (rad)")
Plots.PlotHist(ak.flatten(inv_mass), xlabel="Invariant mass (GeV)")
Plots.PlotHist(ak.flatten(p_pi0), xlabel="$\pi^{0}$ momentum (GeV)")

#* Diphoton sample, plot true momentum distribution of initial photons, check it matches the generated distribution, plus positions
#! incorrect g4 information in existing root files
#TODO Edit analyser to get all particles in the truth tables, recreate data
