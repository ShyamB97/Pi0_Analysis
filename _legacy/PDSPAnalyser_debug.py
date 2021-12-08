#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:49:07 2021

@author: sb16165
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Master
from Master import Unwrap, Vector3, Conditional
from Plots import PlotHist, PlotHist2D, PlotHistComparison
import SelectionQuantities
import Analyser
import Plots
import sys



#data = Master.Data("ROOTFiles/pduneana_Prod4_1GeV_2_9_21.root")
#data = Master.Data("pi0Test_output_PDSPProd4_MC_1GeV_SCE_DataDriven_reco_1K_1_24_03_21.root")
#data = Master.Data("ROOTFiles/pi0Test_output_PDSPProd4_MC_1GeV_SCE_DataDriven_reco_3p5K_29_04_21.root");
file = "ROOTFiles/diphoton_test.root"
#file = "ROOTFiles/diphoton_10K.root"
config = "selection_full.txt"
data = Master.Data(file)
nbins = 25
evd_ID = Master.GetData(file, "EventID")

r = 3 # cm
l_min = -1 # cm
l_max = 4 # cm
cylinder = [r, l_min, l_max]

param, conditional, cut = Master.ParseConfig(config)

true_momentum = data.true_momentum()
start_pos = data.true_start_pos()
direction = true_momentum.Normalise()
energy = true_momentum.Magnitude()
mc_energy = energy
energy = Unwrap(energy)

nHits = data.nHits()
cnn_em = data.cnn_em()
cnn_track = data.cnn_track()
pandoraTag = data.pandoraTag()

# custom data
hit_radial = data.hit_radial()
hit_longitudinal = data.hit_longitudinal()

print("shower pairs")
#shower_pairs = SelectionQuantities.GetShowerPairs(start_pos)
shower_pairs = SelectionQuantities.GetShowerPairs_Hits(nHits)
shower_pairs = Unwrap(shower_pairs)
#f = Analyser.CalculateFeatures(data, mc=True)

#Plots.PlotHist(f["pair second"])

"""
data = q[q != -999] # reject null data
hist, bins = np.histogram(data, nbins) # bin data
x = (bins[:-1] + bins[1:])/2  # get center of bins
x = np.array(x, dtype=float) # convert from object to float
binWidth = bins[1] - bins[0] # calculate bin widths

uncertainty = np.sqrt(hist) # calculate poisson uncertainty if each bin
plt.bar(x, hist, binWidth, yerr=uncertainty, capsize=2, color="C0")
binWidth = round(binWidth, 3)
plt.ylabel("Number of events (bin width=" + str(binWidth) + ")")
plt.xlabel("Shower pair invariant mass (GeV)")
plt.tight_layout()
"""
#Calculated quantities
"""
print("calculating cnn_score...")
cnn_em = data.cnn_em()
cnn_track = data.cnn_track()
cnn_score = SelectionQuantities.CNNScore(cnn_em, cnn_track)
"""


"""
print("energy residual")
#energyResidual = Unwrap( (energy - mc_energy)/ mc_energy)

#print("angle between beam and daughters")
#beam_angle = Unwrap(SelectionQuantities.BeamTrackShowerAngle(beam_start_pos, beam_end_pos, direction))

print("angle between daughters and mc particle")
#mc_angle = Unwrap(SelectionQuantities.DaughterRecoMCAngle(true_start_pos, true_end_pos, direction))

print("shower pairs")
shower_pairs = SelectionQuantities.GetShowerPairs(start_pos)

print("separation")
#pair_separation = SelectionQuantities.ShowerPairSeparation(start_pos, shower_pairs)

print("pair angle")
pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, direction)

print("pair energy")
pair_energies, pair_leading, pair_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, energy)

#pair_energies = Unwrap(pair_energies)
#pair_leading = Unwrap(pair_leading)
#pair_second = Unwrap(pair_second)

print("start hits")
start_hits = SelectionQuantities.GetShowerStartHits(hit_radial, hit_longitudinal)

print("Invariant mass")
#inv_mass = Unwrap(SelectionQuantities.InvariantMass(pair_angle, pair_energies))


#Plots.LeastSqrFit(inv_mass, nbins=50)
"""
"""
start_hits = Unwrap(start_hits)
PlotHist(start_hits, 100, "Shower start hits")
"""


"""
pandoraTag = Unwrap(pandoraTag)
unique, amount = np.unique(pandoraTag, return_counts=True)
unique = list(unique)
unique[0] = str(unique[0]) + "\nN/A"
unique[1] = str(unique[1]) + "\nshower"
unique[2] = str(unique[2]) + "\ntrack"
plt.bar(unique, amount)
"""

"""
cnn_score = Unwrap(cnn_score)
PlotHist(cnn_score[cnn_score != -999], 100, "CNN scores of beam daughters")
"""

"""
pair_separation = pair_separation[pair_separation > 0]
PlotHist(pair_separation[pair_separation < 51], 100, "Shower pair Separation (cm)")
"""

"""
PlotHist2D(nHits, mc_angle, 100, [-999, 510], [0, np.pi], "Number of collection plane hits", "Angle between shower and MC parent (rad)")
"""

"""
PlotHist2D(nHits, beam_angle, 100, [-999, 501], [0, np.pi], "Number of collection plane hits", "Angle between beam track and daughter shower (rad)")
"""

"""
PlotHist2D(nHits, energyResidual, 100, [-999, 501], [-1, 1], "Number of collection plane hits", "Reconstruted energy residual")
"""

"""
inv_mass = inv_mass[inv_mass > 0]
PlotHist(inv_mass[inv_mass < 0.5], 100, "Shower pair invariant mass (GeV)", "", 3)
"""

"""
pair_angle = pair_angle[pair_angle != -999]
PlotHist(pair_angle, 100, "Angle between shower pairs (rad)")
"""

"""
beam_angle = beam_angle[beam_angle != -999]
PlotHist(beam_angle, 100, "Angle between beam track and daughter shower (rad)")
"""

"""
nHits = Unwrap(nHits)
nHits = nHits[nHits != 0]
PlotHist(nHits[nHits < 101], 100, "Number of collection plane hits")
"""


"""
pair_leading = pair_leading[pair_leading > 0]
pair_second = pair_second[pair_second > 0]
PlotHistComparison(pair_leading, pair_second, 100, xlabel="Shower energy (GeV)", label_1="shower with the most energy in a pair", label_2="shower with the least energy in a pair", alpha=0.5, density=False)
"""