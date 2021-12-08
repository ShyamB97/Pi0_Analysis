#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:49:07 2021

@author: sb16165
"""

import sys
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# custom imports
import Master
from Master import Unwrap, Conditional

import SelectionQuantities
from Plots import PlotHist, PlotHist2D, PlotHistComparison

def save(name):
    out = subDirectory + name + reference_filename + ".png"
    plt.savefig(out)
    plt.close()


# handle command line imports
args = sys.argv

if "--help" in args or len(args) == 1:
    print("usage: --file <root file> --outName <file name to append> --outDir <output directory> --beam <1/0>")
    exit(0)
if "--file" in args:
    root_filename = args[args.index("--file") + 1]
    print(root_filename)
else:
    print("no file chosen!")
    exit(1)
if "--outName" in args:
    reference_filename = "_" + args[args.index("--outName") + 1]
else:
    reference_filename = "_" + root_filename[19:-5]
if "--outDir" in args:
    subDirectory = args[args.index("--outDir") + 1] + "/"
else:
    subDirectory = root_filename[19:-5] + "/"
if "--beam" in args:
    beamData = bool(int(args[args.index("--beam") + 1]))
else:
    beamData = True

start_time = time.time()
print("Opening root file: " + root_filename)
os.makedirs(subDirectory, exist_ok=True)
data = Master.Data(root_filename)


print("getting daughter info...")
start_pos = data.start_pos()
direction = data.direction()
energy = data.energy()
mc_energy = data.mc_energy()
nHits = data.nHits()
cnn_em = data.cnn_em()
cnn_track = data.cnn_track()
cnn_score = SelectionQuantities.CNNScore(cnn_em, cnn_track)
pandoraTag = data.pandoraTag()

if beamData is True:
    print("getting beam info...")
    beam_start_pos = data.beam_start_pos()
    beam_end_pos = data.beam_end_pos()

print("getting daughter truth info...")
true_start_pos = data.true_start_pos()
true_end_pos = data.true_end_pos()

cut = False

if cut is True:
    # create the mask
    mask = Master.SelectionMask()
    # initilise mask (set the shape and set all values to 1)
    mask.InitiliseMask(nHits)


    # select data based on pandora Tag
    mask.CutMask(pandoraTag, 11, Conditional.EQUAL)


    # cut the data using the mask
    cnn_score = mask.Apply(cnn_score)
    start_pos = mask.Apply(start_pos)

    direction = mask.Apply(direction)

    energy = mask.Apply(energy)
    mc_energy = mask.Apply(mc_energy)
    nHits = mask.Apply(nHits)

    beam_start_pos = mask.Apply(beam_start_pos, True)
    beam_end_pos = mask.Apply(beam_end_pos, True)

    true_start_pos = mask.Apply(true_start_pos)
    true_end_pos = mask.Apply(true_end_pos)


# calculate quantities from selected data
nHits = Unwrap(nHits)

print("energy residual")
energyResidual = Unwrap( (energy - mc_energy)/ mc_energy )

if beamData is True:
    print("angle between beam and daughters")
    beam_angle = Unwrap(SelectionQuantities.BeamTrackShowerAngle(beam_start_pos, beam_end_pos, direction))

print("angle between daughters and mc particle")
mc_angle = Unwrap(SelectionQuantities.DaughterRecoMCAngle(true_start_pos, true_end_pos, direction))

print("separation")
pair_separation = Unwrap(SelectionQuantities.ShowerPairSeparation(start_pos))

print("pair angle")
pair_angle = Unwrap(SelectionQuantities.ShowerPairAngle(start_pos, direction))

print("true pair angle")
true_dist = true_start_pos - true_end_pos
true_dir = true_dist.Normalise()
true_opening_angle = Unwrap(SelectionQuantities.ShowerPairAngle(start_pos, true_dir))

print("pair energy")
pair_energies, pair_leading, pair_second = SelectionQuantities.ShowerPairEnergy(start_pos, energy)

pair_energies = Unwrap(pair_energies)
pair_leading = Unwrap(pair_leading)
pair_second = Unwrap(pair_second)

print("Invariant mass")
inv_mass = Unwrap(SelectionQuantities.InvariantMass(start_pos, direction, energy))


# save plots of data
print("Saving Plots:")
nbins = 100

print("Pandora Tag")
pandoraTag = Unwrap(pandoraTag)
unique, amount = np.unique(pandoraTag, return_counts=True)
unique = list(unique)
unique[0] = str(unique[0]) + "\nN/A"
unique[1] = str(unique[1]) + "\nshower"
unique[2] = str(unique[2]) + "\ntrack"
plt.bar(unique, amount)
plt.xlabel("Pandora tag of daughters")
plt.ylabel("Number of daughters")
plt.tight_layout()
save("pandora_tag")

print("CNN score")
cnn_score = Unwrap(cnn_score)
PlotHist(cnn_score[cnn_score != -999], nbins, "CNN scores of beam daughters")
save("cnn_score")


print("collection plane hits")
nHits_plot = nHits[nHits != 0]
PlotHist(nHits_plot[nHits_plot < 101], nbins, "Number of collection plane hits")
save("collection_hits")


print("hits vs mc angle")
PlotHist2D(nHits, mc_angle, nbins, [-999, max(nHits)], [0, np.pi], "Number of collection plane hits", "Angle between shower and MC parent (rad)")
save("hits_vs_mcAngle")


if beamData is True:
    print("hits vs beam angle")
    PlotHist2D(nHits, beam_angle, nbins, [-999, max(nHits)], [0, np.pi], "Number of collection plane hits", "Angle between beam track and daughter shower (rad)")
    save("hits_vs_beamAngle")


print("hits vs energy residual")
PlotHist2D(nHits, energyResidual, nbins, [-999, max(nHits)], [-1, 1], "Number of collection plane hits", "Reconstruted energy residual")
save("hits_vs_energyRes")


print("mc_energy vs energy residual")
mc_energy = Unwrap(mc_energy)
PlotHist2D(mc_energy, energyResidual, 100, [min(mc_energy), max(mc_energy)], [-1, 1], "mc energy of beam daughters (GeV)", "Reconstruted energy residual")
save("mcEnergy_vs_energyRes")


print("reco energy vs energy residual")
energy = Unwrap(energy)
PlotHist2D(energy, energyResidual, 100, [min(energy), max(energy)], [-1, 1], "reco energy of beam daughters (GeV)", "Reconstruted energy residual")
save("recoEnergy_vs_energyRes")


if beamData is True:
    print("beam angle")
    beam_angle = beam_angle[beam_angle != -999]
    PlotHist(beam_angle, nbins, "Angle between beam track and daughter shower (rad)")
    save("beam_daughterAngle")


print("pair separation")
pair_separation = pair_separation[pair_separation > 0]
PlotHist(pair_separation[pair_separation < 51], nbins, "Shower pair Separation (cm)")
save("pair_separation")


print("pair angle")
pair_angle = pair_angle[pair_angle != -999]
PlotHist(pair_angle, nbins, "Angle between shower pairs (rad)")
save("shower_pairAngle")


print("pair energies")
pair_leading = pair_leading[pair_leading > 0]
pair_second = pair_second[pair_second > 0]
PlotHistComparison(pair_leading, pair_second, nbins, xlabel="Shower energy (GeV)", label_1="shower with the most energy in a pair", label_2="shower with the least energy in a pair", alpha=0.5, density=False)
save("daughter_energies")


print("invariant mass")
inv_mass = inv_mass[inv_mass > 0]
PlotHist(inv_mass[inv_mass < 0.5], nbins, "Shower pair invariant mass (GeV)", "")
save("inv_mass")

print("true energy vs reco shower energy")
PlotHist2D(mc_energy[energy > 0], energy[energy > 0], nbins, xlabel="true energy (GeV)", ylabel="reconstructed shower energy(GeV)")
save("mc_reco_energy")

print("true opening angle")
PlotHist(true_opening_angle, nbins, "True shower pair angle (rad)")
save("true_opening_angle")

print("true pair angle vs pair angle")
PlotHist2D(true_opening_angle, pair_angle, nbins, xlabel="True shower pair angle (rad)", ylabel="Angle between shower pairs (rad)")
save("true_reco_angle")

print("opening angle residual")
angle_residual = pair_angle - true_opening_angle
PlotHist(angle_residual, nbins, r"$\theta_{reco} - \theta_{true}$ (rad)")
save("angle_residual")


print("done!")
print("ran in %s seconds. " % (time.time() - start_time) )