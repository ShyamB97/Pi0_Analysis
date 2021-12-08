#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:46:49 2021

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
from Master import Unwrap, Vector3, Conditional

import SelectionQuantities
import Plots


def save(name):
    out = subDirectory + name + reference_filename + ".png"
    plt.savefig(out)
    plt.close()


# handle command line imports
args = sys.argv

root_files = []

if "--help" in args or len(args) == 1:
    print("usage: --file1 <root file> --file2 --outName <file name to append> --outDir <output directory> --beam <1/0>")
    exit(0)
if "--file1" in args:
    root_files.append( args[args.index("--file1") + 1] )
    print(root_files[0])
else:
    print("no file 1 chosen!")
    exit(1)
if "--file2" in args:
    root_files.append(args[args.index("--file2") + 1] )
    print(root_files[1])
else:
    print("no file 2 chosen!")
    exit(1)
if "--outName" in args:
    reference_filename = "_" + args[args.index("--outName") + 1]
else:
    reference_filename = "_" + root_files[0][19:-5] + root_files[1][19:-5]
if "--outDir" in args:
    subDirectory = args[args.index("--outDir") + 1] + "/"
else:
    subDirectory = root_files[0][19:-5] + root_files[1][19:-5] + "/"
if "--beam" in args:
    beamData = bool( int(args[args.index("--beam") + 1]) )
else:
    beamData = True


start_time = time.time()


def Calculate(root_file):
    print("Opening root file 1: " + root_file)
    os.makedirs(subDirectory, exist_ok=True)
    data = Master.Data(root_file)

    print("getting daughter info...")
    start_pos = data.start_pos()
    direction = data.direction()
    energy = data.energy()
    mc_energy = data.mc_energy()
    nHits = data.nHits()
    cnn_em = data.cnn_em()
    cnn_track = data.cnn_track()
    cnn_score = Unwrap(SelectionQuantities.CNNScore(cnn_em, cnn_track))
    cnn_score = cnn_score[cnn_score != -999]
    pandoraTag = data.pandoraTag()


    if beamData is True:
        print("getting beam info...")
        beam_start_pos = data.beam_start_pos()
        beam_end_pos = data.beam_end_pos()

    
    print("getting daughter truth info...")
    true_start_pos = data.true_start_pos()
    true_end_pos = data.true_end_pos()


    # calculate quantities from selected data
    nHits = Unwrap(nHits)
    
    print("energy residual")
    energyResidual = Unwrap( (energy - mc_energy)/ mc_energy )
    
    if beamData is True:
        print("angle between beam and daughters")
        beam_angle = Unwrap(SelectionQuantities.BeamTrackShowerAngle(beam_start_pos, beam_end_pos, direction))
    else:
        beam_angle = -999
    
    print("angle between daughters and mc particle")
    mc_angle = Unwrap(SelectionQuantities.DaughterRecoMCAngle(true_start_pos, true_end_pos, direction))
    
    print("separation")
    pair_separation = Unwrap(SelectionQuantities.ShowerPairSeparation(start_pos))
    
    print("pair angle")
    pair_angle = Unwrap(SelectionQuantities.ShowerPairAngle(start_pos, direction))
    
    print("pair energy")
    pair_energies, pair_leading, pair_second = SelectionQuantities.ShowerPairEnergy(start_pos, energy)
    
    pair_energies = Unwrap(pair_energies)
    pair_leading = Unwrap(pair_leading)
    pair_second = Unwrap(pair_second)
    
    print("Invariant mass")
    inv_mass = Unwrap(SelectionQuantities.InvariantMass(start_pos, direction, energy))
    
    return [Unwrap(pandoraTag), cnn_score, nHits, energyResidual, beam_angle, mc_angle, pair_separation, pair_angle, pair_energies, inv_mass, Unwrap(mc_energy), Unwrap(energy)]


q_1 = Calculate(root_files[0])
q_2 = Calculate(root_files[1])

print("Saving Plots:")
nbins = 100
l_1 = root_files[0]
l_2 = root_files[1]


print("Pandora Tag")
plt.figure(2, (12, 5))
plt.subplot(121)

unique, amount = list(np.unique(q_1[0], return_counts=True))
unique[0] = str(unique[0]) + "\nN/A"
unique[1] = str(unique[1]) + "\nshower"
unique[2] = str(unique[2]) + "\ntrack"
plt.bar(unique, amount)
plt.xlabel("Pandora tag of daughters")
plt.ylabel("Number of daughters")
plt.title(l_1)
plt.tight_layout()

plt.subplot(122)

unique, amount = list(np.unique(q_2[0], return_counts=True))
unique[0] = str(unique[0]) + "\nN/A"
unique[1] = str(unique[1]) + "\nshower"
unique[2] = str(unique[2]) + "\ntrack"
plt.bar(unique, amount)
plt.xlabel("Pandora tag of daughters")
plt.ylabel("Number of daughters")
plt.title(l_2)
plt.tight_layout()

save("pandora_tag")


print("CNN score")
Plots.PlotHistComparison(q_1[1], q_2[1], nbins, alpha=0.5, xlabel="CNN scores of beam daughters", label_1=l_1, label_2=l_2)
save("cnn_score")


print("collection plane hits")
nHits_1 = q_1[2][q_1[2] != 0]
nHits_2 = q_2[2][q_2[2] != 0]

Plots.PlotHistComparison(nHits_1[nHits_1 < 101], nHits_2[nHits_2 < 101], nbins, alpha=0.5, xlabel="Number of collection plane hits", label_1=l_1, label_2=l_2)
save("collection_hits")


print("hits vs mc angle")
plt.figure(2, (12, 5))
plt.subplot(121)
Plots.PlotHist2D(q_1[2], q_1[5], nbins, [-999, 510], [0, np.pi], "Number of collection plane hits", "Angle between shower and MC parent (rad)", title=l_1)

plt.subplot(122)
Plots.PlotHist2D(q_2[2], q_2[5], nbins, [-999, 510], [0, np.pi], "Number of collection plane hits", "Angle between shower and MC parent (rad)", title=l_2)
save("hits_vs_mcAngle")


if beamData is True:
    print("hits vs beam angle")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    Plots.PlotHist2D(q_1[2], q_1[4], nbins, [-999, 501], [0, np.pi], "Number of collection plane hits", "Angle between beam track and daughter shower (rad)", title=l_1)
    plt.subplot(122)
    Plots.PlotHist2D(q_2[2], q_2[4], nbins, [-999, 501], [0, np.pi], "Number of collection plane hits", "Angle between beam track and daughter shower (rad)", title=l_2)
    save("hits_vs_beamAngle")


print("hits vs energy residual")
plt.figure(2, (12, 5))
plt.subplot(121)
Plots.PlotHist2D(q_1[2], q_1[3], nbins, [-999, max(q_1[2])], [-1, 1], "Number of collection plane hits", "Reconstruted energy residual", title=l_1)
plt.subplot(122)
Plots.PlotHist2D(q_2[2], q_2[3], nbins, [-999, max(q_2[2])], [-1, 1], "Number of collection plane hits", "Reconstruted energy residual", title=l_2)
save("hits_vs_energyRes")


print("mc energy vs energy residual")
plt.figure(2, (12, 5))
plt.subplot(121)
Plots.PlotHist2D(q_1[10], q_1[3], nbins, [-999, max(q_1[10])], [-1, 1], "mc energy of beam daughters (GeV)", "Reconstruted energy residual", title=l_1)
plt.subplot(122)
Plots.PlotHist2D(q_2[10], q_2[3], nbins, [-999, max(q_2[10])], [-1, 1], "mc energy of beam daughters (GeV)", "Reconstruted energy residual", title=l_2)
save("mc_energy_vs_energyRes")


print("energy vs energy residual")
plt.figure(2, (12, 5))
plt.subplot(121)
Plots.PlotHist2D(q_1[11], q_1[3], nbins, [-999, max(q_1[11])], [-1, 1], "reco energy of beam daughters (GeV)", "Reconstruted energy residual", title=l_1)
plt.subplot(122)
Plots.PlotHist2D(q_2[11], q_2[3], nbins, [-999, max(q_2[11])], [-1, 1], "reco energy of beam daughters (GeV)", "Reconstruted energy residual", title=l_2)
save("energy_vs_energyRes")


if beamData is True:
    print("beam angle")
    beam_angle_1 = q_1[4][q_1[4] != -999]
    beam_angle_2 = q_2[4][q_2[4] != -999]
    Plots.PlotHistComparison(beam_angle_1, beam_angle_2, nbins, alpha=0.5, xlabel="Angle between beam track and daughter shower (rad)", label_1=l_1, label_2=l_2)
    save("beam_daughterAngle")


print("pair separation")
pair_separation_1 = q_1[6][q_1[6] > 0]
pair_separation_2 = q_2[6][q_2[6] > 0]
Plots.PlotHistComparison(pair_separation_1[pair_separation_1 < 51], pair_separation_2[pair_separation_2 < 51], nbins, alpha=0.5, xlabel="Shower pair Separation (cm)", label_1=l_1, label_2=l_2)
save("pair_separation")


print("pair angle")
pair_angle_1 = q_1[7][q_1[7] != -999]
pair_angle_2 = q_2[7][q_2[7] != -999]
Plots.PlotHistComparison(pair_angle_1, pair_angle_2, nbins, alpha=0.5, xlabel="Angle between shower pairs (rad)", label_1=l_1, label_2=l_2)
save("shower_pairAngle")


print("shower energies")
q_1[8] = q_1[8].flatten()
q_2[8] = q_2[8].flatten()
Plots.PlotHistComparison(q_1[8][q_1[8] > 0], q_2[8][q_2[8] > 0], nbins, alpha=0.5, xlabel="Shower energy (GeV)", label_1=l_1, label_2=l_2)
save("shower_energies")


print("invariant mass")
inv_mass_1 = q_1[9][q_1[9] > 0]
inv_mass_2 = q_2[9][q_2[9] > 0]
Plots.PlotHistComparison(inv_mass_1[inv_mass_1 < 0.5], inv_mass_2[inv_mass_2 < 0.5], nbins, alpha=0.5, xlabel="Shower pair invariant mass (GeV)", sf=3, label_1=l_1, label_2=l_2)
save("inv_mass")

print("done!")
print("ran in %s seconds. " % (time.time() - start_time) )



