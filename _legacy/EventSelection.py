#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:08:52 2021

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
from Master import Unwrap, Conditional, Vector3, paramString, paramAxesLabels, StringToCond, GetCondString, InvertConditional

import SelectionQuantities
from Plots import PlotHist, PlotHist2D, PlotHistComparison, Save

import Analyser

def ParseConfig(file_name):
    """
    Function to read a config file which contains a string of values that dictate which parameters
    are cut on, where to cut and the condition. Removes any strings after # to
    allow comments to be written.
    
    In the config file the three quantities is split by a space, and a new line for a different selection e.g:
    
    1 g 0.6
    2 l 50
    
    Translates to selecting events where:
    
    cnn score > 0.6
    nHits < 50
    
    ----- Parameters -----
    param         : quantities to cut on, int based on paramString dictionary
    conditional   : bounds for the cut; >, <, != or =.
    cut           : values to cut on, depends on the parameter
    file          : config file
    line          : line in the file
    ----------------------
    """
    param = []
    conditional = []
    cut = []
    file = open(file_name, "r") # open file
    # parse each line
    for line in file:
        line = line.split("#", 1)[0] # remove comments
        param.append( int(line.split(" ", 2)[0]) ) # get the paramter by splitting the string at the first two space bars
        conditional.append( StringToCond(line.split(" ", 2)[1]) ) # get middle characters and convert to Conditional class object
        cut.append( float(line.split(" ", 2)[2]) ) # get end characters
    return param, conditional, cut


def PlotPandoraTagging(tags, label="", title=""):
    """
    Plots Pandora tagging of each daughter events in a bar plot.
    (make sure you unwrap the data first!)
    ----- Parameters -----
    unique      : type of tags in the data, can be -999, 11 or 13
    amount      : number of times a tag occurs in data
    ----------------------
    """
    unique, amount = list(np.unique(tags, return_counts=True)) # get tags and the number of occurances
    
    # switch from numerical tags to words for easy interpretation
    for i in range(len(unique)):
        if unique[i] == -999:
            unique[i] = str(unique[i]) + "\nN/A"
            continue
        if unique[i] == 11:
            unique[i] = str(unique[i]) + "\nshower"
            continue
        if unique[i] == 13:
            unique[i] = str(unique[i]) + "\ntrack"
            continue
    # plot
    plt.bar(unique, amount, label=label)
    plt.xlabel(paramAxesLabels[0])
    plt.ylabel("Number of daughters")
    plt.title(title)
    if label != "": plt.legend()
    plt.tight_layout()


def PlotSingle(q, nbins):
    """
    Function holding code which plots all the selection quantities and saves them to the output
    directory. Constantly modified and in principle hard to condense into smaller code blocks.
    """
    print("Pandora Tag")
    PlotPandoraTagging(q[0])
    Save("pandora_tag", subDirectory, reference_filename)


    print("CNN score")
    PlotHist(q[1][q[1] != -999], nbins, paramAxesLabels[1])
    Save("cnn_score", subDirectory, reference_filename)


    print("collection plane hits")
    nHits_plot = q[2][q[2] != 0]
    PlotHist(nHits_plot, nbins, paramAxesLabels[2])
    Save("collection_hits", subDirectory, reference_filename)


    print("hits vs mc angle")
    PlotHist2D(q[2], q[5], nbins, [-999, max(q[2])], [0, np.pi], paramAxesLabels[2], paramAxesLabels[5])
    Save("hits_vs_mcAngle", subDirectory, reference_filename)


    if beamData is True:
        print("hits vs beam angle")
        PlotHist2D(q[2], q[4], nbins, [-999, max(q[2])], [0, np.pi], paramAxesLabels[2], paramAxesLabels[4])
        Save("hits_vs_beamAngle", subDirectory, reference_filename)


    print("hits vs energy residual")
    PlotHist2D(q[2], q[3], nbins, [-999, max(q[2])], [-1, 1], paramAxesLabels[2], paramAxesLabels[3])
    Save("hits_vs_energyRes", subDirectory, reference_filename)


    print("mc_energy vs energy residual")
    PlotHist2D(q[8], q[3], 100, [min(q[8]), max_energy], [-1, 1], paramAxesLabels[8], paramAxesLabels[3])
    Save("mcEnergy_vs_energyRes", subDirectory, reference_filename)


    print("reco energy vs energy residual")
    PlotHist2D(q[7], q[3], 100, [min(q[7]), max_energy], [-1, 1], paramAxesLabels[7], paramAxesLabels[3])
    Save("recoEnergy_vs_energyRes", subDirectory, reference_filename)


    if beamData is True:
        print("beam angle")
        q[4] = q[4][q[4] != -999]
        PlotHist(q[4], nbins, paramAxesLabels[4])
        Save("beam_daughterAngle", subDirectory, reference_filename)


    print("pair separation")
    q[9] = q[9][q[9] > 0]
    PlotHist(q[9][q[9] < 51], nbins, paramAxesLabels[9])
    Save("pair_separation", subDirectory, reference_filename)


    print("pair angle")
    PlotHist(q[10][q[10] != -999], nbins, paramAxesLabels[10])
    Save("shower_pairAngle", subDirectory, reference_filename)


    print("pair energies")
    PlotHistComparison(q[11][q[11] > 0], q[12][q[12] > 0], nbins, xlabel="Shower energy (GeV)", label_1=paramAxesLabels[11], label_2=paramAxesLabels[12], alpha=0.5, density=False)
    Save("daughter_energies", subDirectory, reference_filename)


    print("invariant mass")
    inv_mass = q[13][q[13] > 0]
    PlotHist(inv_mass, nbins, paramAxesLabels[13], sf=3)
    Save("inv_mass", subDirectory, reference_filename)


    print("true energy vs reco shower energy")
    PlotHist2D(q[8][q[7] > 0], q[7][q[7] > 0], nbins, [min(q[8]), max_energy], [min(q[7]), max_energy], xlabel=paramAxesLabels[8], ylabel=paramAxesLabels[7])
    Save("mc_reco_energy", subDirectory, reference_filename)

    
    print("true pair angle")
    PlotHist(q[14], nbins, paramAxesLabels[14])
    Save("true_pair_angle", subDirectory, reference_filename)

    
    print("true pair angle vs pair angle")
    PlotHist2D(q[14], q[10], nbins, xlabel=paramAxesLabels[14], ylabel=paramAxesLabels[10])
    Save("true_reco_angle", subDirectory, reference_filename)

    
    print("opening angle residual")
    angle_residual = q[10] - q[14]
    PlotHist(angle_residual, nbins, r"$\theta_{reco} - \theta_{true}$ (rad)")
    Save("angle_residual", subDirectory, reference_filename)


    print("start hits")
    print("Clyinder with radius " + str(r) + "cm, length " + str(l_max - l_min) + "cm, shower start is offset from the center by " + str(l_max - (l_max - l_min)/2) + "cm")
    PlotHist(q[6], nbins, paramAxesLabels[6])
    Save("start_hits", subDirectory, reference_filename)
    
    print("reco shower energy")
    PlotHist(q[7][q[7] > 0], nbins, paramAxesLabels[7])
    Save("reco_energy", subDirectory, reference_filename)


    print("true energy")
    true_energy = q[8][q[8] > 0]
    #true_energy = true_energy[true_energy <= max_energy]
    PlotHist(true_energy, nbins, paramAxesLabels[8])
    Save("true_energy", subDirectory, reference_filename)


    print("nHits vs true energy")
    PlotHist2D(q[2], q[8], nbins, [-999, max(q[2])], [0, max_energy], xlabel=paramAxesLabels[2], ylabel=paramAxesLabels[8])
    Save("nHits_vs_trueEnergy", subDirectory, reference_filename)


    print("nHits vs reco shower energy")
    PlotHist2D(q[2], q[7], nbins, [-999, max(q[2])], [0, max(q[7])], xlabel=paramAxesLabels[2], ylabel=paramAxesLabels[7])
    Save("nHits_vs_recoEnergy", subDirectory, reference_filename)
    
    print("opening angles vs start hits?")
    #PlotHist2D(q[10], q[6])


def PlotComparison(param, conditional, conditional_inv, cut, q_1, q_2, nbins):
    """
    Function holding code which plots all the selection quantities from both the selected and
    rejected data, saving them to the output directory.
    Constantly modified and in principle hard to condense into smaller code blocks.
    ----- Parameters -----
    l_1        : label for selected data, describes the cut made
    l_2        : label for rejected data, describes the cut made
    ----------------------
    """
    if len(param) == 1:
        l_1 = paramString[param[0]] + GetCondString(conditional[0]) + str(cut[0])
        l_2 = paramString[param[0]] + GetCondString(conditional_inv[0]) + str(cut[0])
    else:
        l_1 = "selected events"
        l_2 = "rejected events"
        print("Data selected:")
        for i in range(len(param)):
            print(paramString[param[i]] + GetCondString(conditional[i]) + str(cut[i]))
        print("Data rejected:")
        for i in range(len(param)):
            print(paramString[param[i]] + GetCondString(conditional_inv[i]) + str(cut[i]))
    
    
    print("Pandora Tag")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotPandoraTagging(q_1[0], title=l_1)
    plt.subplot(122)
    PlotPandoraTagging(q_2[0], title=l_2)
    Save("pandora_tag", subDirectory, reference_filename)
    
    
    print("CNN score")
    PlotHistComparison(q_1[1][q_1[1] != -999], q_2[1][q_2[1] != -999], nbins, alpha=0.5, xlabel=paramAxesLabels[1], label_1=l_1, label_2=l_2)
    Save("cnn_score", subDirectory, reference_filename)
    
    
    print("collection plane hits")
    nHits_1 = q_1[2][q_1[2] != 0]
    nHits_2 = q_2[2][q_2[2] != 0]
    
    PlotHistComparison(nHits_1[nHits_1 < 101], nHits_2[nHits_2 < 101], nbins, alpha=0.5, xlabel=paramAxesLabels[2], label_1=l_1, label_2=l_2)
    Save("collection_hits", subDirectory, reference_filename)
    
    
    print("hits vs mc angle")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[2], q_1[5], nbins, [-999, 510], [0, np.pi], paramAxesLabels[2], paramAxesLabels[5], title=l_1)
    
    plt.subplot(122)
    PlotHist2D(q_2[2], q_2[5], nbins, [-999, 510], [0, np.pi], paramAxesLabels[2], paramAxesLabels[5], title=l_2)
    Save("hits_vs_mcAngle", subDirectory, reference_filename)
    
    
    if beamData is True:
        print("hits vs beam angle")
        plt.figure(2, (12, 5))
        plt.subplot(121)
        PlotHist2D(q_1[2], q_1[4], nbins, [-999, 501], [0, np.pi], paramAxesLabels[2], paramAxesLabels[4], title=l_1)
        plt.subplot(122)
        PlotHist2D(q_2[2], q_2[4], nbins, [-999, 501], [0, np.pi], paramAxesLabels[2], paramAxesLabels[4], title=l_2)
        Save("hits_vs_beamAngle", subDirectory, reference_filename)
    
    
    print("hits vs energy residual")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[2], q_1[3], nbins, [-999, max(q_1[2])], [-1, 1], paramAxesLabels[2], paramAxesLabels[3], title=l_1)
    plt.subplot(122)
    PlotHist2D(q_2[2], q_2[3], nbins, [-999, max(q_2[2])], [-1, 1], paramAxesLabels[2], paramAxesLabels[3], title=l_2)
    Save("hits_vs_energyRes", subDirectory, reference_filename)
    
    
    print("mc energy vs energy residual")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[8], q_1[3], nbins, [min(q_1[8]), max_energy], [-1, 1], paramAxesLabels[8], paramAxesLabels[3], title=l_1)
    plt.subplot(122)
    PlotHist2D(q_2[8], q_2[3], nbins, [min(q_2[8]), max_energy], [-1, 1], paramAxesLabels[8], paramAxesLabels[3], title=l_2)
    Save("mc_energy_vs_energyRes", subDirectory, reference_filename)
    
    
    print("energy vs energy residual")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[7], q_1[3], nbins, [min(q_1[7]), max_energy], [-1, 1], paramAxesLabels[7], paramAxesLabels[3], title=l_1)
    plt.subplot(122)
    PlotHist2D(q_2[7], q_2[3], nbins, [min(q_2[7]), max_energy], [-1, 1], paramAxesLabels[7], paramAxesLabels[3], title=l_2)
    Save("energy_vs_energyRes", subDirectory, reference_filename)
    
    
    if beamData is True:
        print("beam angle")
        beam_angle_1 = q_1[4][q_1[4] != -999]
        beam_angle_2 = q_2[4][q_2[4] != -999]
        PlotHistComparison(beam_angle_1, beam_angle_2, nbins, alpha=0.5, xlabel=paramAxesLabels[4], label_1=l_1, label_2=l_2)
        Save("beam_daughterAngle", subDirectory, reference_filename)
    
    
    print("pair separation")
    pair_separation_1 = q_1[9][q_1[9] > 0]
    pair_separation_2 = q_2[9][q_2[9] > 0]
    PlotHistComparison(pair_separation_1[pair_separation_1 < 51], pair_separation_2[pair_separation_2 < 51], nbins, alpha=0.5, xlabel=paramAxesLabels[9], label_1=l_1, label_2=l_2)
    Save("pair_separation", subDirectory, reference_filename)
    
    
    print("pair angle")
    pair_angle_1 = q_1[10][q_1[10] != -999]
    pair_angle_2 = q_2[10][q_2[10] != -999]
    PlotHistComparison(pair_angle_1, pair_angle_2, nbins, alpha=0.5, xlabel=paramAxesLabels[10], label_1=l_1, label_2=l_2)
    Save("shower_pairAngle", subDirectory, reference_filename)
    
    
    print("leading shower energy")
    q_1[11] = q_1[11].flatten()
    q_2[11] = q_2[11].flatten()
    PlotHistComparison(q_1[11][q_1[11] > 0], q_2[11][q_2[11] > 0], nbins, alpha=0.5, xlabel=paramAxesLabels[11], label_1=l_1, label_2=l_2)
    Save("leading_shower_energies", subDirectory, reference_filename)
    
    
    print("sub-leading shower energy")
    q_1[12] = q_1[12].flatten()
    q_2[12] = q_2[12].flatten()
    PlotHistComparison(q_1[12][q_1[12] > 0], q_2[12][q_2[12] > 0], nbins, alpha=0.5, xlabel=paramAxesLabels[12], label_1=l_1, label_2=l_2)
    Save("subleading_shower_energies", subDirectory, reference_filename)
    
    
    print("invariant mass")
    inv_mass_1 = q_1[13][q_1[13] > 0]
    inv_mass_2 = q_2[13][q_2[13] > 0]
    PlotHistComparison(inv_mass_1[inv_mass_1 < 0.5], inv_mass_2[inv_mass_2 < 0.5], nbins, alpha=0.5, xlabel=paramAxesLabels[13], sf=3, label_1=l_1, label_2=l_2)
    Save("inv_mass", subDirectory, reference_filename)


    print("true energy vs reco shower energy")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[8][q_1[7] > 0], q_1[7][q_1[7] > 0], nbins, [min(q_1[8]), max_energy], [min(q_1[7]), max_energy], xlabel=paramAxesLabels[8], ylabel=paramAxesLabels[7], title=l_1)
    plt.subplot(122)
    PlotHist2D(q_2[8][q_2[7] > 0], q_2[7][q_2[7] > 0], nbins, [min(q_2[8]), max_energy], [min(q_2[7]), max_energy], xlabel=paramAxesLabels[8], ylabel=paramAxesLabels[7], title=l_2)
    Save("mc_reco_energy", subDirectory, reference_filename)
    
    
    print("true pair angle")
    PlotHistComparison(q_1[14], q_2[14], nbins, alpha=0.5, xlabel=paramAxesLabels[14], label_1=l_1, label_2=l_2)
    Save("true_pair_angle", subDirectory, reference_filename)

    
    print("true pair angle vs pair angle")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[14], q_1[10], nbins, xlabel=paramAxesLabels[14], ylabel=paramAxesLabels[10], title=l_1)
    plt.subplot(122)
    PlotHist2D(q_2[14], q_2[10], nbins, xlabel=paramAxesLabels[14], ylabel=paramAxesLabels[10], title=l_2)
    Save("true_reco_angle", subDirectory, reference_filename)

    
    print("opening angle residual")
    angle_residual_1 = q_1[10] - q_1[14]
    angle_residual_2 = q_2[10] - q_2[14]
    PlotHistComparison(angle_residual_1, angle_residual_2, nbins, alpha=0.5, xlabel=r"$\theta_{reco} - \theta_{true}$ (rad)", label_1=l_1, label_2=l_2)
    Save("angle_residual", subDirectory, reference_filename)


    print("start hits")
    print("Clyinder with radius " + str(r) + "cm, length " + str(l_max - l_min) + "cm, shower start is offset from the center by " + str(l_max - (l_max - l_min)/2) + "cm")
    PlotHistComparison(q_1[6], q_2[6], nbins, alpha=0.5, xlabel=paramAxesLabels[6], label_1=l_1, label_2=l_2)
    Save("start_hits", subDirectory, reference_filename)


    print("reco shower energy")
    PlotHistComparison(q_1[7][q_1[7] > 0], q_2[7][q_2[7] > 0], nbins, alpha=0.5, xlabel=paramAxesLabels[7], label_1=l_1, label_2=l_2)
    Save("reco_energy", subDirectory, reference_filename)


    print("true energy")
    true_energy_1 = q_1[8][q_1[8] > 0]
    true_energy_1 = true_energy_1[true_energy_1 <= max_energy]
    true_energy_2 = q_2[8][q_2[8] > 0]
    true_energy_2 = true_energy_2[true_energy_2 <= max_energy]
    PlotHistComparison(true_energy_1, true_energy_2, nbins, alpha=0.5, xlabel=paramAxesLabels[8], label_1=l_1, label_2=l_2)
    Save("true_energy", subDirectory, reference_filename)


    print("nHits vs true energy")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[2], q_1[8], nbins, [-999, max(q_1[2])], [0, max_energy], xlabel=paramAxesLabels[2], ylabel=paramAxesLabels[8], title=l_1)
    plt.subplot(122)
    PlotHist2D(q_2[2], q_2[8], nbins, [-999, max(q_1[2])], [0, max_energy], xlabel=paramAxesLabels[2], ylabel=paramAxesLabels[8], title=l_2)
    Save("nHits_vs_trueEnergy", subDirectory, reference_filename)
    

    print("nHits vs reco shower energy")
    plt.figure(2, (12, 5))
    plt.subplot(121)
    PlotHist2D(q_1[2], q_1[7], nbins, [-999, max(q_1[2])], [0, max(q_1[7])], xlabel=paramAxesLabels[2], ylabel=paramAxesLabels[7], title=l_1)
    plt.subplot(122)
    PlotHist2D(q_2[2], q_2[7], nbins, [-999, max(q_2[2])], [0, max(q_2[7])], xlabel=paramAxesLabels[2], ylabel=paramAxesLabels[8], title=l_2)
    Save("nHits_vs_recoEnergy", subDirectory, reference_filename)


start_time = time.time()

# handle command line imports
args = sys.argv

root_filename, reference_filename, subDirectory, beamData, plotType, cutData, config_file, param, conditional, cut = Master.ParseCommandLine(args)

# start hit constants
r = 3 # cm
l_min = -1 # cm
l_max = 4 # cm

max_energy = 0.5 # beam momentum/energy

# if a config file is specified, read the contents
if config_file != None:
    param, conditional, cut = ParseConfig(config_file)

print("Opening root file: " + root_filename)
os.makedirs(subDirectory, exist_ok=True)
data = Master.Data(root_filename)


q_1 = Unwrap(Analyser.CalculateParameters(data, param, conditional, cut, cutData, beamData, [r, l_min, l_max]))

if cutData is True and plotType == "both":
    conditional_inv = [InvertConditional(c) for c in conditional]
    q_2 = Unwrap(Analyser.CalculateParameters(data, param, conditional_inv, cut, cutData, beamData, [r, l_min, l_max]))


print("Saving Plots:")
nbins = 100

print(plotType)

if cutData is True:
    if plotType == "single":
        for i in range(len(param)):
            print(paramString[param[i]] + GetCondString(conditional[i]) + str(cut[i]))
        PlotSingle(q_1, nbins)
    elif plotType == "both":
        PlotComparison(param, conditional, conditional_inv, cut, q_1, q_2, nbins)

else:
    PlotSingle(q_1, nbins)

print("done!")
print("ran in %s seconds. " % (time.time() - start_time) )

