"""
Created on: 04/03/2022 16:01

Author: Shyam Bhuller

Description: Script which will read an ntuple ROOT file of MC and analyse reconstructed shower properties.
"""

import argparse
import os
import awkward as ak
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# custom modules
import Plots
import Master
import vector


if __name__ == "__main__":

    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True subleading shower energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error (GeV)", "Opening angle fractional error (rad)", "Leading shower energy fractional error (GeV)", "Subleading shower energy fractional error (GeV)", "$\pi^{0}$ momentum fractional error (GeV)"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]
    f_l = ["2 showers", "3 showers, unmerged", "angular vector sum", "spatial vector sum", "angular scalar sum", "spatial scalar sum"]
    fe_range = [-1, 1]

    parser = argparse.ArgumentParser(description="Study em shower merging for pi0 decays")
    parser.add_argument("-f", "--file", dest="file", type=str, default="ROOTFiles/pi0_0p5GeV_100K_5_7_21.root", help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms")
    parser.add_argument("-s", "--save", dest="save", type=bool, default=False, help="whether to save the plots")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/shower_merge/", help="directory to save plots")
    #args = parser.parse_args("-a energy".split()) #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    main()