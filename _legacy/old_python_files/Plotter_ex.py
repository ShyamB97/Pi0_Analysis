#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:49:07 2021

@author: sb16165

To run this code, in a unix console type ./Plotter_ex.py or
you can run it in spyder/your choice of python IDE
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt

def Unravel(data):
    """
    A function to flatten nested arrays from the root tree
    """
    _list = []
    for obj in data:
        for item in obj:
            _list.append(item)
    return np.array(_list)


### MAIN CODE ###
print("opening file")
file = uproot.open("pduneana_Prod4_1GeV_2_9_21.root") # open file with uproot

tree_pduneana = file['pduneana/beamana'] # access directory with data

print("accessing data")
# Get data, in this case the number of daughter hits on the collection plane, see
# https://wiki.dunescience.org/wiki/PDSPAnalyzer for the entire list (or open file with root browser)
reco_daughter_nHits_collection = tree_pduneana['reco_daughter_PFP_nHits_collection'].arrays(library="np")["reco_daughter_PFP_nHits_collection"]


reco_daughter_nHits_collection = Unravel(reco_daughter_nHits_collection) # flatten nested array 

reco_daughter_nHits_collection = reco_daughter_nHits_collection[reco_daughter_nHits_collection != 0] # exclude no hits
reco_daughter_nHits_collection = reco_daughter_nHits_collection[reco_daughter_nHits_collection <= 100] # only plot data upto 100 hits

# plot data
print("creating plot")
height, edges, _ = plt.hist(reco_daughter_nHits_collection, 100)
binWidth = round((edges[-1] - edges[0]) / len(edges), 2)
plt.ylabel("Number of events (bin width=" + str(binWidth) + ")")
plt.xlabel("number of collection plane hits")

plot_name = "plot.png"
plt.savefig(plot_name)

print("plot saved as " + plot_name)