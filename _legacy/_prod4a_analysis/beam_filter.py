"""
Created on: 19/01/2022 18:48

Author: Shyam Bhuller

Description: Code to demonstrate how to filter beam PFParticles, a pre-selection
previously done in the LArSoft analyser. Uses new scheme of awkward arrays.
"""
import uproot
import awkward as ak
import numpy as np

#* open files
#!NOTE only works with root files created with the analyser after the creation date of this code.
file = uproot.open("../ROOTFiles/test_new.root")
events = file["pduneana/beamana"]

#* retrieve neccessary quantities
cnn = events["reco_daughter_PFP_nHits_collection"].array() # use CNN score as an example 
pfpNum = events["reco_PFP_ID"].array() # PFParticle number
pfpMother = events["reco_PFP_Mother"].array() # PFParticle's mother (-999 if none)
beamID = events["beamNum"].array() # number of the particle which was the beam

#* create mask
mask = pfpMother == beamID # we want PFP's whose mother is the beam
mask = np.logical_or(mask, pfpNum == beamID) # in this special case (particle gun sim), also include the beam in the Array of showers to keep.

#* apply mask to data
cnn = ak.to_list(cnn[mask])

#TODO when writing new code, include this as a core function, to retain previous functionality
#TODO add option to not include the beam in kept events