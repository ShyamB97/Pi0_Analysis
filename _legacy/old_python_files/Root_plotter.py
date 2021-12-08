# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import uproot
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

def GetFileNames(directory="/ROOT_files", _filter=None):
    ROOTFiles = []
    path = os.getcwd() + directory
    
    for _, _, files in os.walk(path):
        for file in files:
            if file.endswith(".root"):
                path_file = os.path.join(directory[1:], file)
                ROOTFiles.append(path_file)
    
    return ROOTFiles

def plotTrackDirs(tracks):
    coord = ['x', 'y', 'z']

    fig, ax = plt.subplots(1, 3)
    ax = ax.ravel()
    
    x = 0
    perms = []
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            if i != j and [j, i] not in perms:
                ax[x].scatter(tracks[i], tracks[j], s=1)
                ax[x].set_xlabel(coord[i])
                ax[x].set_ylabel(coord[j])
                ax[x].set_title("track directions : (" + coord[i] + "," + coord[j] + ")" )
                ax[x].set_aspect("equal")
                perms.append([i, j])
                x+=1

def plotdEdxVsRange(dEdx, residualRange, cutoff=50):
    dEdx = [dEdx[i] for i in range(len(dEdx)) if dEdx[i] < cutoff]
    residualRange = [residualRange[i] for i in range(len(dEdx)) if dEdx[i] < cutoff]
    plt.hist2d(residualRange, dEdx, bins=100, cmap="binary", norm=matplotlib.colors.LogNorm())
    plt.xlabel("Reisudal range (cm)")
    plt.ylabel("dE/dx (MeV/cm)")

def PlotPIDs(pids):
    ids = np.unique(pids)
    pids = pids.tolist()
    number = [pids.count(_id) for _id in ids]
    plt.bar([str(_id) for _id in ids], number)
    plt.xlabel("Particle ID")
    plt.ylabel("Count")


fileNames = GetFileNames()
for f in fileNames:
    if(f.find("dEdx") > 0):
        print(f)
        file = uproot.open(f)

tree = file['anaMod/tree']
print(tree.keys())

dEdx = tree.arrays(library="np")['dEdx']
residualRange = tree.arrays(library="np")['ResidualRange']
pids = tree.arrays(library="np")['pdgs']

tracks = [tree.arrays(library="np")['trackDirX'], tree.arrays(library="np")['trackDirY'], tree.arrays(library="np")['trackDirZ']]

dEdx = np.hstack([l for l in dEdx])
residualRange = np.hstack([l for l in residualRange])
pids = np.hstack([l for l in pids])

tracks[0] = np.hstack([l for l in tracks[0]])
tracks[1] = np.hstack([l for l in tracks[1]])
tracks[2] = np.hstack([l for l in tracks[2]])

#plotdEdxVsRange(dEdx, residualRange, 50)
















