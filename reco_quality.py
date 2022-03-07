"""
Created on: 22/01/2022 15:41

Author: Shyam Bhuller

Description: compare recontructed MC to MC truth.
"""
import argparse
import os
import awkward as ak
import time
import numpy as np
# custom modules
import Plots
import Master
import vector

@Master.timer
def MCTruth(events : Master.Data):
    """Calculate true shower pair quantities.

    Args:
        sortEnergy (ak.array): mask to sort shower pairs by true energy

    Returns:
        tuple of ak.Array: calculated quantities
    """
    #* get the primary pi0
    mask_pi0 = np.logical_and(events.trueParticles.number == 1, events.trueParticles.pdg == 111)
    photons = events.trueParticles.truePhotonMask

    #* get pi0 -> two photons only
    #mask_daughters = ak.all(events.trueParticles.pdg != 11, axis=1)
    #mask_daughters = np.logical_and(events.trueParticles.mother == 1, mask_daughters)

    #* compute start momentum of dauhters
    p_daughter = events.trueParticles.momentum[photons]
    sum_p = ak.sum(p_daughter, axis=1)
    sum_p = vector.magntiude(sum_p)
    p_daughter_mag = vector.magntiude(p_daughter)
    p_daughter_mag = p_daughter_mag[events.SortedTrueEnergyMask]

    #* compute true opening angle
    angle = np.arccos(vector.dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

    #* compute invariant mass
    e_daughter = events.trueParticles.energy[photons]
    inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    p_pi0 = events.trueParticles.momentum[mask_pi0]
    p_pi0 = vector.magntiude(p_pi0)
    return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0

@Master.timer
def RecoQauntities(events : Master.Data):
    """Calculate reconstructed shower pair quantities.

    Args:
        events (Data): events to study

    Returns:
        tuple of ak.Array: calculated quantities + array which masks null shower pairs
    """
    #* leading + subleading energies
    # get shower pairs
    #pairs = ShowerPairsByHits(nHits)
    #global energy_pair, sortindex
    #energy_pair = ak.flatten(GetPairValues(pairs, energy), -1)
    #sortedPairs = ak.sort(energy_pair, ascending=True)
    #sortedPairs = ak.unflatten(energy_pair[sortindex], 1, 0) # sort showers by true photon energy
    #leading = sortedPairs[:, :, 1:]
    #secondary = sortedPairs[:, :, :-1]
    #! not needed if we match to true photons
    sortEnergy = events.SortedTrueEnergyMask
    sortedPairs = ak.unflatten(events.recoParticles.energy[sortEnergy], 1, 0)
    leading = sortedPairs[:, :, 1:]
    secondary = sortedPairs[:, :, :-1]

    #* opening angle
    #direction_pair = GetPairValues(pairs, r_dir)
    direction_pair = ak.unflatten(events.recoParticles.direction[sortEnergy], 1, 0)
    direction_pair_mag = vector.magntiude(direction_pair)
    angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

    #* Invariant Mass
    inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

    #* sum energy
    #total_energy = leading + secondary # assuming the daughters are photon showers

    #* pi0 momentum
    # create momentum vectors assuming these are photon showers (not neccessarily true) -> E * dir
    shower_mom = vector.prod(sortedPairs, direction_pair)
    pi0_momentum = vector.magntiude(ak.sum(shower_mom, axis=2))/1000

    null_dir = np.logical_or(direction_pair[:, :, 1:].x == -999, direction_pair[:, :, :-1].x == -999) # mask shower pairs with invalid direction vectors
    null = np.logical_or(leading < 0, secondary < 0) # mask of shower pairs with invalid energy
    
    #* filter null data
    pi0_momentum = np.where(null_dir, -999, pi0_momentum)
    pi0_momentum = np.where(null, -999, pi0_momentum)

    leading = leading/1000
    secondary = secondary/1000

    leading = np.where(null, -999, leading)
    leading = np.where(null_dir, -999, leading)
    secondary = np.where(null, -999, secondary)
    secondary = np.where(null_dir, -999, secondary)

    angle = np.where(null, -999, angle)
    angle = np.where(null_dir, -999, angle)

    inv_mass = inv_mass/1000
    inv_mass = np.where(null, -999,inv_mass)
    inv_mass = np.where(null_dir, -999,inv_mass)

    return inv_mass, angle, leading, secondary, pi0_momentum, ak.unflatten(null, 1, 0)


def Error(reco, true, null):
    """Calcuate fractional error, filter null data and format data for plotting.

    Args:
        reco (ak.array): reconstructed quantity
        true (ak.array): true quantity
        null (ak.array): mask for events without shower pairs, reco direction or energy

    Returns:
        tuple of np.array: flattened numpy array of errors, reco and truth
    """
    true = true[null]
    true = ak.where( ak.num(true, 1) > 0, true, [np.nan]*len(true) )
    reco = ak.flatten(reco, 1)[null]
    print(f"reco pairs: {len(reco)}")
    print(f"true pairs: {len(true)}")
    error = (reco / true) - 1
    return ak.to_numpy(ak.ravel(error)), ak.to_numpy(ak.ravel(reco)), ak.to_numpy(ak.ravel(true))


def PlotSingle(data, ranges, labels, bins, subDirectory, names, save):
    """Plot 1D histograms.

    Args:
        data (list[Array]): data to plot (should be flattened)
        ranges (list of length 2): plot range
        labels (list[string]): list of x labels
        bins (int/Array): number of bins or specific bins
        subDirectory (string): directory to save plots
        names (list[string]): name of plots to save
        save (bool): whether to save plots or not
    """
    if save is True: os.makedirs(subDirectory, exist_ok=True)
    for i in range(len(data)):
        if len(ranges[i]) == 2:
            d = data[i][data[i] > ranges[i][0]]
            d = d[d < ranges[i][1]]
        Plots.PlotHist(d, bins, labels[i])
        if save is True: Plots.Save(names[i], subDirectory)


def PlotReco(reco, nDaughters, labels, names):
    """Plot reconstructed quantities.

    Args:
        reco (np.array): numpy array of reconstructed quantities.
    """
    reco[reco == -999] = None
    outDirReco = outDir + "reco/"
    if save is True: os.makedirs(outDirReco, exist_ok=True)
    
    Plots.PlotBar(ak.to_numpy(nDaughters), xlabel="Number of reconstructed daughter objects per event")
    if save is True: Plots.Save("nDaughters", outDir)
    #Plots.PlotHist(total_energy/1000, bins, "Sum shower energy (GeV))")
    #if save is True: Plots.Save("sum_energy", outDir)
    for i in range(len(names)):
        Plots.PlotHist(reco[i], bins , labels[i])
        if save is True: Plots.Save(names[i], outDirReco)


def PlotTruth(truth, labels, names):
    outDirTrue = outDir + "truths/"
    truth[truth == -999] = None
    new_bins = [np.linspace(0, ak.mean(truth[0])+0.1, bins, True), bins, bins, bins, np.linspace(0.4, ak.mean(truth[4])+0.1, bins, True)]

    if save is True: os.makedirs(outDirTrue, exist_ok=True)

    for i in range(len(names)):
        Plots.PlotHist(truth[i], new_bins[i], labels[i])
        if save is True: Plots.Save(names[i], outDirTrue)


def main():
    #* load data
    events = Master.Data(file)
    nDaughters = ak.count(events.recoParticles.nHits, -1) # for plotting

    #* filter
    valid = Master.Pi0MCMask(events, -1)
    print(f"Number of showers before filtering: {ak.num(events.recoParticles.direction, 0)}")
    events.Filter([valid], [valid], returnCopy=False)
    print(f"Number of showers after filtering: {ak.num(events.recoParticles.direction, 0)}")
    showers, _, selection_mask = events.GetMCMatchingFilters()

    events.Filter([showers, selection_mask], [selection_mask], returnCopy=False)

    #* calculate quantities
    mct = MCTruth(events)
    rmc = RecoQauntities(events)

    # keep track of events with no shower pairs
    null = ak.flatten(rmc[-1], -1)
    null = ak.num(null, 1) > 0

    #* compute errors
    error = []
    reco = []
    true = []
    for i in range(len(names)):
        print(names[i])
        e, r, t = Error(rmc[i], mct[i], null)
        error.append(e)
        reco.append(r)
        true.append(t)

    error = np.nan_to_num(error, nan=-999)
    reco = np.nan_to_num(reco, nan=-999)
    true = np.nan_to_num(true, nan=-999)

    t_r = [[-998, max(true[0])], [-998, max(true[1])], [-998, max(true[2])], [-998, max(true[3])], [-998, max(true[4])]]
    #e_r = [[-998, max(error[0])], [-998, max(error[1])], [-998, max(error[2])], [-998, max(error[3])], [-998, max(error[4])]]
    e_r = [[-1, 1]]*5 # plot bounds for fractional errors

    #* make plots
    if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
    for j in range(len(error)):
        for i in range(len(true)):
            Plots.PlotHist2D(true[i], error[j], bins, x_range=t_r[i], y_range=e_r[j], xlabel=t_l[i], ylabel=e_l[j])
            if save is True: Plots.Save( names[j]+"_"+names[i] , outDir + "2D/")

    #PlotSingle(error, e_r, e_l, bins, outDir + "fractional_errors/", names, save)
    PlotReco(reco, nDaughters, r_l, names)
    PlotTruth(true, t_l, names)


if __name__ == "__main__":
    names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
    t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True secondary shower energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
    e_l = ["Invariant mass fractional error (GeV)", "Opening angle fractional error (rad)", "Leading shower energy fractional error (GeV)", "Secondary shower energy fractional error (GeV)", "$\pi^{0}$ momentum fractional error (GeV)"]
    r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]

    parser = argparse.ArgumentParser(description="Study em shower merging for pi0 decays.")
    parser.add_argument("-f", "--file", dest="file", type=str, default="ROOTFiles/pi0_0p5GeV_100K_5_7_21.root", help="ROOT file to open.")
    parser.add_argument("-b", "--nbins", dest="bins", type=int, default=50, help="number of bins when plotting histograms.")
    parser.add_argument("-s", "--save", dest="save", type=bool, default=False, help="whether to save the plots.")
    parser.add_argument("-d", "--directory", dest="outDir", type=str, default="pi0_0p5GeV_100K/match_MC/", help="directory to save plots.")
    #args = parser.parse_args("") #! to run in Jutpyter notebook
    args = parser.parse_args() #! run in command line

    file = args.file
    bins = args.bins
    save = args.save
    outDir = args.outDir
    main()