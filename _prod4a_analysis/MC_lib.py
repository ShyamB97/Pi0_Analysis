from base_libs.Master import ITEM, QUANTITIY, Vector3, Unwrap, SelectionMask, Conditional, Data
from base_libs.SelectionQuantities import Angle, GetShowerPairValues
from Analyser import CalculateDataFrameNew, SortPairsByEnergy
from base_libs.Plots import PlotHist, PlotHistComparison, Save, PlotHist2D

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plotLabels = {
        ITEM.PANDORA_TAG                : "Pandora tag",
        QUANTITIY.CNN_SCORE             : "CNN score",
        ITEM.HITS                       : "Number of hits",
        ITEM.SHOWER_LENGTH              : "Shower length (cm)",
        ITEM.SHOWER_ANGLE               : "Shower angle (rad)",
        QUANTITIY.BEAM_ANGLE            : "Beam-shower angle (rad)",
        QUANTITIY.MC_ANGLE              : "True/reco shower angle (rad)",
        QUANTITIY.START_HITS            : "Start hits",
        QUANTITIY.SHOWER_PAIRS          : "Shower pairs index",
        QUANTITIY.SHOWER_SEPERATION     : "Separation (cm)",
        QUANTITIY.OPENING_ANGLE         : "Opening angle (rad)",
        QUANTITIY.DIST_VERTEX           : "Distance to decay vertex (cm)",
        QUANTITIY.DECAY_VERTEX          : "Decay vertex position (cm)",
        QUANTITIY.LEADING_ENERGY        : "Leading shower energy (GeV)",
        QUANTITIY.SECONDARY_ENERGY      : "Secondary shower energy (GeV)",
        QUANTITIY.INVARIANT_MASS        : "Invariant mass (GeV)",
        QUANTITIY.TRUE_OPENING_ANGLE    : "True opening angle (rad)",
        QUANTITIY.TRUE_INVARIANT_MASS   : "True Invariant mass (GeV)"
}


def Pi0Decay(pair, pos, pdg, mom, debug=False) -> bool:
    """
    Algorithm which determines if a shower pair are the product of a Pi0 decay (exlcuding pi0 -> gamma e+ e-).
    """
    pos_1 : Vector3 = pos[pair[0]] # get shower start positions
    pos_2 : Vector3 = pos[pair[1]]
    pdg_1 : int     = pdg[pair[0]] # get shower pdgs
    pdg_2 : int     = pdg[pair[1]]
    mom_1 : Vector3 = mom[pair[0]] # shower momenta
    mom_2 : Vector3 = mom[pair[1]]
    pi0 = False
    # First check shower pairs come from the same parent particle
    if(pos_1 == pos_2):
        if debug is True: print("particles have the same start pos")
        # Next check both particles are photons
        if pdg_1 == 22 and pdg_2 == 22:
            if debug is True: print("both are photon showers")
            angle = Angle(mom_1, mom_2)
            inv_mass = (2 * mom_1.Magnitude() * mom_2.Magnitude() * (1 - np.cos(angle)) )**0.5
            if debug is True: print(inv_mass)
            if debug is True: print((inv_mass * 1000) - 134.9768)
            # Finally check invariant mass is acceptably close to the pi0 mass.
            if( abs((inv_mass * 1000) - 134.9768) < 0.0005 ):
                if debug is True: print("pi0 event")
                pi0 = True
    return pi0


def FindPi0Signal(quantities, truths):
    """
    Determine whether an event conatins a Pi0 deacy
    """
    signal = []
    for i in range(len(quantities[QUANTITIY.SHOWER_PAIRS])):
        pairs = quantities[QUANTITIY.SHOWER_PAIRS][i]
        daughter_pos = truths[ITEM.TRUE_START_POS][i]
        daughter_pdg = truths[ITEM.TRUE_PDG][i]
        daughter_mom = truths[ITEM.TRUE_MOMENTUM][i]
        #print(pair)
        tmp = []
        for j in range(len(pairs)):
            pi0 = Pi0Decay(pairs[j], daughter_pos, daughter_pdg, daughter_mom, False) # currently have one pair per event
            tmp.append(pi0)
        signal.append(tmp)
    return signal


def BlankQuantities():
    """
    Creates blank quantity dictionary.
    """
    return {
        ITEM.PANDORA_TAG            : None,
        QUANTITIY.CNN_SCORE         : None,
        ITEM.HITS                   : None,
        ITEM.SHOWER_LENGTH          : None,
        ITEM.SHOWER_ANGLE           : None,
        QUANTITIY.BEAM_ANGLE        : None,
        QUANTITIY.MC_ANGLE          : None,
        QUANTITIY.START_HITS        : None,
        QUANTITIY.SHOWER_PAIRS      : None,
        QUANTITIY.SHOWER_SEPERATION : None,
        QUANTITIY.OPENING_ANGLE     : None,
        QUANTITIY.DIST_VERTEX       : None,
        QUANTITIY.DECAY_VERTEX      : None,
        QUANTITIY.LEADING_ENERGY    : None,
        QUANTITIY.SECONDARY_ENERGY  : None,
        QUANTITIY.INVARIANT_MASS    : None
    }


def Filter(quantities : dict, signal : list, shower_pairs, update_shower_pairs : bool = True):
    """
    Filters signal from background based on mc truth signal.
    """
    # create blank signal/background dictionaries
    pi0 = BlankQuantities()
    background = BlankQuantities()

    # sort data into catergories which need to be filtered differently due to there data structure
    beamData = [ITEM.BEAM_END_POS, ITEM.BEAM_START_POS, ITEM.EVENT_ID, ITEM.RUN]
    unpaired = [QUANTITIY.BEAM_ANGLE, QUANTITIY.MC_ANGLE, QUANTITIY.CNN_SCORE, QUANTITIY.START_HITS]
    unpaired.extend( list(set([x for x in ITEM]) - set(beamData)) ) # add everything but beam data fomr ITEM
    vectors = [ITEM.BEAM_END_POS, ITEM.BEAM_START_POS, ITEM.DIRECTION, ITEM.START_POS, ITEM.TRUE_END_POS, ITEM.TRUE_MOMENTUM, ITEM.TRUE_START_POS, QUANTITIY.DECAY_VERTEX]

    def FilterShowers(target, showers, pair, pair_index, added_showers):
        if quantity in unpaired:
            for i in range(2):
                # shower pairs could contain the same showers, so keep track to avoid duplicate data
                if pair[i] not in added_showers:

                        target.append(showers[pair[i]])
                        added_showers.append(pair[i])
        else:
            # don't need to keep track of shower pairs as they should all be unique
            if quantity is QUANTITIY.SHOWER_PAIRS and update_shower_pairs is True:
                # shower pairs are the index of the shower in data
                # so to match the data struture of signal/backgound, we need to redefine them
                # the new indices of the showers will be their index in target i.e.
                # the length of target itself at the time it is added (example below)
                # showers = [0, 1, 2, 3] -> signals are 0 and 2
                # shower pair = [0, 2]
                # now filter so showers = [0, 1, 2, 3] -> [0, 2]
                # update shower pair so [0, 2] -> [0, 1] because shower as index 2 in the old dataset is now at index 1
                if pair_index == 0:
                    pair_new = [0, 1]
                    added_showers.extend(pair)
                else:
                    pair_new = [-999, -999]
                    for k in range(2):
                        if pair[k] in added_showers:
                            pair_new[k] = added_showers.index(pair[k])
                        else:
                            pair_new[k] = len(added_showers)
                            added_showers.append(pair[k])
                target.append( pair_new )
            else:
                target.append( showers[pair_index] )

    # loop through each quantity
    for quantity in quantities:
        print(quantity)
        new_s = []
        new_b = []
        # loop through each event
        for e in range(len(signal)):
            evt_s = []
            evt_b = []
            evt_data = quantities[quantity][e] # data in event, single value (beam) or nested list
            if quantity in beamData:
                # beamData is not dobule nested data, so just look at evt_data.
                """ Does this even do anything? perhaps just assign quantities[quantity] to new_s and new_b and skip looping into elements """
                if True in signal[e]:
                    evt_s = evt_data
                if False in signal[e]:
                    evt_b = evt_data
            else:
                # loop through each shower pair in the event
                added_showers_s = []
                added_showers_b = []
                for j in range(len(signal[e])):
                    # if it is a signal append shower pair quantitiy to signal, same for background
                    pair = shower_pairs[e][j]
                    if signal[e][j] is True:
                        FilterShowers(evt_s, evt_data, pair, j, added_showers_s)
                    if signal[e][j] is False:
                        FilterShowers(evt_b, evt_data, pair, j, added_showers_b)
                evt_s = np.array(evt_s, object) # list -> ndarray to do math operations
                evt_b = np.array(evt_b, object)
            new_s.append(evt_s)
            new_b.append(evt_b)
        # switch list of vectors to vector of lists
        if quantity in vectors:
            new_s = Vector3.ListToVector3(new_s)
            new_b = Vector3.ListToVector3(new_b)
        else:
            new_s = np.array(new_s, object)
            new_b = np.array(new_b, object)
        pi0[quantity] = new_s
        background[quantity] = new_b
    return pi0, background


def PlotQuantities(data_1, data_2=None, save : bool = True, subDirectory : str = "out/", label_1 ="signal", label_2 = "background"):
    """
    Plot calculated quantities.
    """
    os.makedirs(subDirectory, exist_ok=True)
    for item in data_1:
        if type(data_1[item]) is not Vector3:
            if item is QUANTITIY.SHOWER_PAIRS: continue
            print(item)
            s = Unwrap(data_1[item])
            if data_2 is not None: b = Unwrap(data_2[item])
            
            if item is ITEM.HITS:
                s = s[s < 200]
                if data_2 is not None: b = b[b < 200]
            if item in [QUANTITIY.DIST_VERTEX, QUANTITIY.SHOWER_SEPERATION, ITEM.SHOWER_LENGTH]:
                s = s[s < 100]
                if data_2 is not None: b = b[b < 100]
            if item in [QUANTITIY.LEADING_ENERGY, QUANTITIY.SECONDARY_ENERGY]:
                s = s[s > 0]
                s = s[s < 0.5]
                if data_2 is not None:
                    b = b[b > 0]
                    b = b[b < 0.5]
            if item in [QUANTITIY.INVARIANT_MASS, QUANTITIY.TRUE_INVARIANT_MASS]:
                s = s[s < 0.5]
                if data_2 is not None: b = b[b < 0.5]

            if data_2 is not None:
                PlotHistComparison(s[s != -999], b[b != -999], bins=50, xlabel=plotLabels[item], label_1=label_1, label_2=label_2, alpha=0.5)
            else:
                PlotHist(s[s != -999], bins=50, xlabel=plotLabels[item])
            if item is QUANTITIY.INVARIANT_MASS:
                plt.axvline(0.13497, ymax=30, label="$m_{\pi_{0}}$", color="black")
            if save is True:
                Save(item.name, subDirectory)
    if save is False:
        plt.show()


def CalculateResidual(q, true_q, _min=0, _max=5):
    q = Unwrap(q)
    true_q = Unwrap(true_q)

    true_q = true_q[q > 0]
    q = q[q > 0]
    q = q[true_q > 0]
    true_q = true_q[true_q > 0]

    true_q = true_q[q < 5]
    q = q[q < 5]
    q = q[true_q < 5]
    true_q = true_q[true_q < 5]

    print([min(q), max(q)])
    print([min(true_q), max(true_q)])

    return q - true_q


def PlotResiduals(data_list: dict, quantities: dict, save: bool = False, subDirectory: str = "out/"):

    energy_resiudal = CalculateResidual(
        data_list[ITEM.ENERGY], data_list[ITEM.TRUE_ENERGY])
    opening_angle_residual = CalculateResidual(
        quantities[QUANTITIY.OPENING_ANGLE], quantities[QUANTITIY.TRUE_OPENING_ANGLE])
    invariant_mass_residual = CalculateResidual(
        quantities[QUANTITIY.INVARIANT_MASS], quantities[QUANTITIY.TRUE_INVARIANT_MASS])

    residuals = {
        "energy residual (GeV)": energy_resiudal,
        "opening angle residual (rad)": opening_angle_residual,
        "Invariant mass residual (GeV)": invariant_mass_residual
    }

    for r in residuals:
        PlotHist(residuals[r], bins=50, xlabel=r)
        print(r[:-6])
        if save is True:
            Save(r[:-6], subDirectory)


def PlotCorrelations(data_list: dict, quantities: dict, target = QUANTITIY.INVARIANT_MASS, outDir : str = "", outName: str = "correlations.png", x_range=[0, 0.5]):

    single_quantities_tag = [ITEM.PANDORA_TAG, QUANTITIY.CNN_SCORE, ITEM.HITS, ITEM.SHOWER_LENGTH, ITEM.SHOWER_ANGLE, QUANTITIY.BEAM_ANGLE, QUANTITIY.MC_ANGLE, QUANTITIY.START_HITS]
    single_quantities = [quantities[q] for q in single_quantities_tag]
    single_quantities.append(data_list[ITEM.ENERGY])

    single_quantities = [GetShowerPairValues(q, quantities[QUANTITIY.SHOWER_PAIRS]) for q in single_quantities]
    single_quantities = [Unwrap(q) for q in single_quantities]
    single_quantities = SortPairsByEnergy(single_quantities, index=8)
    single_quantities = single_quantities[:-1]

    for i in range(len(single_quantities_tag)):
        item = single_quantities_tag[i]
        quantities.update( {item : single_quantities[i]} )

    i = 0
    bins=50
    fig = plt.figure(figsize=(32, 12))
    for item in quantities:
        print(item)
        if item is QUANTITIY.SHOWER_PAIRS:
            continue

        x = Unwrap(quantities[target])
        y_range = []

        ### CUSTOM DICTIONARY VALUE
        if item == "RESIDUAL":
            y_range = [-1, 1]

        if item is ITEM.PANDORA_TAG:
            y_range = [11, 14]

        if item is QUANTITIY.BEAM_ANGLE:
            y_range = [0, 7]

        if item is QUANTITIY.SHOWER_SEPERATION:
            y_range = [0, 300]
        
        if item is QUANTITIY.DIST_VERTEX:
            y_range = [0, 200]

        if item is QUANTITIY.TRUE_INVARIANT_MASS:
            y_range = [0, 1]

        if item is ITEM.HITS:
            y_range = [0, 751]
        
        if item is ITEM.SHOWER_LENGTH:
            y_range = [0, 600]

        if item is ITEM.SHOWER_ANGLE:
            y_range = [0, 0.2]

        if item in single_quantities_tag:
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, quantities[item][0], bins, x_range, y_range, None, "leading "+plotLabels[item])
            i+=1
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, quantities[item][1], bins, x_range, y_range, None, "secondary "+plotLabels[item])
            i+=1
            continue

        plt.subplot(4, 8, i+1)
        if item in [QUANTITIY.DECAY_VERTEX]:
            y_range = [-1000, 1000]
            PlotHist2D(x, Unwrap(quantities[item].x), bins, x_range, y_range, None, str(plotLabels[item] + " x"))
            i+=1
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, Unwrap(quantities[item].y), bins, x_range, y_range, None, str(plotLabels[item] + " y"))
            i+=1
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, Unwrap(quantities[item].z), bins, x_range, y_range, None, str(plotLabels[item] + " z"))
            i+=1
            continue
        else:
            PlotHist2D(x, Unwrap(quantities[item]), bins, x_range, y_range, None, plotLabels[item])
            i+=1
            continue
    fig.text(0.5, 0.005, plotLabels[target], ha='center')
    plt.savefig(outDir + outName, dpi=500)
    plt.close()