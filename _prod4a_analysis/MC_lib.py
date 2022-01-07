#import sys
#sys.path.append('../_base_libs/') # need this to import custom scripts in different directories
from Master import ITEM, QUANTITY, Vector3, Unwrap, SelectionMask, Conditional, Data, UpdateShowerPair, CutDict
from SelectionQuantities import Angle, GetShowerPairValues, CalculateQuantities
from Analyser import CalculateDataFrameNew, SortPairsByEnergy
from Plots import PlotHist, PlotHistComparison, Save, PlotHist2D, PlotBar, PlotBarComparision

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plotLabels = {
        ITEM.PANDORA_TAG                   : "Pandora tag",
        QUANTITY.CNN_SCORE                : "CNN score",
        ITEM.HITS                          : "Number of hits",
        ITEM.SHOWER_LENGTH                 : "Shower length (cm)",
        ITEM.SHOWER_ANGLE                  : "Shower angle (rad)",
        QUANTITY.BEAM_ANGLE               : "Beam-shower angle (rad)",
        QUANTITY.MC_ANGLE                 : "True/reco shower angle (rad)",
        QUANTITY.START_HITS               : "Start hits",
        QUANTITY.SHOWER_PAIRS             : "Shower pairs index",
        QUANTITY.SHOWER_PAIR_PANDORA_TAGS : "average pandora tag",
        QUANTITY.SHOWER_SEPERATION        : "Separation (cm)",
        QUANTITY.OPENING_ANGLE            : "Opening angle (rad)",
        QUANTITY.DIST_VERTEX              : "Distance to decay vertex (cm)",
        QUANTITY.DECAY_VERTEX             : "Decay vertex position (cm)",
        QUANTITY.LEADING_ENERGY           : "Leading shower energy (GeV)",
        QUANTITY.SECONDARY_ENERGY         : "Secondary shower energy (GeV)",
        QUANTITY.INVARIANT_MASS           : "Invariant mass (GeV)",
        QUANTITY.TRUE_OPENING_ANGLE       : "True opening angle (rad)",
        QUANTITY.TRUE_INVARIANT_MASS      : "True Invariant mass (GeV)"
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
    for i in range(len(quantities[QUANTITY.SHOWER_PAIRS])):
        pairs = quantities[QUANTITY.SHOWER_PAIRS][i]
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


def Filter(quantities : dict, signal : list, shower_pairs, update_shower_pairs : bool = True, _print : bool = False):
    """
    Filters signal from background based on mc truth signal.
    """
    # create blank signal/background dictionaries
    pi0 = CalculateQuantities(empty=True)
    background = CalculateQuantities(empty=True)

    # sort data into catergories which need to be filtered differently due to there data structure
    beamData = [ITEM.BEAM_END_POS, ITEM.BEAM_START_POS, ITEM.EVENT_ID, ITEM.RUN]
    unpaired = [QUANTITY.BEAM_ANGLE, QUANTITY.MC_ANGLE, QUANTITY.CNN_SCORE, QUANTITY.START_HITS]
    unpaired.extend( list(set([x for x in ITEM]) - set(beamData)) ) # add everything but beam data fomr ITEM
    vectors = [ITEM.BEAM_END_POS, ITEM.BEAM_START_POS, ITEM.DIRECTION, ITEM.START_POS, ITEM.TRUE_END_POS, ITEM.TRUE_MOMENTUM, ITEM.TRUE_START_POS, QUANTITY.DECAY_VERTEX]

    def FilterShowers(target, showers, pair, pair_index, added_showers):
        if quantity in unpaired:
            for i in range(2):
                # shower pairs could contain the same showers, so keep track to avoid duplicate data
                if pair[i] not in added_showers:

                        target.append(showers[pair[i]])
                        added_showers.append(pair[i])
        else:
            # don't need to keep track of shower pairs as they should all be unique
            if quantity is QUANTITY.SHOWER_PAIRS and update_shower_pairs is True:
                target.append( UpdateShowerPair(pair, pair_index, added_showers) )
            else:
                target.append( showers[pair_index] )

    # loop through each quantity
    for quantity in quantities:
        if _print is True : print(quantity)
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
            if item is QUANTITY.SHOWER_PAIRS: continue
            print(item)

            s = Unwrap(data_1[item])
            if data_2 is not None: b = Unwrap(data_2[item])
            
            if item is ITEM.HITS:
                s = s[s < 200]
                if data_2 is not None: b = b[b < 200]
            if item in [QUANTITY.DIST_VERTEX, QUANTITY.SHOWER_SEPERATION, ITEM.SHOWER_LENGTH]:
                s = s[s < 100]
                if data_2 is not None: b = b[b < 100]
            if item in [QUANTITY.LEADING_ENERGY, QUANTITY.SECONDARY_ENERGY]:
                s = s[s > 0]
                s = s[s < 0.5]
                if data_2 is not None:
                    b = b[b > 0]
                    b = b[b < 0.5]
            if item in [QUANTITY.INVARIANT_MASS, QUANTITY.TRUE_INVARIANT_MASS]:
                s = s[s < 0.5]
                if data_2 is not None: b = b[b < 0.5]

            s = s[s != -999]
            if data_2 is not None: b = b[b != -999]

            if data_2 is not None:
                if item in [ITEM.PANDORA_TAG, QUANTITY.SHOWER_PAIR_PANDORA_TAGS]:
                    PlotBarComparision(s, b, width=0.4, xlabel=plotLabels[item], label_1=label_1, label_2=label_2)
                else:
                    PlotHistComparison(s, b, bins=50, xlabel=plotLabels[item], label_1=label_1, label_2=label_2, alpha=0.5)
            else:
                if item in [ITEM.PANDORA_TAG, QUANTITY.SHOWER_PAIR_PANDORA_TAGS]:
                    print("bar plot")
                    PlotBar(s, width=0.4, xlabel=plotLabels[item])
                else:
                    PlotHist(s, bins=50, xlabel=plotLabels[item])
            if item is QUANTITY.INVARIANT_MASS:
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
    return q - true_q


def PlotResiduals(data_list: dict, quantities: dict, save: bool = False, subDirectory: str = "out/"):

    energy_resiudal = CalculateResidual(
        data_list[ITEM.ENERGY], data_list[ITEM.TRUE_ENERGY])
    opening_angle_residual = CalculateResidual(
        quantities[QUANTITY.OPENING_ANGLE], quantities[QUANTITY.TRUE_OPENING_ANGLE])
    invariant_mass_residual = CalculateResidual(
        quantities[QUANTITY.INVARIANT_MASS], quantities[QUANTITY.TRUE_INVARIANT_MASS])

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


def PlotCorrelations(data_list: dict, quantities: dict, target = QUANTITY.INVARIANT_MASS, outDir : str = "", outName: str = "correlations.png", x_range=[0, 0.5]):

    single_quantities_tag = [ITEM.PANDORA_TAG, QUANTITY.CNN_SCORE, ITEM.HITS, ITEM.SHOWER_LENGTH, ITEM.SHOWER_ANGLE, QUANTITY.BEAM_ANGLE, QUANTITY.MC_ANGLE, QUANTITY.START_HITS]
    single_quantities = [quantities[q] for q in single_quantities_tag]
    single_quantities.append(data_list[ITEM.ENERGY])

    single_quantities = [GetShowerPairValues(q, quantities[QUANTITY.SHOWER_PAIRS]) for q in single_quantities]
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
        if item is QUANTITY.SHOWER_PAIRS:
            continue

        x = Unwrap(quantities[target])
        y_range = []

        ### CUSTOM DICTIONARY VALUE
        if item == "RESIDUAL":
            y_range = [-1, 1]

        if item is ITEM.PANDORA_TAG:
            y_range = [11, 14]

        if item is QUANTITY.BEAM_ANGLE:
            y_range = [0, 7]

        if item is QUANTITY.SHOWER_SEPERATION:
            y_range = [0, 300]
        
        if item is QUANTITY.DIST_VERTEX:
            y_range = [0, 200]

        if item is QUANTITY.TRUE_INVARIANT_MASS:
            y_range = [0, 1]

        if item is ITEM.HITS:
            y_range = [0, 751]
        
        if item is ITEM.SHOWER_LENGTH:
            y_range = [0, 600]

        if item is ITEM.SHOWER_ANGLE:
            y_range = [0, 0.2]

        if item in single_quantities_tag:
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, quantities[item][0], bins, x_range, y_range, None, "leading "+plotLabels[item], newFigure=False)
            i+=1
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, quantities[item][1], bins, x_range, y_range, None, "secondary "+plotLabels[item], newFigure=False)
            i+=1
            continue

        plt.subplot(4, 8, i+1)
        if item in [QUANTITY.DECAY_VERTEX]:
            y_range = [-1000, 1000]
            PlotHist2D(x, Unwrap(quantities[item].x), bins, x_range, y_range, None, str(plotLabels[item] + " x"), newFigure=False)
            i+=1
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, Unwrap(quantities[item].y), bins, x_range, y_range, None, str(plotLabels[item] + " y"), newFigure=False)
            i+=1
            plt.subplot(4, 8, i+1)
            PlotHist2D(x, Unwrap(quantities[item].z), bins, x_range, y_range, None, str(plotLabels[item] + " z"), newFigure=False)
            i+=1
            continue
        else:
            PlotHist2D(x, Unwrap(quantities[item]), bins, x_range, y_range, None, plotLabels[item], newFigure=False)
            i+=1
            continue
    fig.text(0.5, 0.005, plotLabels[target], ha='center')
    plt.savefig(outDir + outName, dpi=500)
    plt.close()


def PlotShowerPairCorrelations(data : dict, quantities : dict, save: bool = False, subDirectory: str = "out/"):
    """
    Plot the leading and secondary shower properties against eachother.
    """
    single_quantities_tag = [ITEM.PANDORA_TAG, QUANTITY.CNN_SCORE, ITEM.HITS, ITEM.SHOWER_LENGTH, ITEM.SHOWER_ANGLE,
                             QUANTITY.BEAM_ANGLE, QUANTITY.MC_ANGLE, QUANTITY.START_HITS]
    single_quantities = [quantities[q] for q in single_quantities_tag]
    single_quantities.append(data[ITEM.ENERGY])

    single_quantities = [GetShowerPairValues(q, quantities[QUANTITY.SHOWER_PAIRS]) for q in single_quantities]
    single_quantities = [Unwrap(q) for q in single_quantities]
    single_quantities = SortPairsByEnergy(single_quantities, index=8)

    _dict_sorted = {}
    single_quantities_tag.append(ITEM.ENERGY)
    plotLabels.update({ITEM.ENERGY: "shower energy (GeV)"})
    for i in range(len(single_quantities_tag)):
        _dict_sorted.update({single_quantities_tag[i]: single_quantities[i]})

    for item in _dict_sorted:
        bins = 100
        l = _dict_sorted[item][0] # leading shower
        s = _dict_sorted[item][1] # secondary shower

        x_range = [-998, max(l)]
        y_range = [-998, max(s)]

        if item is ITEM.PANDORA_TAG:
            bins = 2
        if item is ITEM.ENERGY:
            x_range = [0, 1]
            y_range = [0, 1]
        if item is QUANTITY.START_HITS:
            bins = 20
        if item is ITEM.HITS:
            x_range = [0, 600]
            y_range = [0, 600]

        PlotHist2D(l, s, bins, x_range, y_range, "leading " + plotLabels[item], "secondary " + plotLabels[item])
        if save is True:
            Save(item.name + "_correlation", subDirectory)


def AdvancedCNNScoreMask(cnn_score, shower_pairs, energy) -> SelectionMask:
    """
    Create more advanced selections on the CNN score.
    """
    mask = SelectionMask()
    mask.InitiliseMask(cnn_score) # create mask based on the shape of data
    for i in range(len(shower_pairs)):
        for j in range(len(shower_pairs[i])):
            # get leading and secondary shower energy
            if energy[i][shower_pairs[i][j][0]] > energy[i][shower_pairs[i][j][1]]:
                le_index = shower_pairs[i][j][0]
                se_index = shower_pairs[i][j][1]
            else:
                le_index = shower_pairs[i][j][1]
                se_index = shower_pairs[i][j][0]
            # cut on CNN score
            if cnn_score[i][le_index] < 0:
                mask.mask[i][le_index] = 0
            if cnn_score[i][se_index] < 0:
                mask.mask[i][se_index] = 0

            if cnn_score[i][le_index] < 0.4:
                mask.mask[i][le_index] = 0
            if cnn_score[i][se_index] < 0.4:
                mask.mask[i][se_index] = 0

            if 1.25 - cnn_score[i][le_index] > cnn_score[i][se_index]:
                mask.mask[i][le_index] = 0
                mask.mask[i][se_index] = 0
    return mask


def CutEfficiency(_all, signal, background, selection=None, mask=None, outDir=None, allPairs=True):
    #nEvt, nShower, nPair = Stats(*_all)
    signal_nEvt, signal_nShower, signal_nPair = Stats(*signal)
    background_nEvt, background_nShower, background_nPair = Stats(*background)

    if mask == None and selection != None:
        new_data = CutDict(_all[0], *selection, None)
        new_quantities = CalculateQuantities(new_data, allPairs, *selection)
    elif mask != None and selection == None:
        new_data = CutDict(_all[0], custom_mask=mask)
        new_quantities = CalculateQuantities(new_data, allPairs=allPairs)
    else:
        print("no selection specified")
        return

    signal_id = FindPi0Signal(new_quantities, new_data)
    new_quantities_f = Filter(new_quantities.copy(), signal_id, new_quantities[QUANTITY.SHOWER_PAIRS])
    new_data_f = Filter(new_data.copy(), signal_id, new_quantities[QUANTITY.SHOWER_PAIRS])

    #new_nEvt, new_nShower, new_nPair = Stats(new_data, new_quantities)
    new_signal_nEvt, new_signal_nShower, new_signal_nPair = Stats(new_data_f[0], new_quantities_f[0])
    new_background_nEvt, new_background_nShower, new_background_nPair = Stats(new_data_f[1], new_quantities_f[1])

    #diff_nEvt = abs(new_nEvt - nEvt)
    #diff_nShower = abs(new_nShower - nShower)
    #diff_nPair = abs(new_nPair - nPair)
    
    #diff_signal_nEvt = abs(new_signal_nEvt - signal_nEvt)
    #diff_signal_nShower = abs(new_signal_nShower - signal_nShower)
    diff_signal_nPair = abs(new_signal_nPair - signal_nPair)
    
    #diff_background_nEvt = abs(new_background_nEvt - background_nEvt)
    #diff_background_nShower = abs(new_background_nShower - background_nShower)
    diff_background_nPair = abs(new_background_nPair - background_nPair)

    signalPercentageLost = (diff_signal_nPair / signal_nPair)
    backgroundPercentageLost = (diff_background_nPair / background_nPair)
    sbr_before = (signal_nPair/background_nPair)
    sbr_after = (new_signal_nPair/new_background_nPair)

    out = ["percentage of signal shower pairs lost: %.3f" % signalPercentageLost,
           "percentage of background shower pairs lost: %.3f" % backgroundPercentageLost,
           "shower pair signal background ratio before: %.3f" % sbr_before,
           "shower pair signal background ratio after: %.3f" % sbr_after]

    if outDir != None:
        with open(outDir + "cut_performance.txt", "w") as text_file:
            for line in out:
                print(line)
                text_file.write(line + "\n")
    else:
        for line in out:
            print(line)

    return signalPercentageLost, backgroundPercentageLost


def Stats(data, quantities):
    """
    return general Stats of the data.
    """
    number_of_events = len(Unwrap(data[ITEM.EVENT_ID]))
    number_of_showers = len(Unwrap(data[ITEM.HITS]))
    number_of_shower_pairs = len(Unwrap(quantities[QUANTITY.SHOWER_SEPERATION]))
    return number_of_events, number_of_showers, number_of_shower_pairs 