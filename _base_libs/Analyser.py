#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 17:11:09 2021

@author: sb16165
"""

import numpy as np
import pandas as pd

# custom imports
import Master
import SelectionQuantities


def CalculateParameters(data, param=None, conditional=None, cut=None, cutData=False, beamData=False, cylinder=[3, -1, 4], pair_hits=True):
    """
    will retrieve data in the root file created by the pi0 analyser.
    Then the data is used to calcuate the selection quantities and if
    a conditional is given, will apply a cut on the whole dataset.
    ----- Parameters -----
    data        : object of type Master.Data(), contains functions 
                  capable of retrieving data from the root file
    ----------------------
    """    
    print("getting daughter info...")
    start_pos = data.start_pos()
    direction = data.direction()
    energy = data.energy()
    mc_energy = data.mc_energy()
    nHits = data.nHits()
    cnn_em = data.cnn_em()
    cnn_track = data.cnn_track()
    pandoraTag = data.pandoraTag()
    
    if beamData is True:
        print("getting beam info...")
        beam_start_pos = data.beam_start_pos()
        beam_end_pos = data.beam_end_pos()
    
    print("getting daughter truth info...")
    true_start_pos = data.true_start_pos()
    true_end_pos = data.true_end_pos()
    true_momentum = data.true_momentum()
    
    # custom data
    hit_radial = data.hit_radial()
    hit_longitudinal = data.hit_longitudinal()
    
    
    # calculate quantities from selected data
    print("calculate CNN score")
    cnn_score = SelectionQuantities.CNNScore(cnn_em, cnn_track)
    
    print("energy residual")
    energyResidual = (energy - mc_energy)/ mc_energy
    
    if beamData is True:
        print("angle between beam and daughters")
        beam_angle = SelectionQuantities.BeamTrackShowerAngle(beam_start_pos, beam_end_pos, direction)
    else:
        beam_angle = []
        for i in range(len(nHits)):
            evt = []
            for j in range(len(nHits[i])):
                evt.append(-999)
            beam_angle.append(evt)
        beam_angle = np.array(beam_angle, object)
    
    print("angle between daughters and mc particle")
    mc_angle = SelectionQuantities.DaughterRecoMCAngle(true_start_pos, true_end_pos, direction)
    
    print("start hits")
    start_hits = SelectionQuantities.GetShowerStartHits(hit_radial, hit_longitudinal, cylinder)
    
    ### apply cuts on data that is unpaired ###
    parameters = [pandoraTag, cnn_score, nHits, energyResidual, beam_angle, mc_angle, start_hits, energy, mc_energy]
    
    if cutData is True:
        mask = Master.SelectionMask() # create mask object
        mask.InitiliseMask(nHits) # create mask array based on a template structure
        
        for i in range(len(param)):
            if param[i] < len(parameters):
                mask.CutMask(parameters[param[i]], cut[i], conditional[i]) # make a cut on the mask based on the quantity we want to cut
        
        parameters = [mask.Apply(p) for p in parameters] # apply the cut mask to each parameter

    
        ### cut data needed to calculate paired values ###
        start_pos = mask.Apply(start_pos)
        direction = mask.Apply(direction)
        energy = mask.Apply(energy)
    
    print("shower pairs")
    if pair_hits == False:
        shower_pairs = SelectionQuantities.GetShowerPairs(start_pos)
    if pair_hits == True:
        shower_pairs = SelectionQuantities.GetShowerPairs_Hits(parameters[2])
    
    print("separation")
    pair_separation = SelectionQuantities.ShowerPairSeparation(start_pos, shower_pairs)
    pair_separation = np.reshape(pair_separation, (len(nHits)))
    
    print("pair angle")
    pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, direction)
    
    print("true pair angle")
    true_dir = true_momentum.Normalise()
    true_pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, true_dir)
    
    print("pair energy")
    pair_energies, pair_leading, pair_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, energy)
    
    print("Invariant mass")
    inv_mass = SelectionQuantities.InvariantMass(pair_angle, pair_energies)
    
    ### apply cuts on data that is paired ###
    parameters_pair = [pair_separation, pair_angle, pair_leading, pair_second, inv_mass, true_pair_angle]

    if cutData is True:
        paired_mask = Master.SelectionMask()
        paired_mask.InitiliseMask(pair_separation) # choose some arbitrary data to get its shape (can't use shower_pairs)

        for i in range(len(param)):
            # check the parameter to cut is a pair quantitiy
            if param[i] > len(parameters) - 1:
                # check the parameter number is in range
                if param[i] < (len(parameters) + len(parameters_pair)):
                    _iter = param[i] - len(parameters) # convert from dictionary iteration to parameter pair iteration
                    paired_mask.CutMask(parameters_pair[_iter], cut[i], conditional[i]) 
        
        parameters_pair = [paired_mask.Apply(p) for p in parameters_pair] # cut each pair quantity
        shower_pairs = paired_mask.Apply(shower_pairs) # apply the mask to the shower_pairs index
        parameters = [SelectionQuantities.GetShowerPairValues(p, shower_pairs) for p in parameters] # get the events contributing to the cut shower pairs


    parameters = [Master.Unwrap(p) for p in parameters]
    parameters_pair = [Master.Unwrap(p) for p in parameters_pair]

    return parameters, parameters_pair


def SortPairsByEnergy(single, index=7, skip=[]):
    """
    Sorts shower pair quantities by energy, this requires that the only data inputted is the shower pair quantities i.e.
    you must cut on the data to only accept shower pairs.
    ----- Parameters -----
    data        : object of type Master.Data(), contains functions 
                  capable of retrieving data from the root file
    ----------------------
    """
    
    def Sort(d):
        """
        changes the shape of the nested data from [events, pairs] to [pairs, events].
        Not needed anywhere else.
        """
        l = [] # leading shower
        s = [] # subleading shower
        for i in range(len(d)):
            l.append( d[i][0] )
            s.append( d[i][1] )
        return [np.array(l, object), np.array(s, object)]
    
    sorted_data =  [[] for i in range(len(single))] # created empty array
    
    # first get the shower pair elements
    i = 0
    e = 0
    while i < len(single[index]):
        energy_0 = single[index][i] # sort by reconstructed energy
        energy_1 = single[index][i+1] # data unwrapped, so even elements is the first shower, odd is the second shower
        
        # get the leading and sub-leading indices
        if energy_0 >= energy_1:
            leading = i
            second = i+1
        if energy_0 < energy_1:
            leading = i+1
            second = i
        
        # add data to the sorted array
        for j in range(len(single)):
            if j in skip:
                sorted_data[j].append( [single[j][e], single[j][e]] ) # for beam data i.e. event number
            else:
                sorted_data[j].append([ single[j][leading], single[j][second] ])
        e += 1
        i += 2

    sorted_data = [Sort(d) for d in sorted_data] # reshape, use numpy?

    return sorted_data


def SignalByEnergyResidual(energy_residual, deviation=0.2, avg=True):
    """
    Defines a binary value 1, 0 to shower pairs based on the energy residuals of the showers in the pair,
    either considers the average or the best energy residual in the pair i.e. close to zero.
    ----- Parameters -----
    analog_signal       : function which maps the residual to a continuous distribution, to be discretised using the deviation
    signal              : binary signal calculated from analog_signal, 1 is for events well reconstructed, 0 for poor event reconstruction (ideally)
    ----------------------
    """
    if avg == True:
        # define the analogue signal as an average
        analog_signal = ( np.array(energy_residual[0]) + np.array(energy_residual[1]) ) / 2
    else:
        # define the analogue signal as the showers in a pair closest to zero energy residual
        energy_residual = np.array(energy_residual)
        analog_signal = np.argmin(abs(energy_residual), 0) # need to use the absolute value as residual renages from -1 to 1
        analog_signal = [energy_residual[analog_signal[i], i] for i in range(len(analog_signal))]

    signal = [0] * len(analog_signal) # define signal all with 0 initially
    for i in range(len(signal)):
        # if the energy residual is within -deviation and deviation, it is a signal, otherwise it is background (0)
        if -deviation < analog_signal[i] < deviation:
            signal[i] = 1
    
    print(np.unique(signal, return_counts=True)) # print info about the amount of signal and background
    return np.array(signal, object)


def CalculateFeatures(data : Master.Data, param=None, conditional=None, cut=None, cylinder=[3, -1, 4], mc=False):
    """
    will retrieve data in the root file created by the pi0 analyser.
    Then the data is used to calcuate the selection quantities and if
    a conditional is given, will apply a cut on the whole dataset.
    ----- Parameters -----
    data        : object of type Master.Data(), contains functions 
                  capable of retrieving data from the root file
    ----------------------
    """
    print("getting daughter info...")
    nHits = data.nHits()
    cnn_em = data.cnn_em()
    cnn_track = data.cnn_track()
    pandoraTag = data.pandoraTag()
    
    true_momentum = data.true_daugther_momentum()
    
    if mc is False:
        start_pos = data.start_pos()
        direction = data.direction()
        energy = data.energy()
        mc_energy = data.true_daughter_energy()
    if mc is True:
        start_pos = data.true_daugther_start_pos()
        direction = true_momentum.Normalise()
        energy = true_momentum.Magnitude()
        mc_energy = energy

    # custom data
    hit_radial = data.hit_radial()
    hit_longitudinal = data.hit_longitudinal()
    
    # calculate quantities from selected data
    print("calculate CNN score")
    cnn_score = SelectionQuantities.CNNScore(cnn_em, cnn_track)
    
    print("energy residual")
    energyResidual = (energy - mc_energy)/ mc_energy  
    
    print("start hits (not used for anything yet)")
    start_hits = SelectionQuantities.GetShowerStartHits(hit_radial, hit_longitudinal, cylinder)

    parameters = [pandoraTag, cnn_score, nHits, energyResidual, start_hits, energy, mc_energy] # store single shower properties in a list

    if param != None or conditional != None or cut != None:
        mask = Master.SelectionMask() # create mask object
        mask.InitiliseMask(nHits) # create mask array based on a template structure
        
        for i in range(len(param)):
            if param[i] < len(parameters):
                mask.CutMask(parameters[param[i]], cut[i], conditional[i]) # make a cut on the mask based on the quantity we want to cut
        
        parameters = [mask.Apply(p) for p in parameters] # apply the cut mask to each parameter

        ### cut data needed to calculate paired values ###
        start_pos = mask.Apply(start_pos)
        direction = mask.Apply(direction)
        energy = mask.Apply(energy)
        
    print("shower pairs")
    #shower_pairs = SelectionQuantities.GetShowerPairs(start_pos)
    shower_pairs = SelectionQuantities.GetShowerPairs_Hits(parameters[2])
    
    print("separation")
    pair_separation = SelectionQuantities.ShowerPairSeparation(start_pos, shower_pairs)
    pair_separation = np.reshape(pair_separation, (len(nHits)))
    
    print("pair angle")
    pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, direction)

    print("true pair angle")
    true_dir = true_momentum.Normalise()
    true_pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, true_dir)
    
    print("tangental distance to decay vertex")
    dist_vertex = SelectionQuantities.GetDecayVertexDistance(pair_separation, pair_angle)
    
    print("decay vertex")
    decay_vertex = SelectionQuantities.GetDecayVertex(shower_pairs, start_pos, direction, dist_vertex)
    
    print("pair energy")
    pair_energies, pair_leading, pair_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, energy)
    pair_mc_energies, pair_mc_leading, pair_mc_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, mc_energy)
    
    print("total true energy")
    total_true_energy = pair_mc_leading + pair_mc_second
    
    print("pair energy difference")    
    pair_energy_diff = pair_leading - pair_second
    
    print("Invariant mass")
    inv_mass = SelectionQuantities.InvariantMass(pair_angle, pair_energies)
    
    print("True Invariant mass")
    true_inv_mass = SelectionQuantities.InvariantMass(true_pair_angle, pair_mc_energies)
    

    parameters = [SelectionQuantities.GetShowerPairValues(p, shower_pairs) for p in parameters] # get the events contributing to the cut shower pairs
    parameters = [Master.Unwrap(p) for p in parameters]
    parameters = SortPairsByEnergy(parameters, index=5)


    print("total collection plane in shower hits")
    total_hits = parameters[2][0] + parameters[2][1]
    
    print("average cnn score")
    avg_cnn_score = (parameters[1][0] + parameters[1][1]) / 2
    
    
    parameters_pair = [pair_separation, pair_angle, pair_leading, pair_second, pair_energy_diff, dist_vertex, decay_vertex.x, decay_vertex.y, decay_vertex.z, inv_mass]
    parameters_pair = [Master.Unwrap(p) for p in parameters_pair]


    features = np.array([ avg_cnn_score, total_hits, *parameters_pair[0:-1], parameters_pair[-1], Master.Unwrap(total_true_energy), Master.Unwrap(true_pair_angle), Master.Unwrap(true_inv_mass) ])

    f_name = []

    for i in range(len(Master.feature_values)):
        f_name.append(Master.feature_values[i])

    features = pd.DataFrame(features.T, columns=f_name)

    features = features[features["invariant mass"] > 0]
    features = features[features["average CNN score"] > 0]
    features = features[features["pair energy difference"] > 0]

    return features


def CalculateDataFrame(data : Master.Data, param=None, conditional=None, cut=None, cylinder=[3, -1, 4], beamData=False):
    print("getting daughter info...")
    nHits = data.nHits()
    cnn_em = data.cnn_em()
    cnn_track = data.cnn_track()
    pandoraTag = data.pandoraTag()
    
    start_pos = data.start_pos()
    direction = data.direction()
    energy = data.energy()
    
    print("getting daughter truth info...")
    true_start_pos = data.true_daugther_start_pos()
    true_end_pos = data.true_daugther_end_pos()
    mc_energy = data.true_daughter_energy()
    true_momentum = data.true_daugther_momentum()
    
    if beamData is True:
        print("getting beam info...")
        beam_start_pos = data.beam_start_pos()
        beam_end_pos = data.beam_end_pos()
    
    # custom data
    hit_radial = data.hit_radial()
    hit_longitudinal = data.hit_longitudinal()
    
    # calculate quantities from selected data
    print("calculate CNN score")
    cnn_score = SelectionQuantities.CNNScore(cnn_em, cnn_track)
    
    print("energy residual")
    energyResidual = (energy - mc_energy)/ mc_energy
    
    if beamData is True:
        print("angle between beam and daughters")
        beam_angle = SelectionQuantities.BeamTrackShowerAngle(beam_start_pos, beam_end_pos, direction)
    else:
        beam_angle = []
        for i in range(len(nHits)):
            evt = []
            for j in range(len(nHits[i])):
                evt.append(-999)
            beam_angle.append(evt)
        beam_angle = np.array(beam_angle, object)
    
    print("angle between daughters and mc particle")
    mc_angle = SelectionQuantities.DaughterRecoMCAngle(true_start_pos, true_end_pos, direction)
    
    print("start hits")
    start_hits = SelectionQuantities.GetShowerStartHits(hit_radial, hit_longitudinal, cylinder)
    
    parameters = [pandoraTag, cnn_score, nHits, energyResidual, beam_angle, mc_angle, start_hits, energy, mc_energy] # store single shower properties in a list
    
    if param != None or conditional != None or cut != None:
        mask = Master.SelectionMask() # create mask object
        mask.InitiliseMask(nHits) # create mask array based on a template structure
        
        for i in range(len(param)):
            if param[i] < len(parameters):
                mask.CutMask(parameters[param[i]], cut[i], conditional[i]) # make a cut on the mask based on the quantity we want to cut
        
        parameters = [mask.Apply(p) for p in parameters] # apply the cut mask to each parameter
    
        ### cut data needed to calculate paired values ###
        start_pos = mask.Apply(start_pos)
        direction = mask.Apply(direction)
        energy = mask.Apply(energy)
        mc_energy = mask.Apply(mc_energy)
    
    
    print("shower pairs")
    shower_pairs = SelectionQuantities.GetShowerPairs_Hits(parameters[2])
    
    print("separation")
    pair_separation = SelectionQuantities.ShowerPairSeparation(start_pos, shower_pairs)
    pair_separation = np.reshape(pair_separation, (len(nHits)))
    
    print("pair angle")
    pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, direction)
    
    print("true pair angle")
    true_dir = true_momentum.Normalise()
    true_pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, true_dir)
    
    print("tangental distance to decay vertex")
    dist_vertex = SelectionQuantities.GetDecayVertexDistance(pair_separation, pair_angle)
    
    print("decay vertex")
    decay_vertex = SelectionQuantities.GetDecayVertex(shower_pairs, start_pos, direction, dist_vertex)
    
    print("pair energy")
    pair_energies, pair_leading, pair_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, energy)
    pair_mc_energies, pair_mc_leading, pair_mc_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, mc_energy)
    
    print("total true energy")
    total_true_energy = pair_mc_leading + pair_mc_second
    
    print("pair energy difference")    
    pair_energy_diff = pair_leading - pair_second
    
    print("Invariant mass")
    inv_mass = SelectionQuantities.InvariantMass(pair_angle, pair_energies)
    
    print("True Invariant mass")
    true_inv_mass = SelectionQuantities.InvariantMass(true_pair_angle, pair_mc_energies)
    
    
    parameters = [SelectionQuantities.GetShowerPairValues(p, shower_pairs) for p in parameters] # get the events contributing to the cut shower pairs
    parameters = [Master.Unwrap(p) for p in parameters]
    parameters = SortPairsByEnergy(parameters, index=7)
    
    
    print("total collection plane in shower hits")
    total_hits = parameters[2][0] + parameters[2][1]
    
    print("average cnn score")
    avg_cnn_score = (parameters[1][0] + parameters[1][1]) / 2
    
    
    parameters_pair = [pair_separation, pair_angle, pair_leading, pair_second, pair_energy_diff, dist_vertex, decay_vertex.x, decay_vertex.y, decay_vertex.z, inv_mass]
    parameters_pair = [Master.Unwrap(p) for p in parameters_pair]
    
    
    features = [ avg_cnn_score, total_hits, *parameters_pair[0:-1], parameters_pair[-1], Master.Unwrap(total_true_energy), Master.Unwrap(true_pair_angle), Master.Unwrap(true_inv_mass) ]
    
    parameters.pop(7) # already keep shower energy
    for i in range(len(parameters)):
        features.append(parameters[i][0])
        features.append(parameters[i][1])
    
    features = np.array(features)
    
    f_name = []
    
    for i in range(len(Master.feature_values)):
        f_name.append(Master.feature_values[i])
    
    shower_names = [Master.paramString.get(i) for i in Master.paramString]
    shower_names = shower_names[0:9]
    shower_names.pop(7)
    
    for i in range(len(shower_names)):
        f_name.append("leading " + shower_names[i])
        f_name.append("secondary " + shower_names[i])
    
    features = pd.DataFrame(features.T, columns=f_name)
    
    features = features[features["invariant mass (GeV)"] > 0]
    features = features[features["average CNN score"] > 0]
    features = features[features["pair energy difference (GeV)"] > 0]
    return features


def CalculateDataFrameNew(data : Master.Data, param=None, conditional=None, cut=None, cylinder=[3, -1, 4], beamData=False, mc=False):
    """
    Create Panda dataframe with features and truth quantities. No selection is applied, to be done in post.
    """

    """---RETRIEVE DATA---"""
    print("getting daughter truth info")
    true_start_pos = data.true_daugther_start_pos()
    true_end_pos = data.true_daugther_end_pos()
    mc_energy = data.true_daughter_energy()
    true_momentum = data.true_daugther_momentum()

    print("getting daughter info")
    nHits = data.nHits()
    cnn_em = data.cnn_em()
    cnn_track = data.cnn_track()
    pandoraTag = data.pandoraTag()
    
    if mc is False:
        start_pos = data.start_pos()
        direction = data.direction()
        energy = data.energy()
    if mc is True:
        start_pos = data.true_daugther_start_pos()
        direction = true_momentum.Normalise()
        energy = true_momentum.Magnitude()
    
    if beamData is True:
        print("getting beam info")
        beam_start_pos = data.beam_start_pos()
        beam_end_pos = data.beam_end_pos()
    
    print("getting custom data")
    hit_radial = data.hit_radial()
    hit_longitudinal = data.hit_longitudinal()
    """---RETRIEVE DATA---"""


    """---CALCULATE SHOWER QUANTITIES---"""
    print("calculating shower quantities")
    print("CNN score")
    cnn_score = SelectionQuantities.CNNScore(cnn_em, cnn_track)
    
    if beamData is True:
        print("angle between beam and daughters")
        beam_angle = SelectionQuantities.BeamTrackShowerAngle(beam_start_pos, beam_end_pos, direction)
    else:
        beam_angle = []
        for i in range(len(nHits)):
            evt = []
            for j in range(len(nHits[i])):
                evt.append(-999)
            beam_angle.append(evt)
        beam_angle = np.array(beam_angle, object)
    
    print("angle between daughters and mc particle")
    mc_angle = SelectionQuantities.DaughterRecoMCAngle(true_start_pos, true_end_pos, direction)
    
    print("start hits")
    start_hits = SelectionQuantities.GetShowerStartHits(hit_radial, hit_longitudinal, cylinder)
    
    single_parameters = [
        start_pos.x,
        start_pos.y,
        start_pos.z,
        direction.x,
        direction.y,
        direction.z,
        pandoraTag,
        cnn_em,
        cnn_track,
        cnn_score,
        nHits,
        energy,
        mc_energy,
        start_hits,
        beam_angle
    ] # store single shower properties in a list
    """---CALCULATE SHOWER QUANTITIES---"""


    if param != None or conditional != None or cut != None:
        print("doing preselection")
        mask = Master.SelectionMask() # create mask object
        mask.InitiliseMask(nHits) # create mask array based on a template structure
        
        for i in range(len(param)):
            if param[i] < len(single_parameters):
                mask.CutMask(single_parameters[param[i]], cut[i], conditional[i]) # make a cut on the mask based on the quantity we want to cut
        
        single_parameters = [mask.Apply(p) for p in single_parameters] # apply the cut mask to each parameter
        start_pos = mask.Apply(start_pos)
        direction = mask.Apply(direction)
        energy = mask.Apply(energy)
        mc_energy = mask.Apply(mc_energy)
        true_momentum = mask.Apply(true_momentum)


    """---CALCULATE SHOWER PAIR QUANTITIES---"""
    print("calculating shower pair quantities")
    print("pairing showers")
    shower_pairs = SelectionQuantities.GetShowerPairs_Hits(single_parameters[10])
    
    print("shower pair separation")
    pair_separation = SelectionQuantities.ShowerPairSeparation(start_pos, shower_pairs)
    pair_separation = np.reshape(pair_separation, (len(nHits)))
    
    print("opening angle")
    pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, direction)
    
    print("true opening angle")
    true_dir = true_momentum.Normalise()
    true_pair_angle = SelectionQuantities.ShowerPairAngle(shower_pairs, true_dir)
    
    print("tangental distance to decay vertex")
    dist_vertex = SelectionQuantities.GetDecayVertexDistance(pair_separation, pair_angle)
    
    print("decay vertex")
    decay_vertex = SelectionQuantities.GetDecayVertex(shower_pairs, start_pos, direction, dist_vertex)
    
    print("pair energy")
    pair_energies, pair_leading, pair_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, energy)
    pair_mc_energies, pair_mc_leading, pair_mc_second = SelectionQuantities.ShowerPairEnergy(shower_pairs, mc_energy)
    
    print("true total energy")
    total_true_energy = pair_mc_leading + pair_mc_second
    
    print("pair energy difference")    
    pair_energy_diff = pair_leading - pair_second
    
    print("Invariant mass")
    inv_mass = SelectionQuantities.InvariantMass(pair_angle, pair_energies)
    
    print("True invariant mass")
    true_inv_mass = SelectionQuantities.InvariantMass(true_pair_angle, pair_mc_energies)
    
    print("sorting single shower properties by shower pairs")
    single_parameters = [SelectionQuantities.GetShowerPairValues(p, shower_pairs) for p in single_parameters]
    single_parameters = [Master.Unwrap(p) for p in single_parameters]
    single_parameters = SortPairsByEnergy(single_parameters, index=11)
    
    print("total collection plane in shower hits")
    total_hits = single_parameters[10][0] + single_parameters[10][1]
    
    print("average cnn score")
    avg_cnn_score = (single_parameters[9][0] + single_parameters[9][1]) / 2
        
    parameters_pair = [
        avg_cnn_score,
        total_hits,
        pair_separation,
        pair_angle,
        pair_energy_diff,
        dist_vertex,
        decay_vertex.x,
        decay_vertex.y,
        decay_vertex.z,
        inv_mass,
        total_true_energy,
        true_pair_angle,
        true_inv_mass
    ]
    parameters_pair = [Master.Unwrap(p) for p in parameters_pair]
    """---CALCULATE SHOWER PAIR QUANTITIES---"""


    """---CREATE DATAFRAME---"""
    print("creating dataframe")
    features = [*parameters_pair]
    for i in range(len(single_parameters)):
        features.append(single_parameters[i][0])
        features.append(single_parameters[i][1])
    
    features = np.array(features)

    print("creating headers")
    f_name = []    
    for i in range(len(Master.feature_values_pairs)):
        f_name.append(Master.feature_values_pairs[i])
    
    for i in range(len(Master.feature_values_single_shower)):
        f_name.append("leading " + Master.feature_values_single_shower[i])
        f_name.append("secondary " + Master.feature_values_single_shower[i])
    
    features = pd.DataFrame(features.T, columns=f_name)
    """---CREATE DATAFRAME---"""

    """---CUT EMPTY/BAD EVENTS---"""
    features = features[features["invariant mass (GeV)"] > 0]
    features = features[features["average CNN score"] > 0]
    features = features[features["pair energy difference (GeV)"] > 0]
    """---CUT EMPTY/BAD EVENTS---"""
    return features
