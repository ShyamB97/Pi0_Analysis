#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:09:56 2021

@author: sb16165
"""

import numpy as np
from Master import Vector3, ITEM, QUANTITIY, CutDict
import sys


def Angle(v_0 : Vector3, v_1 : Vector3):
    """
    Calculates the angle between two vectors (or list of vectors).
    ----- Parameters -----
    v_n         : nth Vector3
    dot         : dot product of the two vectors
    mag_n       : magnitude of vector n
    cos_angle   : cosine of the angle
    ----------------------
    """
    if type(v_0) is not Vector3 or type(v_1) is not Vector3:
        raise TypeError("Inputs must be of type Master.Vector3")

    dot = v_0.x * v_1.x + v_0.y * v_1.y + v_0.z * v_1.z

    mag_0 = v_0.Magnitude()
    mag_1 = v_1.Magnitude()

    cos_angle = dot / ( mag_0 * mag_1 )

    # account for floating point error in calculation
    if cos_angle > 1:
        cos_angle = 1
    if cos_angle < -1:
        cos_angle = -1
    
    return np.arccos(cos_angle)


def GetShowerPairValues(parameter, pairs, _type=list):
    """
    Used to retrieve parameters of the shower pairs
    ----- Parameters -----
    paired_values       : parameters only for the showers making a pair
    evt                 : event level parameters
    evt_pairs           : shower pairs per event
    evt_paired values   : parameters of the showers forming pairs in the event
    ----------------------
    """
    paired_values = []
    for i in range(len(parameter)):
        evt = parameter[i]
        evt_pairs = pairs[i]
        
        if _type is list:
            evt_paired_values = []
            for j in range(len(evt_pairs)):
                values = [ evt[k] for k in evt_pairs[j]]
                evt_paired_values.extend(values)
            paired_values.append(evt_paired_values)
        else:
            if len(pairs[i]) > 0:
                paired_values.append(evt)
    
    return np.array(paired_values, dtype=object)


def GetShowerPairs(start_pos : Vector3):
    """
    Pairs daughter events based on their start vector separation.
    ----- Parameters -----
    pairs       : events paired by proximity
    evt         : start positions of daughters in a single event
    evt_pairs   : paired daughters per event
    min_dist    : smallest distance calculatated between one shower and all the others in an event
    dist        : distance between the jth daughter event and all other daughters (kth)
    pair_index  : index of the closests daughter to the jth daughter
    true_pairs  : events paired with duplicates and lone showers removed
    copy        : list to keep track of pairs already found
    pair        : current pair to check duplicates for
    rev         : pair in reverse order
    ----------------------
    """
    pairs = []
    for i in range(len(start_pos.x)):
        # calculate all possible shower pairs exlcuding null data
        evt = start_pos[i]
        evt_pairs = []
        if len(evt.x) > 1: # need more than one daughter
            for j in range(len(evt.x)):
                min_dist = sys.float_info.max # set the initial min distance to a large number
                pair_index = 0 # set initial index to 0
                if evt[j].x != -999: # ensure it is not null data
                    
                    # compare jth daughter to all others
                    for k in range(len(evt.x)):
                        if j == k: continue # dont compare to self
                        if evt[k].x == -999: continue # dont compare to null
                        dist = Vector3.Magnitude(evt[j] - evt[k]) # calculate distance
                        # update the minimum distance and daughter to pair
                        if dist < min_dist:
                            min_dist = dist
                            pair_index = k
                    evt_pairs.append([j, pair_index])
        
        # remove lone showers and duplicates
        true_pairs = []
        copy = []
        for p in range(len(evt_pairs)):
            pair = evt_pairs[p] # get pair
            rev = pair[::-1] # get the reverse of the list
            if rev in evt_pairs and rev not in copy:
                true_pairs.append(rev)
                copy.append(rev)
                copy.append(pair)
    
        pairs.append(true_pairs)
    return np.array(pairs, dtype=object)


def GetShowerPairs_Hits(nHits):
    shower_pairs = []
    for i in range(len(nHits)):
        shower_pair = []
        if(len(nHits[i]) > 1):
            index = np.linspace(0, len(nHits[i]) - 1, len(nHits[i]), dtype=int)
            index = [x for _,x in sorted(zip(nHits[i], index))]
            shower_pair = [ [index[-1], index[-2]] ] # wierd duoble index to work with other code
        shower_pairs.append(shower_pair)
    
    return np.array(shower_pairs, object)


def ShowerPairSeparation(start_pos : Vector3, shower_pairs):
    """
    Gets separation of paired daughter events, either all possible pairs or 
    the actual pairs (closest separation per daughter).
    ----- Parameters -----
    dists           : distance between the jth daughter event and all other daughters
    pair_dists      : distances per daughter pair
    evt_pairs_dist  : distances of the daughter pairs for all events
    x_0, ...        : start positions of the first shower pair
    x_1, ...        : start positions of the second shower pair
    ----------------------
    """
    evt_pairs_dist = []
    for i in range(len(shower_pairs)):
        pair_dists = []
        if(len(shower_pairs[i]) != 0):
            for pair in shower_pairs[i]:
                v_0 = start_pos[i][pair[0]]
                v_1 = start_pos[i][pair[1]]

                pair_dists.append(Vector3.Magnitude(v_1 - v_0))

        evt_pairs_dist.append(np.array(pair_dists, dtype=object))

    return np.array(evt_pairs_dist, object)


def ShowerPairAngle(shower_pairs, _dir : Vector3):
    """
    Gets the angle between daughter pairs.
    ----- Parameters -----
    x_0, ...        : direction vector components of the first shower pair
    x_1, ...        : direction vector components of the second shower pair
    evt_angles      : angles of shower per event
    ----------------------
    """

    # get the angle between the shower pairs
    angles = []
    for i in range(len(shower_pairs)):
        evt_angles = []
        if(len(shower_pairs[i]) != 0):
            for pair in shower_pairs[i]:
                v_0 = _dir[i][pair[0]]
                v_1 = _dir[i][pair[1]]
                
                #if v_0.x == -999 or v_1.x == -999:
                #    print("we have null data!")

                evt_angles.append( Angle(v_0, v_1) )
        angles.append(np.array(evt_angles))
    return np.array(angles, object)


def ShowerPairEnergy(shower_pairs, shower_energy):
    """
    Gets the Energy of the daughter pairs.
    ----- Parameters -----
    energy          : list of energy for pair 0 and pair 1
    pairs_energy    : shower energy paired by daughters for each event
    leading_energy  : shower in pair with the higher energy
    secondary_energy: shower in pair with the lower energy
    energy_min      : shower in pair with the higher energy
    energy_min      : shower in pair with the lower energy
    ----------------------
    """

    pairs_energy = []
    leading_energy = []
    secondary_energy = []
    for i in range(len(shower_pairs)):
        evt_pairs = shower_pairs[i]
        evt_energy = shower_energy[i]
    
        energy = []
        energy_min = []
        energy_max = []

        for j in range(len(evt_pairs)):
            pair = evt_pairs[j]
            paired_energy = [evt_energy[pair[0]],  evt_energy[pair[1]]]
 
            energy.append(paired_energy)
            energy_min.append(min(paired_energy))
            energy_max.append(max(paired_energy))

        pairs_energy.append(np.array(energy, object))
        leading_energy.append(np.array(energy_max, object))
        secondary_energy.append(np.array(energy_min, object))
    return np.array(pairs_energy, object), np.array(leading_energy, object), np.array(secondary_energy, object)


def BeamTrackShowerAngle(beam_start : Vector3, beam_end : Vector3, daughter_dir : Vector3):
    """
    Calculates the angle between the overall beam track direction and the 
    direction of its daughters.
    ----- Parameters -----
    beam_dists_x, ...   : component of beam distances travelled
    beam_dists          : beam distances travelled
    evt_x, ...          : direction vector compnents of daughters per event
    angle               : angle between beam track and daughter directions
    ----------------------
    """
    beam_dist_x = beam_end.x - beam_start.x
    beam_dist_y = beam_end.y - beam_start.y
    beam_dist_z = beam_end.z - beam_start.z


    beam_dist = ( beam_dist_x**2 + beam_dist_y**2 + beam_dist_z**2 )**0.5

    angles = []
    for i in range(len(daughter_dir.x)):
        evt = daughter_dir[i]

        angle = []
        if beam_dist[i] != 0:
            for j in range(len(evt.x)):
                if evt.x[j] == -999:
                    angle.append(-999)
                else:
                    angle.append( Angle(evt[j],
                                  Vector3(beam_dist_x[i], beam_dist_y[i], beam_dist_z[i]) ) )
        else:
            # add nulls for each daughter in events with no beam
            num_daughters = len( evt.x )
            angle = [-999] * num_daughters

        angles.append(angle)

    return np.array(angles, object)


def DaughterRecoMCAngle(true_start : Vector3, true_end : Vector3, _dir : Vector3):
    """
    Reconstrcut the angle between the reco daughter particle and the true MC particle.
    ----- Parameters -----
    true_dist_x, ...   : components of the distance of the MC particle
    angle              : recoMC angle per daughter in an event
    angles             : recoMC angles per event
    ----------------------
    """
    true_dist_x = true_end.x - true_start.x
    true_dist_y = true_end.y - true_start.y
    true_dist_z = true_end.z - true_start.z

    angles = []
    for i in range(len(true_start.x)):
        angle = []
        for j in range(len(true_start.x[i])):    
            if true_start.x[i][j] == -999:
                angle.append(-999)
            else:
                angle.append( Angle(Vector3(_dir.x[i][j], _dir.y[i][j], _dir.z[i][j]), 
                                    Vector3(true_dist_x[i][j], true_dist_y[i][j], true_dist_z[i][j]) ) )
        angles.append(angle)
    return np.array(angles, object)


def CNNScore(em, track):
    """
    Calculate the em/track like CNN score per daughter
    ----- Parameters -----
    cnnScore    : normalised em/shower like score
    ----------------------
    """
    cnnScore = []
    for i in range(len(em)):
        cnnScore_evt = []
        for j in range(len(em[i])):
            if em[i][j] != -999:
                cnnScore_evt.append( em[i][j] / (em[i][j] + track[i][j]) )
            else:
                cnnScore_evt.append(-999)
        cnnScore.append(cnnScore_evt)

    return np.array(cnnScore, object)


def ResidualShowerEnergy(shower_energy, true_energy):
    """
    Calculate the shower energy residual per daughter.
    ----- Parameters -----
    residual_energy    : energy residual per event per shower
    evt                : ennergy of daughters in an event
    evt_res            : energy residual per shower
    ----------------------
    """
    residual_energy = []
    for i in range(len(shower_energy)):
        evt = shower_energy[i]
        evt_res = []
        for j in range(len(evt)):
            if(evt[j] != -999):
                evt_res.append( (evt[j] - true_energy[i][j]) / true_energy[i][j] )
            else:
                evt_res.append(-999)
        residual_energy.append(evt_res)
    
    return np.array(residual_energy, object)


def InvariantMass(shower_pair_angles, shower_pair_energy):
    """
    Calculate the invariant mass from shower pairs.
    ----- Parameters -----
    inv_mass                : invariant mass per event per shower pair
    evt_angle               : angle between shower pairs for an event
    showerPair_energy       : energy of the showers forming the pair for an event
    angle                   : opening angle for a pair
    energy                  : shower pair energies
    ----------------------
    """    
    inv_mass = []
    for i in range(len(shower_pair_angles)):
        evt_angle = shower_pair_angles[i]
        evt_energy = shower_pair_energy[i]

        evt_inv_mass = []
        for j in range(len(evt_angle)):
            angle = evt_angle[j]
            energies = evt_energy[j]

            if angle != -999:
                if energies[0] >= 0 and energies[1] >= 0:
                    # calculate the invariant mass using particle kinematics
                    evt_inv_mass.append( np.sqrt(2 * energies[0] * energies[1] * (1 - np.cos(angle))) )
                else:
                    evt_inv_mass.append(-999)
            else:
                evt_inv_mass.append(-999)

        inv_mass.append(np.array(evt_inv_mass, object))

    return np.array(inv_mass, object)


def GetNumResonantParticles(start_pos : Vector3):
    """
    Gets the number of pairs i.e. the number of particles which produce these daughters,
    assuming it decays only into two shower pairs.
    ----- Parameters -----
    numParticles    : number of pairs in an event
    ----------------------
    """
    pairs = GetShowerPairs(start_pos)

    numParticles = []
    for i in range(len(pairs)):
        numParticles.append(len(pairs[i]))
    return np.array(numParticles, object)


def GetResonantMomentum(start_pos : Vector3, shower_energy):
    """
    Returns the monemtum of the parent per shower pair per event, assuming the
    daugthers are photons.
    ----- Parameters -----
    energies    : shower pair energies
    momenta     : resonant momenta
    evt         : energy pairs per event
    evt_mom     : resontant momenta per event
    e0, e1      : shower energies in a pair
    ----------------------
    """
    energies, _, _ = ShowerPairEnergy(start_pos, shower_energy)
    
    momenta = []
    for i in range(len(energies)):
        evt = energies[i]
        
        evt_mom = []
        for j in range(len(evt)):
            
            e0 = evt[j][0]
            e1 = evt[j][1]
            
            if evt[j] != -999:
                if e0 != -999 and e1 != -999:
                    evt_mom.append(e0 + e1)
                else:
                    evt_mom.append(-999)
            else:
                evt_mom.append(-999)

        momenta.append(evt_mom)
    return np.array(momenta, object)


def GetResonanceEnergy(start_pos : Vector3, _dir : Vector3, shower_energy):
    """
    Calculate the emergy of the parent particle produding the shower pairs.
    ----- Parameters -----
    mass        : Invariant mass calculated from shower pair energies
    momentum    : momentum of the parent particle assuming the daughters are massless
    res_mass    : parent mass of a pair for an event
    res_mom     : parent momentum of a pair for an event
    evt_energy  : parent energy for an event 
    energy      : parent energies per event
    ----------------------
    """
    mass = InvariantMass(start_pos, _dir, shower_energy)
    momentum = GetResonantMomentum(start_pos, _dir, shower_energy)
    
    energy = []
    for i in range(len(mass)):
        evt_energy = []
        for j in range(len(mass[i])):
            
            res_mass = mass[i][j]
            res_mom = momentum[i][j]
            
            if res_mass != -999 and res_mom != -999:
                evt_energy.append( ( res_mass**2 + res_mom**2 )**0.5 )
            else:
                evt_energy.append(-999)
        energy.append(evt_energy)
    return np.array(energy, object)


def GetShowerStartHits(hit_radial, hit_longitudinal, geometry=[1, -1, 4]):
    """
    Will return the number of hits within a cylindrical geometry about the shower
    start point.
    ----- Parameters -----
    geometry        : cylinder bounds,
                      [radius,
                       legnth along shower direction behind the start point,
                       legnth along shower direction in front the start point]
    start_hits      : number of hits within the geometry of each shower
    evt_r           : radial value to compare for each shower per event
    evt_l           : longitudinal value to compare for each shower per event
    shower_r        : radial value to compare for a shower
    shower_l        : longitudinal value to compare for a shower
    hits            : number of hits within the geometry for a shower
    ----------------------
    """
    start_hits = []
    for i in range(len(hit_radial)):
        # per event
        evt_r = hit_radial[i]
        evt_l = hit_longitudinal[i]
        evt_hits = []
        for j in range(len(evt_r)):
            # per shower
            shower_r = evt_r[j]
            shower_l = evt_l[j]
            hits = 0
            for k in range(len(shower_r)):
                # per hit
                if shower_r[k] < geometry[0] and geometry[1] < shower_l[k] < geometry[2]:
                    # hit near the start!
                    hits += 1
            evt_hits.append(hits)
        start_hits.append(evt_hits)
    return np.array(start_hits, object)


def GetDecayVertexDistance(pair_separation, pair_angle):
    """
    Calculates the tangential distcance from the midpoint of the shower start positions to the decay vertex.
    ----- Parameters -----
    dist_vertex     : tangential distance to shower pairs from the decay vertex
    evt_vertex      : vertex distance per event
    ----------------------
    """
    dist_vertex = []
    for i in range(len(pair_separation)):
        evt_vertex = []
        for j in range(len(pair_separation[i])):
            evt_vertex.append( abs(pair_separation[i][j] / ( 2 * np.tan(pair_angle[i][j]))) )
        dist_vertex.append(evt_vertex)
    return np.array(dist_vertex, dtype=object)


def GetDecayVertex(shower_pairs, start_pos : Vector3, direction : Vector3, dist_vertex : Vector3):
    """
    Estimate the shower vertex from the direction and start positions of the paired showers.
    ----- Parameters -----
    x               : x component of decay vertex
    y               : y component of decay vertex
    z               : z component of decay vertex
    evt_x           : x compenents of decay vertices per event
    evt_y           : y compenents of decay vertices per event
    evt_z           : z compenents of decay vertices per event
    evt_pos         : start posisitons per event
    evt_dir         : directions per event
    pos_1           : position of 1st shower in pair
    pos_2           : position of 2nd shower in pair
    dir_1           : direction of 1st shower in pair
    dir_2           : direction of 2st shower in pair
    midpoint        : point in the middle and parallel to the pair start positions
    avg_dir         : average direction of the shower pairs
    vertex          : pi0 decay vertex per shower pair
    ----------------------
    """
    # need to append each compnent of the vector for easier access
    x = []
    y = []
    z = []
    for i in range(len(shower_pairs)):
        evt_x = []
        evt_y = []
        evt_z = []
        if(len(shower_pairs[i]) != 0):
            for j in range(len(shower_pairs[i])):
                
                # get shower pair properties
                evt_pos = start_pos[i]
                evt_dir = direction[i]
                pos_1 = evt_pos[shower_pairs[i][j][0]]
                pos_2 = evt_pos[shower_pairs[i][j][1]]
                dir_1 = evt_dir[shower_pairs[i][j][0]]
                dir_2 = evt_dir[shower_pairs[i][j][1]]
                
                # check if null data exists i.e. shower pairing went wrong
                #if pos_1.x == -999 or pos_2.x == -999:
                #    print("we have null data!")
                
                midpoint = (pos_1 + pos_2) * 0.5
                avg_dir = (dir_1 + dir_2) * 0.5
                vertex = midpoint - avg_dir * dist_vertex[i][j]
                
                evt_x.append(vertex.x)
                evt_y.append(vertex.y)
                evt_z.append(vertex.z)
        x.append(evt_x)
        y.append(evt_y)
        z.append(evt_z)
    
    return Vector3(np.array(x, dtype=object), np.array(y, dtype=object), np.array(z, dtype=object))


def GetAllShowerPairs(nHits):
    """
    #returns all permutation of shower pairs, exludes self paired showers
    #and duplicate pairs. uses nHits to get the correct array lengths and doesn't
    #use any data.
    """

    def _unique(evt_pairs, pair):
        if len(evt_pairs) == 0:
            return True
        else:
            for p in evt_pairs:
                if pair[0] in p and pair[1] in p:
                    return False
                else:
                    continue
        return True

    all_pairs = []
    for i in range(len(nHits)):
        evt_pairs = []
        daughterIndex = range(len(nHits[i]))
        for j in daughterIndex:
            pairs = []
            for k in daughterIndex:
                if j == k: continue
                pair = [j, k]
                if _unique(evt_pairs, pair) is True:
                    pairs.append(pair)

            evt_pairs.extend(pairs)
        all_pairs.append(evt_pairs)
    return all_pairs


def CalculateQuantities(data_list : dict, allPairs : bool = False, param = None, conditional = None, cut = None):
    
    cnn_score = CNNScore(data_list[ITEM.CNN_EM], data_list[ITEM.CNN_TRACK]) # need to move this to ITEM dictionary

    data_list.update({QUANTITIY.CNN_SCORE : cnn_score})

    data_list = CutDict(data_list, param, conditional, cut)
    
    # compute quantities
    if allPairs is True:
        shower_pairs = GetAllShowerPairs(data_list[ITEM.HITS])
    else:
        shower_pairs = GetShowerPairs_Hits(data_list[ITEM.HITS])
    separation = ShowerPairSeparation(data_list[ITEM.START_POS], shower_pairs)
    opening_angle = ShowerPairAngle(shower_pairs, data_list[ITEM.DIRECTION])
    dist_vertex = GetDecayVertexDistance(separation, opening_angle)
    decay_vertex = GetDecayVertex(shower_pairs, data_list[ITEM.START_POS], data_list[ITEM.DIRECTION], dist_vertex)
    pair_energies, pair_leading, pair_second = ShowerPairEnergy(shower_pairs, data_list[ITEM.ENERGY])
    # truth quantities
    true_opening_angle = ShowerPairAngle(shower_pairs, data_list[ITEM.TRUE_MOMENTUM].Normalise())
    pair_mc_energies, _, _ = ShowerPairEnergy(shower_pairs, data_list[ITEM.TRUE_ENERGY])

    # create quantities dictionary
    return {
        ITEM.PANDORA_TAG                : data_list[ITEM.PANDORA_TAG],
        QUANTITIY.CNN_SCORE             : data_list[QUANTITIY.CNN_SCORE],
        ITEM.HITS                       : data_list[ITEM.HITS],
        ITEM.SHOWER_LENGTH              : data_list[ITEM.SHOWER_LENGTH],
        ITEM.SHOWER_ANGLE               : data_list[ITEM.SHOWER_ANGLE],
        QUANTITIY.BEAM_ANGLE            : BeamTrackShowerAngle(data_list[ITEM.BEAM_START_POS], data_list[ITEM.BEAM_END_POS], data_list[ITEM.DIRECTION]),
        QUANTITIY.MC_ANGLE              : DaughterRecoMCAngle(data_list[ITEM.TRUE_START_POS], data_list[ITEM.TRUE_END_POS], data_list[ITEM.DIRECTION]),
        QUANTITIY.START_HITS            : GetShowerStartHits(data_list[ITEM.HIT_RADIAL], data_list[ITEM.HIT_LONGITUDINAL], [3, -1, 4]),
        QUANTITIY.SHOWER_PAIRS          : shower_pairs,
        QUANTITIY.SHOWER_SEPERATION     : separation,
        QUANTITIY.OPENING_ANGLE         : opening_angle,
        QUANTITIY.DIST_VERTEX           : dist_vertex,
        QUANTITIY.DECAY_VERTEX          : decay_vertex,
        QUANTITIY.LEADING_ENERGY        : pair_leading,
        QUANTITIY.SECONDARY_ENERGY      : pair_second,
        QUANTITIY.INVARIANT_MASS        : InvariantMass(opening_angle, pair_energies),
        QUANTITIY.TRUE_OPENING_ANGLE    : true_opening_angle,
        QUANTITIY.TRUE_INVARIANT_MASS   : InvariantMass(true_opening_angle, pair_mc_energies)
    }