### Place this in a script that is inteded to be ran ###
import sys
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_base_libs') # need this to import custom scripts in different directories
sys.path.append('/home/sb16165/Documents/Pi0_Analysis/_prod4a_analysis') # need this to import custom scripts in different directories
######
from Master import Data, DataList, Conditional, Unwrap, ITEM, QUANTITY, feature_values_pairs, feature_values_single_shower
from SelectionQuantities import CalculateQuantities, GetShowerPairValues
from Analyser import SortPairsByEnergy
from MC_lib import FindPi0Signal

import numpy as np
import pandas as pd


def IntegretyCheck(name, tolerance=10E-12):
    """
    checks panda dataframe for shower pair mismatch, does so by calculating the invariant mass again
    and comparing it to the stored value (which is calculated pre-sorting the data).
    """
    data = pd.read_csv(name)
    data= data[data["invariant mass (GeV)"] != -999] # need to exlude null data (calculations won't match)
    inv_mass = data["invariant mass (GeV)"]
    inv_mass_calc = np.sqrt( (2 * data["leading energy (GeV)"] * data["secondary energy (GeV)"] * (1 - np.cos(data["opening angle (rad)"]))))

    match = np.allclose(inv_mass, inv_mass_calc, rtol=tolerance)
    print("matches: " + str(match))

    dif = abs(inv_mass - inv_mass_calc)
    print("absolute maximum: " + str(max(dif)))
    print("absolute maximum: " + str(min(dif)))
    return match


def ConvertEventInfo(data, reference):
    """
    change shape of event level data to shower pair shaped data
    """
    new = []
    for i in range(len(reference)):
        new_evt = []
        for j in range(len(reference[i])):
            new_evt.append(data[i])
        new.append(new_evt)
    return new


def SortByEnergy(single, energy_index=11, beam=[15,16]):
    """
    Sort events in dataset by energy of showers in a pair
    """
    for i in range(len(single[energy_index])):
        energy_evt = single[energy_index][i]
        for j in range(len(energy_evt)):
            energy_pair = energy_evt[j]
            for k in range(len(single)):
                if k not in beam: # don't sort event info
                    if energy_pair[0] < energy_pair[1]:
                        single[k][i][j].reverse()


filename="ROOTFiles/Prod4a_6GeV_BeamSim_00.root"
data = DataList(filename)

quantities = CalculateQuantities(data, allPairs=True)
pi0Signal = FindPi0Signal(quantities, data)

single_parameters = [
        data[ITEM.START_POS].x,
        data[ITEM.START_POS].y,
        data[ITEM.START_POS].z,
        data[ITEM.DIRECTION].x,
        data[ITEM.DIRECTION].y,
        data[ITEM.DIRECTION].z,
        data[ITEM.PANDORA_TAG],
        data[ITEM.CNN_EM],
        data[ITEM.CNN_TRACK],
        quantities[QUANTITY.CNN_SCORE],
        data[ITEM.HITS],
        data[ITEM.ENERGY],
        data[ITEM.TRUE_ENERGY],
        quantities[QUANTITY.START_HITS],
        quantities[QUANTITY.BEAM_ANGLE],
        data[ITEM.EVENT_ID], # custom to this dataset
        data[ITEM.RUN]
    ] # store single shower properties in a list

print("sorting single shower properties by shower pairs")

for p in range(len(single_parameters)):
    _type = list
    if p in [15, 16]:
        _type = int
    single_parameters[p] = GetShowerPairValues(single_parameters[p], quantities[QUANTITY.SHOWER_PAIRS], _type, unique=False, returnPaired=True)


SortByEnergy(single_parameters)
single_parameters[15] = ConvertEventInfo(single_parameters[15], single_parameters[11])
single_parameters[16] = ConvertEventInfo(single_parameters[16], single_parameters[11])

single_parameters = [Unwrap(s) for s in single_parameters]
single_parameters[15] = single_parameters[15][single_parameters[15] != -999]
single_parameters[16] = single_parameters[16][single_parameters[16] != -999]
single_parameters[15] = np.repeat(single_parameters[15][:, np.newaxis], 2, axis=1)
single_parameters[16] = np.repeat(single_parameters[16][:, np.newaxis], 2, axis=1)


hits = single_parameters[10]
eventID = single_parameters[15]

print("total collection plane in shower hits")
total_hits = single_parameters[10][:, 0] + single_parameters[10][:, 1]

print("average cnn score")
avg_cnn_score = (single_parameters[9][:, 0] + single_parameters[9][:, 1]) / 2

  
parameters_pair = [
    avg_cnn_score,
    total_hits,
    quantities[QUANTITY.SHOWER_PAIR_PANDORA_TAGS],
    quantities[QUANTITY.SHOWER_SEPERATION],
    quantities[QUANTITY.OPENING_ANGLE],
    quantities[QUANTITY.LEADING_ENERGY] - quantities[QUANTITY.SECONDARY_ENERGY], # pair energy difference
    quantities[QUANTITY.DIST_VERTEX],
    quantities[QUANTITY.DECAY_VERTEX].x,
    quantities[QUANTITY.DECAY_VERTEX].y,
    quantities[QUANTITY.DECAY_VERTEX].z,
    quantities[QUANTITY.INVARIANT_MASS],
    single_parameters[12][:, 0] + single_parameters[12][:, 1], # total true energy
    quantities[QUANTITY.TRUE_OPENING_ANGLE],
    quantities[QUANTITY.TRUE_INVARIANT_MASS],
    pi0Signal # custom to this data
]
parameters_pair = [Unwrap(p) for p in parameters_pair]
"""---CALCULATE SHOWER PAIR QUANTITIES---"""


"""---CREATE DATAFRAME---"""
print("creating dataframe")
features = [*parameters_pair]
for i in range(len(single_parameters)):
    features.append(single_parameters[i][:, 0])
    features.append(single_parameters[i][:, 1])

features = np.array(features)

print("creating headers")
f_name = []    
for i in range(len(feature_values_pairs)):
    f_name.append(feature_values_pairs[i])
f_name.append("true pi0 event")

for i in range(len(feature_values_single_shower)):
    f_name.append("leading " + feature_values_single_shower[i])
    f_name.append("secondary " + feature_values_single_shower[i])
f_name.append("leading event ID")
f_name.append("secondary event ID")
f_name.append("leading run number")
f_name.append("secondary run number")

features = pd.DataFrame(features.T, columns=f_name)
"""---CREATE DATAFRAME---"""

features.to_csv("BDT_input/Prod4a_6GeV_BeamSim_00_allshower.csv", index=False)
IntegretyCheck("BDT_input/Prod4a_6GeV_BeamSim_00_allshower.csv")
