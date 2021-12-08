from base_libs.Master import Data, DataList, Conditional, Unwrap, ITEM, QUANTITIY, feature_values_pairs, feature_values_single_shower
from base_libs.SelectionQuantities import CalculateQuantities, GetShowerPairValues
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
    inv_mass = data["invariant mass (GeV)"]
    inv_mass_calc = np.sqrt( (2 * data["leading energy (GeV)"] * data["secondary energy (GeV)"] * (1 - np.cos(data["opening angle (rad)"]))))

    match = np.allclose(inv_mass, inv_mass_calc, rtol=tolerance)
    print("matches: " + str(match))

    dif = abs(inv_mass - inv_mass_calc)
    print("absolute maximum: " + str(max(dif)))
    print("absolute maximum: " + str(min(dif)))
    return match

filename="ROOTFiles/Prod4a_1GeV_BeamSim_00.root"
data = DataList(filename, ITEM.HITS, Conditional.GREATER, 50)

quantities = CalculateQuantities(data)
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
        quantities[QUANTITIY.CNN_SCORE],
        data[ITEM.HITS],
        data[ITEM.ENERGY],
        data[ITEM.TRUE_ENERGY],
        quantities[QUANTITIY.START_HITS],
        quantities[QUANTITIY.BEAM_ANGLE],
        data[ITEM.EVENT_ID], # custom to this dataset
        data[ITEM.RUN]
    ] # store single shower properties in a list

print("sorting single shower properties by shower pairs")

for p in range(len(single_parameters)):
    _type = list
    if p in [15, 16]:
        _type = int
    single_parameters[p] = GetShowerPairValues(single_parameters[p], quantities[QUANTITIY.SHOWER_PAIRS], _type)

single_parameters = [Unwrap(p) for p in single_parameters]
single_parameters = SortPairsByEnergy(single_parameters, index=11, skip=[15, 16])


print("total collection plane in shower hits")
total_hits = single_parameters[10][0] + single_parameters[10][1]

print("average cnn score")
avg_cnn_score = (single_parameters[9][0] + single_parameters[9][1]) / 2

  
parameters_pair = [
    avg_cnn_score,
    total_hits,
    quantities[QUANTITIY.SHOWER_SEPERATION],
    quantities[QUANTITIY.OPENING_ANGLE],
    quantities[QUANTITIY.LEADING_ENERGY] - quantities[QUANTITIY.SECONDARY_ENERGY], # pair energy difference
    quantities[QUANTITIY.DIST_VERTEX],
    quantities[QUANTITIY.DECAY_VERTEX].x,
    quantities[QUANTITIY.DECAY_VERTEX].y,
    quantities[QUANTITIY.DECAY_VERTEX].z,
    quantities[QUANTITIY.INVARIANT_MASS],
    single_parameters[12][0] + single_parameters[12][1], # total true energy
    quantities[QUANTITIY.TRUE_OPENING_ANGLE],
    quantities[QUANTITIY.TRUE_INVARIANT_MASS],
    pi0Signal # custom to this data
]
parameters_pair = [Unwrap(p) for p in parameters_pair]
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

"""---CUT EMPTY/BAD EVENTS---"""
features = features[features["invariant mass (GeV)"] > 0]
features = features[features["average CNN score"] > 0]
features = features[features["pair energy difference (GeV)"] > 0]
"""---CUT EMPTY/BAD EVENTS---"""

features.to_csv("BDT_input/Prod4a_1GeV_BeamSim_00_allshower_test.csv")
IntegretyCheck("BDT_input/Prod4a_1GeV_BeamSim_00_allshower_test.csv")