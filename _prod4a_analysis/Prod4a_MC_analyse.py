import Master
from Master import ITEM, QUANTITIY
import SelectionQuantities
import Plots

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

class G4(Enum):
    """
    Geant4 particle information.
    """
    PDG = 0 # particle pdgs
    START_POS = 1 # start positions


def Pi0Filter(data_list, g4):
    """
    Filter data to to only keep events where a Pi0 (111)
    was generated in the Geant4 simulation.
    """
    # create blank data list dictionary
    data_list_new = {
        ITEM.START_POS        : [],
        ITEM.DIRECTION        : [],
        ITEM.ENERGY           : [],
        ITEM.TRUE_ENERGY      : [],
        ITEM.HITS             : [],
        ITEM.CNN_EM           : [],
        ITEM.CNN_TRACK        : [],
        ITEM.PANDORA_TAG      : [],
        ITEM.BEAM_START_POS   : [],
        ITEM.BEAM_END_POS     : [],
        ITEM.TRUE_START_POS   : [],
        ITEM.TRUE_END_POS     : [],
        ITEM.TRUE_MOMENTUM    : [],
        ITEM.HIT_RADIAL       : [],
        ITEM.HIT_LONGITUDINAL : [],
        ITEM.SHOWER_LENGTH    : [],
        ITEM.SHOWER_ANGLE     : [],
        ITEM.EVENT_ID         : [],
        ITEM.TRUE_PDG         : []
        #ITEM.RUN              : []
    }

    g4_new = {
        G4.PDG : [],
        G4.START_POS : []
    }

    # loop through recovered data
    for item in ITEM:
        print("filtering: " + str(item))
        item_list = []

        # loop through each event
        for i in range(len(g4[G4.PDG])):
            if 111 in g4[G4.PDG][i]:
                item_list.append(data_list[item][i])

        # switch from a list of Vector3's to a Vector3 of lists
        if type(item_list[0]) is Master.Vector3:
            item_list = Master.Vector3.ListToVector3(item_list)
        data_list_new[item] = item_list
    
    # loop through g4 data
    for item in G4:
        print("filtering: " + str(item))
        item_list = []

        # loop through each event
        for i in range(len(g4[G4.PDG])):
            if 111 in g4[G4.PDG][i]:
                item_list.append(g4[item][i])
        # switch from a list of Vector3's to a Vector3 of lists
        if type(item_list[0]) is Master.Vector3:
            item_list = Master.Vector3.ListToVector3(item_list)
        g4_new[item] = item_list

    return data_list_new, g4_new


data = Master.Data("ROOTFiles/pi0_83.root")

# create data dictionary
data_list = Master.DataList(data)


print("getting G4 information")
g4_data = {
    G4.PDG : data.g4_pdg(),
    G4.START_POS : data.g4_start_pos()
}

data_list, g4_data = Pi0Filter(data_list, g4_data) # filter for events with Geant4 Pi0's

quantities = SelectionQuantities.CalculateQuantities(data_list)

# plot all non Vector3 quantities
for q in quantities:
    if type(quantities[q]) is not Master.Vector3:
        plt.figure()
        quantity = Master.Unwrap(quantities[q])
        Plots.PlotHist( quantity[quantity != -999], bins=25, xlabel=q.name)

plt.figure()
inv_mass = quantities[QUANTITIY.INVARIANT_MASS]
inv_mass_new = []
for i in range(len(inv_mass)):
    if len(inv_mass[i]) == 0:
        inv_mass_new.append(-1)
    else:
        for item in inv_mass[i]:
            inv_mass_new.append(item)
inv_mass = np.array(inv_mass_new)

candidates = np.array(data_list[ITEM.EVENT_ID])
#run = np.array(data_list[ITEM.RUN])

candidates = candidates[inv_mass > 0]
#run = run[inv_mass > 0]
inv_mass = inv_mass[inv_mass > 0]
#run = run[inv_mass < 1]
inv_mass = inv_mass[inv_mass < 1]
Plots.PlotHist(inv_mass, bins=25, xlabel=QUANTITIY.INVARIANT_MASS)
plt.show()