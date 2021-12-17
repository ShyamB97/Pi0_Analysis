# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:50:53 2021

@author: sb16165
"""

import uproot
import numpy as np
from enum import Enum


# Dictionary of parameters we can caculate and cut on
paramString = {
    0  : "pandora tag" ,
    1  : "CNN score",
    2  : "nHits" ,
    3  : "energy residual",
    4  : "beam angle",
    5  : "mc angle",
    6  : "start hits",
    7  : "energy",
    8  : "mc_energy",
    9  : "pair seperation",
    10 : "pair angle",
    11 : "pair leading",
    12 : "pair second",
    13 : "invariant mass",
    14 : "true pair angle"
}

paramAxesLabels = {
    0  : "Pandora tag of daughters",
    1  : "CNN scores of beam daughters",
    2  : "Number of collection plane hits",
    3  : "Reconstruted energy residual",
    4  : "Angle between beam track and daughter shower (rad)",
    5  : "Angle between shower and MC parent (rad)",
    6  : "Hits close to the shower start",
    7  : "reconstructed shower energy (GeV)",
    8  : "mc energy of beam daughters (GeV)",
    9  : "Shower pair Separation (cm)",
    10 : "Angle between shower pairs (rad)",
    11 : "Leading shower energy (GeV)",
    12 : "Secondary shower energy (GeV)",
    13 : "Shower pair invariant mass (GeV)",
    14 : "True shower pair angle (rad)"
}

feature_values = {
    0  : "average CNN score",
    1  : "collection plane hits of shower pairs",
    2  : "pair seperation (cm)",
    3  : "opening angle (rad)",
    4  : "leading energy (GeV)",
    5  : "secondary energy (GeV)",
    6  : "pair energy difference (GeV)",
    7  : "decay vertex tangential distance (cm)",
    8  : "decay vertex x position (cm)",
    9  : "decay vertex y position (cm)",
    10 : "decay vertex z position (cm)",
    11 : "invariant mass (GeV)",
    12 : "total true energy (GeV)",
    13 : "true opening angle (GeV)",
    14 : "true invariant mas (GeV)s"
}

feature_values_single_shower = {
    0  : "start position x (cm)",
    1  : "start position y (cm)",
    2  : "start position z (cm)",
    3  : "direction x",
    4  : "direction y",
    5  : "direction z",
    6  : "pandora tag",
    7  : "em score",
    8  : "track score",
    9  : "cnn score",
    10 : "number of collection plane hits",
    11 : "energy (GeV)",
    12 : "true energy (GeV)",
    13 : "start hits",
    14 : "beam angle (rad)"
}

feature_values_pairs = {
    0  : "average CNN score",
    1  : "collection plane hits of shower pairs",
    2  : "average pandora tag",
    3  : "pair seperation (cm)",
    4  : "opening angle (rad)",
    5  : "pair energy difference (GeV)",
    6  : "decay vertex tangential distance (cm)",
    7  : "decay vertex x position (cm)",
    8  : "decay vertex y position (cm)",
    9  : "decay vertex z position (cm)",
    10 : "invariant mass (GeV)",
    11 : "total true energy (GeV)",
    12 : "true opening angle (rad)",
    13 : "true invariant mass (GeV)"
}

def ParseCommandLine(args):
    """
    Parse the command line options given to a script for analysing/plotting analyser outputs.
    ----- Parameters -----
    root_filename       : root file path
    reference_filename  : name to use when saving plots
    subDirectory        : output directory
    beamData            : are we processing beam data?
    plotType            : when selecting data, plot either "single" (accepted data) or "both" (rejected and accepted)
    cutData             : do we make a selection on the data?
    config_file         : configuration file to define the selection criteria
    param               : parameters to cut on
    conditional         : conditions of the cut
    cut                 : values to cut on
    ----------------------
    """
    if "--help" in args or len(args) == 1:
        print("usage: --file <root file> --outName <file name to append> --outDir <output directory> --parameter <0 - 8> --conditional < l, g, ne, e > --cut <cut value> --beam <1/0> --plot <single/both> --config <config file for multiple cuts>")
        exit(0)
    if "--file" in args:
        root_filename = args[args.index("--file") + 1]
    else:
        print("no file chosen!")
        exit(1)
    if "--outName" in args:
        reference_filename = "_" + args[args.index("--outName") + 1]
    else:
        reference_filename = "_" + root_filename[19:-5]
    if "--outDir" in args:
        subDirectory = args[args.index("--outDir") + 1] + "/"
    else:
        subDirectory = root_filename[19:-5] + "/"
    if "--beam" in args:
        beamData = bool(int(args[args.index("--beam") + 1]))
    else:
        beamData = True

    if "--plot" in args:
        plotType = args[args.index("--plot") + 1]
    else:
        plotType = "single"
    
    if "--config" in args:
        config_file = args[args.index("--config") + 1]
        cutData = True
        conditional = None
        param = None
        cut = None
    elif "--parameter" in args:
        param = [int(args[args.index("--parameter") + 1])]
        if "--conditional" in args:
            conditional = [StringToCond(args[args.index("--conditional") + 1])]
            if conditional == -999:
                print("Invalid input check help text:")
                exit(1)
        else:
            print("need to specify a condition!")
        if "--cut" in args:
            cut = [float(args[args.index("--cut") + 1])]
        else:
            print("need to specify a cut!")
            exit(1)
        cutData = True
        config_file = None
    else:
        config_file = None
        cutData = False
        conditional = None
        param = None
        cut = None
    return root_filename, reference_filename, subDirectory, beamData, plotType, cutData, config_file, param, conditional, cut



def ParseConfig(file_name : str):
    """
    Function to read a config file which contains a string of values that dictate which parameters
    are cut on, where to cut and the condition. Removes any strings after # to
    allow comments to be written.
    
    In the config file the three quantities is split by a space, and a new line for a different selection e.g:
    
    1 g 0.6
    2 l 50
    
    Translates to selecting events where:
    
    cnn score > 0.6
    nHits < 50
    
    ----- Parameters -----
    param         : quantities to cut on, int based on paramString dictionary
    conditional   : bounds for the cut; >, <, != or =.
    cut           : values to cut on, depends on the parameter
    file          : config file
    line          : line in the file
    ----------------------
    """
    param = []
    conditional = []
    cut = []
    file = open(file_name, "r") # open file
    # parse each line
    for line in file:
        line = line.split("#", 1)[0] # remove comments
        param.append( int(line.split(" ", 2)[0]) ) # get the paramter by splitting the string at the first two space bars
        conditional.append( StringToCond(line.split(" ", 2)[1]) ) # get middle characters and convert to Conditional class object
        cut.append( float(line.split(" ", 2)[2]) ) # get end characters
    return param, conditional, cut


class Conditional(Enum):
    GREATER = 1
    LESS = 2
    EQUAL = 3
    NOT_EQUAL = 4


def InvertConditional(conditional : Conditional):
    """
    Will invert a provided conditional. Uses Master.Conditional
    """
    if conditional == Conditional.GREATER:
        return Conditional.LESS
    if conditional == Conditional.LESS:
        return Conditional.GREATER
    if conditional == Conditional.EQUAL:
        return Conditional.NOT_EQUAL
    if conditional == Conditional.NOT_EQUAL:
        return Conditional.EQUAL


def GetCondString(conditional : Conditional):
    """
    Will convert the meaning of a conditional to string format. Uses Master.Conditional
    """
    if conditional == Conditional.NOT_EQUAL:
        return " != "
    if conditional == Conditional.EQUAL:
        return " = "
    if conditional == Conditional.GREATER:
        return " > "
    if conditional == Conditional.LESS:
        return " < "
    else:
        return " ??? "


def StringToCond(string : str):
    """
    Parses the command line input for --conditional into a variable of type Master.Conditional.
    """
    if string == "ne":
        return Conditional.NOT_EQUAL
    elif string == "e":
        return Conditional.EQUAL
    elif string == "g":
        return Conditional.GREATER
    elif string == "l":
        return Conditional.LESS
    else:
        return -999


class Vector3():
    """
    Vector data structure.
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __call__(self):
        return

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)    

    def __mul__(self, other):
        return Vector3(other * self.x, other * self.y, other * self.z)

    def __rmul__(self, other):
        return Vector3(other * self.x, other * self.y, other * self.z)

    def __getitem__(self, i):
        return Vector3(self.x[i], self.y[i], self.z[i])

    def __len__(self):
        return len(self.x)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def Magnitude(self):
        return ( self.x**2 + self.y**2 + self.z**2 )**0.5

    def Normalise(self):
        mag = self.Magnitude()
        return Vector3(self.x / mag, self.y / mag, self.z / mag)
    
    def RemoveNull(self):
        return Vector3(self.x[self.x != -999], self.y[self.y != -999], self.z[self.z != -999])

    def ListToVector3(_list):
        _x = []
        _y = []
        _z = []

        def AppendComponents(vector, l_x, l_y, l_z):
            if type(vector) is not Vector3:
                raise TypeError("list must only contain Vector3 objects!")
            else:
                l_x.append(vector.x)
                l_y.append(vector.y)
                l_z.append(vector.z)

        for i in range(len(_list)):
            _evtX = []
            _evtY = []
            _evtZ = []
            if hasattr(_list[i], '__iter__') is False:
                AppendComponents(_list[i], _x, _y, _z)
            else:
                for j in range(len(_list[i])):            
                    AppendComponents(_list[i][j], _evtX, _evtY, _evtZ)
                _y.append(np.array(_evtY, object))
                _z.append(np.array(_evtZ, object))
                _x.append(np.array(_evtX, object))
        return Vector3(np.array(_x, object), np.array(_y, object), np.array(_z, object))


def GetData(file : str, name : str, convert : bool = False):
    """
    Get dataset from ROOT file. will unwrap if needed.
    ----- Parameters -----
    tree_pduneana   : directory the datasets are stored in, by default this should
                      never change.
    data            : recovered data from ROOT file, converted into a nested numoy array, 
                      first depth is the events if the data set is nested. The following
                      depths depend on the data in question
    ----------------------
    """
    file = uproot.open(file)
    tree_pduneana = file['pduneana/beamana']
    try:
        data = tree_pduneana[name].arrays(library="np")[name]
    except uproot.KeyInFileError:
        print(name + " not found in " + "file, moving on...")
        return None
    
    # converts the top level data into a list
    # needed to convert from stlVectors(c++ nested vectors) to lists
    if convert is True:
        for i in range(len(data)):
            data[i] = data[i].tolist()
    
    return data


def Unwrap(data):
    """
    Unwraps nested numpy arrays, needed because the datasets per event do not
    have the same length so is hard for plotting functions to interpret.
    Use mainly for plotting data. Note this unrwaps data at a depth of one,
    for higher depths i.e. nested arrays of nested arrays, use the function
    more than once i.e. Unravel(Unravel(...
    ----- Parameters -----
    _list   : list of data unwrapped.
    obj     : object stored at the first depth
    ----------------------
    """
    _list = []
    for obj in data:
        try:
            for item in obj:
                _list.append(item)
        except TypeError:
            _list.append(obj)
    return np.array(_list, object)


# compress functions for quantities likely to be retrieved together
class Data:
    """
    Will retrieve supported datasets as needed, to keep computational times down.
    Most data applies to the shower, or MC truth particle producing the shower,
    if not it is stated in the name.
    """
    def __init__(self, filename : str):
        self.filename = filename
    
    # Event ID
    def eventID(self):
        return np.array(GetData(self.filename, "EventID"), dtype=object)
    def run_number(self):
        return np.array(GetData(self.filename, "Run"))

    # reco daughter data
    def start_pos(self):
        return Vector3(
            GetData(self.filename, "reco_daughter_allShower_startX"),
            GetData(self.filename, "reco_daughter_allShower_startY"),
            GetData(self.filename, "reco_daughter_allShower_startZ")
        )
    def direction(self):
        return Vector3(
            GetData(self.filename, "reco_daughter_allShower_dirX"),
            GetData(self.filename, "reco_daughter_allShower_dirY"),
            GetData(self.filename, "reco_daughter_allShower_dirZ")
        )
    def energy(self):
        return GetData(self.filename, "reco_daughter_allShower_energy") / 1000 # MeV -> GeV
    def nHits(self):
        return GetData(self.filename, "reco_daughter_PFP_nHits_collection")
    def cnn_em(self):
        return GetData(self.filename, "reco_daughter_PFP_emScore_collection")
    def cnn_track(self):
        return GetData(self.filename, "reco_daughter_PFP_trackScore_collection")

    # beam particle (parent) data
    def beam_start_pos(self):
        return Vector3(
            GetData(self.filename, "reco_beam_startX"),
            GetData(self.filename, "reco_beam_startY"),
            GetData(self.filename, "reco_beam_startZ")
        )
    def beam_end_pos(self):
        return Vector3(
            GetData(self.filename, "reco_beam_endX"),
            GetData(self.filename, "reco_beam_endY"),
            GetData(self.filename, "reco_beam_endZ")
        )
    
    def pandoraTag(self):
        return GetData(self.filename, "pandoraTag")
    
    # quantities to calculate cylinder hits
    def hit_radial(self):
        return GetData(self.filename, "hitRadial", convert=True)
    def hit_longitudinal(self):
        return GetData(self.filename, "hitLongitudinal", convert=True)
    
    # CNN score without averaging the track and EM score (i.e. not the same way as done in PDSPAnalyser.py)
    def CNNScore(self):
        return GetData(self.filename, "CNNScore_collection")
    
    def shower_length(self):
        return GetData(self.filename, "reco_daughter_allShower_length")
    
    def shower_angle(self):
        return GetData(self.filename, "reco_daughter_allShower_coneAngle")

    # Geant4 Particle data
    def g4_pdg(self):
        return GetData(self.filename, "g4_Pdg")
    def g4_start_pos(self):
        return Vector3(
            GetData(self.filename, "g4_startX"),
            GetData(self.filename, "g4_startY"),
            GetData(self.filename, "g4_startZ")
        )

    # mc truth daughter data
    def true_daugther_pdg(self):
        return GetData(self.filename, "reco_daughter_PFP_true_byHits_pdg")
    def true_daugther_start_pos(self):
        return Vector3(
            GetData(self.filename, "reco_daughter_PFP_true_byHits_startX"),
            GetData(self.filename, "reco_daughter_PFP_true_byHits_startY"),
            GetData(self.filename, "reco_daughter_PFP_true_byHits_startZ")
        )
    def true_daugther_end_pos(self):
        return Vector3(
            GetData(self.filename, "reco_daughter_PFP_true_byHits_endX"),
            GetData(self.filename, "reco_daughter_PFP_true_byHits_endY"),
            GetData(self.filename, "reco_daughter_PFP_true_byHits_endZ")
        )
    def true_daugther_momentum(self):
        return Vector3(
            GetData(self.filename, "reco_daughter_PFP_true_byHits_pX"),
            GetData(self.filename, "reco_daughter_PFP_true_byHits_pY"),
            GetData(self.filename, "reco_daughter_PFP_true_byHits_pZ")
        )
    def true_daughter_energy(self):
        return GetData(self.filename, "reco_daughter_PFP_true_byHits_startE")

    def true_parent_pdg(self):
        return GetData(self.filename, "reco_daughter_PFP_true_byHits_parent_pdg")


class SelectionMask:
    """
    Class which holds a mask of the data retrieved from a root file. the mask
    has values of 1 initially and you cut the mask (1 -> 0) depending on various
    selections you make on various quantities of the data. You can repeatidly cut on
    the mask and once it is finally applied to a set of data, the data will contain
    daughter and beam events which satisfy all the cuts made to the mask.
    
    This code only works on doubly nested data (nested lists can be different in size)
    """
    def __init__(self, mask=None):
        if mask is None:
            mask = []
        self.mask = mask

    def InitiliseMask(self, reference):
        """
        Create the mask, and set its shape to the reference data
        """
        for i in range(len(reference)):
            # create a list of ones with the same shape as the ith reference i.e. beam event
            evt_mask = np.ones(reference[i].shape)
            self.mask.append(evt_mask)
        self.mask = np.array(self.mask, object)
    
    def CutMask(self, parameter, cut, conditional):
        """
        goes through each element of the mask and parameter, if the nested element of the parameter
        satisfies a condition, the mask at that position is kept (remains 1), otherwise it is cut (1 -> 0).
        """
        for i in range(len(parameter)):
            evt_parameter = parameter[i]
            evt_mask = self.mask[i]
            for j in range(len(evt_parameter)):
                
                # keep events where parameter[i][j] == cut
                if conditional == Conditional.EQUAL and evt_parameter[j] != cut:
                    evt_mask[j] = 0
                
                # keep events where parameter[i][j] != cut
                if conditional == Conditional.NOT_EQUAL and evt_parameter[j] == cut:
                    evt_mask[j] = 0
                
                # keep events where parameter[i][j] < cut
                if conditional == Conditional.LESS and evt_parameter[j] >= cut:
                    evt_mask[j] = 0
                
                # keep events where parameter[i][j] > cut
                if conditional == Conditional.GREATER and evt_parameter[j] <= cut:
                    evt_mask[j] = 0

    def ApplyMask(self, data, beamData):
        """
        Remove elements of data where mask[i][j] == 0. works on beam data by
        saying if ANY daughters have a mask = 1, it is kept.
        """
        selected_data = []
        for i in range(len(self.mask)):
            evt_mask = self.mask[i]
            evt_data = data[i]
            new_evt = []
            
            if beamData is True:
                if 1 in evt_mask:
                    new_evt = evt_data
                    selected_data.append(new_evt)
                else:
                    selected_data.append( np.array(new_evt) )
            else:
                for j in range(len(evt_mask)):
                    if evt_mask[j] == 1:
                        new_evt.append(evt_data[j])
                selected_data.append( np.array(new_evt) )

        return np.array(selected_data, object)

    def ApplyMaskVector(self, data, beamData):
        """
        The ApplyMask function but to work on a Vector3 of data.
        """
        selected_data = Vector3(self.ApplyMask(data.x, beamData), 
                        self.ApplyMask(data.y, beamData), 
                        self.ApplyMask(data.z, beamData))
        return selected_data


    def ApplyMaskShowerPair(self, data, pairs):
        """
        Use mask to cut on shower pair properties
        """
        selected_data = []
        for i in range(len(self.mask)):
            evt_mask = self.mask[i]
            evt_data = data[i]
            evt_pairs = pairs[i]
            keep_index = []
            new_evt = []
            for j in range(len(evt_mask)):
                if evt_mask[j] == 1:
                    keep_index.append(j)
            for k in range(len(evt_data)):
                if evt_pairs[k][0] in keep_index and evt_pairs[k][1] in keep_index:
                    new_evt.append(evt_data[k])
            selected_data.append( np.array(new_evt) )
        return selected_data


    def Apply(self, data, beamData=False, showerPairs=None):
        """
        Function to call when actually using this class. will call ApplyMask or
        ApplyMaskVector depeding on the type.
        """
        if type(data) is Vector3 and showerPairs is None:
            selected_data = self.ApplyMaskVector(data, beamData)
        if type(data) is not Vector3 and showerPairs is None:
            selected_data = self.ApplyMask(data, beamData)
        if type(data) is not Vector3 and showerPairs is not None:
            print("cutting on shower pairs")
            selected_data = self.ApplyMaskShowerPair(data, showerPairs)
        return selected_data

    def ApplyMaskToSelf(self):
        """
        Remove all 0 elements in the mask. Done so you can cut the mask and the data in stages
        rather than at once.
        """
        self.mask = self.ApplyMask(self.mask)


class ITEM(Enum):
    """
    Information retrievable from the analyser.
    """
    START_POS = 0 # shower start position
    DIRECTION = 1 # shower direction
    ENERGY = 2 # shower energy
    HITS = 3 # number of collection plane hits
    CNN_EM = 4 # shower em-like score
    CNN_TRACK = 5 # shower track-like score
    PANDORA_TAG = 6 # pandora tag

    BEAM_START_POS = 7 # beam start position
    BEAM_END_POS = 8 # beam end position

    TRUE_START_POS = 9 # true shower start position
    TRUE_END_POS = 10 # true shower end position
    TRUE_MOMENTUM = 11 # true shower particle momentum
    TRUE_ENERGY = 12 # true shower energy
    TRUE_PDG = 13 # true daughter pdg

    HIT_RADIAL = 14 # radial component of distance from hits to cylindrical geometry
    HIT_LONGITUDINAL = 15 # longitudinal ""

    SHOWER_LENGTH = 16 # shower length
    SHOWER_ANGLE = 17 # cone angle of shower

    EVENT_ID = 18 # event number
    RUN = 19 # run number



class QUANTITY(Enum):
    """
    Values needed to be calculated using ITEM.
    """
    CNN_SCORE = 0 # cnn score i.e. how shower like
    BEAM_ANGLE = 1 # angle of shower wrt to beam
    MC_ANGLE = 2 # angle of shower wrt to true particle that produced the shower
    START_HITS = 3 # hits close to start of shower
    SHOWER_PAIRS = 4 # shower pairs 
    SHOWER_PAIR_PANDORA_TAGS = 5 # pandora tag of both showers in a pair
    SHOWER_SEPERATION = 6 # distance between shower pair start positions
    OPENING_ANGLE = 7 # angle between shower pairs
    DIST_VERTEX = 8 # estimated distance to Pi0 decay vertex
    DECAY_VERTEX = 9 # estimated Pi0 point of decay
    LEADING_ENERGY = 10 # largest shower energy in a pair
    SECONDARY_ENERGY = 11 # smallest shower energy in a pair
    INVARIANT_MASS = 12 # shower pair invariant mass
    # truth quantities
    TRUE_OPENING_ANGLE = 13
    TRUE_INVARIANT_MASS = 14


def BugFixCNN(cnn_score):
    """
    Fixed bugged CNN score from analyser
    """
    for i in range(len(cnn_score)):
        for j in range(len(cnn_score[i])):
            if cnn_score[i][j] != -999:
                cnn_score[i][j] = 2 - cnn_score[i][j]
    return cnn_score


# create data dictionary
def DataList(filename : str, param : ITEM = None, conditional : Conditional = None, cut=None):
    data =  Data(filename)
    _dict = {
    ITEM.START_POS        : data.start_pos(),
    ITEM.DIRECTION        : data.direction(),
    ITEM.ENERGY           : data.energy(),
    ITEM.TRUE_ENERGY      : data.true_daughter_energy(),
    ITEM.HITS             : data.nHits(),
    ITEM.CNN_EM           : data.cnn_em(),
    ITEM.CNN_TRACK        : data.cnn_track(),
    QUANTITY.CNN_SCORE   : BugFixCNN(data.CNNScore()), # need to move enum to ITEM
    ITEM.PANDORA_TAG      : data.pandoraTag(),
    ITEM.BEAM_START_POS   : data.beam_start_pos(),
    ITEM.BEAM_END_POS     : data.beam_end_pos(),
    ITEM.TRUE_START_POS   : data.true_daugther_start_pos(),
    ITEM.TRUE_END_POS     : data.true_daugther_end_pos(),
    ITEM.TRUE_MOMENTUM    : data.true_daugther_momentum(),
    ITEM.TRUE_PDG         : data.true_daugther_pdg(),
    ITEM.HIT_RADIAL       : data.hit_radial(),
    ITEM.HIT_LONGITUDINAL : data.hit_longitudinal(),
    ITEM.SHOWER_LENGTH    : data.shower_length(),
    ITEM.SHOWER_ANGLE     : data.shower_angle(),
    ITEM.EVENT_ID         : data.eventID(),
    ITEM.RUN              : data.run_number()
    }
    # some information about the data
    print("Number of events: " + str(len(_dict[ITEM.EVENT_ID])))
    print("Number of showers: " + str(len(Unwrap(_dict[ITEM.HITS]))))
    return CutDict(_dict, param, conditional, cut)


def UpdateShowerPair(pair, pair_index, added_showers):
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
    return pair_new


def UpdateShowerPairs(shower_pairs):
    """
    Update index of shower pairs after cutting data
    """
    new_pairs = []
    for i in range(len(shower_pairs)):
        new_pairs_evt = []
        added_showers = []
        for j in range(len(shower_pairs[i])):
            new_pairs_evt.append( UpdateShowerPair(shower_pairs[i][j], j, added_showers) )
        new_pairs.append(new_pairs_evt)
    return np.array(new_pairs, object)


def CutDict(_dict : str, param = None, conditional : Conditional = None, cut = None):

    def GetShowersInPairs(parameter, pairs, _type=list):
        """
        Used to retrieve values for the unique showers in the pairs
        """
        paired_values = []
        for i in range(len(parameter)):
            evt = parameter[i]
            evt_pairs = pairs[i]
            
            if _type is list:
                evt_paired_values = []
                already_added = []
                for j in range(len(evt_pairs)):
                    values = [ evt[k] for k in evt_pairs[j]]
                    for v in range(len(values)):
                        if evt_pairs[j][v] not in already_added:
                            already_added.append( evt_pairs[j][v] )
                            evt_paired_values.append(values[v])
                paired_values.append(evt_paired_values)
            else:
                if len(pairs[i]) > 0:
                    paired_values.append(evt)
        
        return np.array(paired_values, dtype=object)

    single_param = [ITEM.PANDORA_TAG, QUANTITY.CNN_SCORE, ITEM.HITS, ITEM.SHOWER_LENGTH, ITEM.SHOWER_ANGLE, QUANTITY.BEAM_ANGLE, QUANTITY.MC_ANGLE, QUANTITY.START_HITS] # single shower quantities in the quantities dataset
    contains_shower_pairs = QUANTITY.SHOWER_PAIRS in _dict

    if ITEM.EVENT_ID in _dict:
        print("current number of events: " + str(len(Unwrap(_dict[ITEM.EVENT_ID]))))

    if ITEM.HITS in _dict:
        print("current number of showers: " + str(len(Unwrap(_dict[ITEM.HITS]))))

    if contains_shower_pairs is True:
        print("current number of shower pairs: " + str(len(Unwrap(_dict[QUANTITY.SHOWER_SEPERATION]))))

    if param != None or conditional != None or cut != None:
        for i in range(len(param)):
            # check parameter to cut is actually in the dictionary of data
            if param[i] not in _dict: continue

            # check if data is in bound of cuts, if not then skip
            if conditional == Conditional.GREATER:
                if min(Unwrap(_dict[param[i]])) > cut: continue
            if conditional == Conditional.LESS:
                if max(Unwrap(_dict[param[i]])) < cut: continue
            if conditional == Conditional.EQUAL:
                if len(set(Unwrap(_dict[param[i]]))) == 1: continue
            if conditional == Conditional.NOT_EQUAL:
                if cut not in set(Unwrap(_dict[param[i]])): continue

            print("Cutting: " + str(param[i]))
            mask = SelectionMask()

            # initialse mask depending on the structure of data to cut
            if contains_shower_pairs is False:
                mask.InitiliseMask(_dict[ITEM.HITS])
            else:
                # dont cut on QUANTITY.SHOWER_PAIRS as the data type is a list, so when infering
                # the shape of the mask, it will get the correct structure i.e. event[shower pair data]
                # instead of: event[shower pair[data0, data1]]
                mask.InitiliseMask(_dict[QUANTITY.SHOWER_SEPERATION])

            mask.CutMask(_dict[param[i]], cut[i], conditional[i])
            for item in _dict:
                if item in [ITEM.BEAM_START_POS, ITEM.BEAM_END_POS, ITEM.EVENT_ID, ITEM.RUN]:
                    evtLevel = True
                else:
                    evtLevel = False
                if contains_shower_pairs:
                    # assume we are cutting on quantities dataset
                    if item in single_param or param[i] in single_param:
                        continue # dont cut on single shower properties
                    _dict.update( {item : mask.Apply(_dict[item], evtLevel)} )
                else:
                    _dict.update( {item : mask.Apply(_dict[item], evtLevel)} )
            if contains_shower_pairs:
                # handle single parameter data:
                for single in single_param:
                    _dict.update( {single : GetShowersInPairs(_dict[single], _dict[QUANTITY.SHOWER_PAIRS] ) } )
                # update shower pair indices
                _dict[QUANTITY.SHOWER_PAIRS] = UpdateShowerPairs(_dict[QUANTITY.SHOWER_PAIRS])


    if ITEM.EVENT_ID in _dict:
        print("remaining number of events: " + str(len(Unwrap(_dict[ITEM.EVENT_ID]))))

    if ITEM.HITS in _dict:
        print("remaining number of showers: " + str(len(Unwrap(_dict[ITEM.HITS]))))

    if QUANTITY.SHOWER_SEPERATION in _dict:
        print("remaining number of shower pairs: " + str(len(Unwrap(_dict[QUANTITY.SHOWER_SEPERATION]))))
    return _dict