"""
Created on: 04/02/2022 12:26

Author: Shyam Bhuller

Description: Module containing core components of analysis code. 
"""
import uproot
import awkward as ak
import time
import vector
import numpy as np
import itertools

def timer(func):
    """Decorator which times a function.

    Args:
        func (function): function to time
    """
    def wrapper_function(*args, **kwargs):
        """times func, returns outputs
        Returns:
            any: func output
        """
        s = time.time()
        out = func(*args,  **kwargs)
        print(f'{func.__name__!r} executed in {(time.time()-s):.4f}s')
        return out
    return wrapper_function


class IO:
    #? handle opening root file, and setting the ith event
    #? should this be a module rather than class?
    def __init__(self, _filename) -> None:
        self.file = uproot.open(_filename)["pduneana/beamana"]
        self.nEvents = len(self.file["EventID"].array())
    def Get(self, item : str) -> ak.Array:
        try:
            return self.file[item].array()
        except uproot.KeyInFileError:
            print(f"{item} not found in file, moving on...")
            return None


"""
class contains relevant data, and functions to retrieve/calculate data.
TODO currently loads all data from root files when initialized, try to see if you can define null instance varaibles and
TODO assign them when called for the first time (not sure if possible)

TODO make an abstract class particle from which TrueParticles and RecoParticles are derived from
TODO move functions into classes e.g. shower matching, pair by hits, opening angle, invariant mass etc.

Events
    root file name
    RecoParticles
    TrueParticles

TrueParticles
    number
    mother
    pdg code
    energy
    momentum
    mass
    start/end position

RecoParticles
    number
    mother
    beam
    start position
    direction
    nHits
    hitRad
    hitLong
    energy

"""

class Event:
    def __init__(self, _filename : str) -> None:
        self.filename = _filename
        self.io = IO(self.filename)
        self.trueParticles = TrueParticles(self.io, self)
        self.recoParticles = RecoParticles(self.io, self)

    def SortByTrueEnergy(self):
        photonEnergy = self.trueParticles.energy
        photonEnergy = photonEnergy[photonEnergy == 22]
        return ak.argsort(photonEnergy, ascending=True)

    @timer
    def MatchMC(self, photon_dir, shower_dir, cut=0.25):
        """ Matches Reconstructed showers to true photons and selected the best events
            i.e. ones which have both errors of less than 0.25 radians. Only works for
            events with two reconstructed showers and two true photons per event.

        Args:
            photon_dir (ak.Record created by vector): direction of true photons
            shower_dir (ak.Record created by vector): direction of reco showers

        Returns:
            ak.Array: shower indices in order of true photon number
            ak.Array: mask which indicates which pi0 decays are "good"
        """
        # angle of all reco showers wrt to each true photon per event i.e. error
        angle_error_0 = vector.angle(shower_dir, photon_dir[:, 0])
        angle_error_1 = vector.angle(shower_dir, photon_dir[:, 1])

        # get smallest angle wrt to each true photon
        m_0 = ak.unflatten(ak.min(angle_error_0, -1), 1)
        m_1 = ak.unflatten(ak.min(angle_error_1, -1), 1)
        angles = ak.concatenate([m_0, m_1], -1)

        # get shower which had the smallest angle
        #NOTE reco showers are now sorted by true photon number essentially. 
        m_0 = ak.unflatten(ak.argmin(angle_error_0, -1), 1)
        m_1 = ak.unflatten(ak.argmin(angle_error_1, -1), 1)
        showers = ak.concatenate([m_0, m_1], -1)

        # get events where both reco MC angles are less than 0.25 radians
        mask = np.logical_and(angles[:, 0] < cut, angles[:, 1] < cut)

        # check how many showers had the same reco match to both true particles
        same_match = showers[:, 0][mask] == showers[:, 1][mask]
        print(f"number of events where both photons match to the same shower: {ak.count(ak.mask(same_match, same_match) )}")

        return showers, mask


class TrueParticles:
    leaves=(
        "g4_num",
        "g4_mother",
        "g4_pX",
        "g4_pY",
        "g4_pZ",
        "g4_startE"
    )
    #? do something with leaves?
    def __init__(self, io : IO, events : Event) -> None:
        self.number = io.Get("g4_num")
        self.mother = io.Get("g4_mother")
        self.pdg = io.Get("g4_Pdg")
        self.energy = io.Get("g4_startE")
        self.momentum = ak.zip({"x" : io.Get("g4_pX"),
                                "y" : io.Get("g4_pY"),
                                "z" : io.Get("g4_pZ")})
        self.events = events # parent of TrueParticles


class RecoParticles:
    leaves=(
        "reco_daughter_PFP_nHits_collection",
        "reco_daughter_allShower_dirX",
        "reco_daughter_allShower_dirY",
        "reco_daughter_allShower_dirZ",
        "reco_daughter_allShower_energy"
    )
    #? do something with leaves?
    def __init__(self, io : IO, events : Event) -> None:
        self.nHits = io.Get("reco_daughter_PFP_nHits_collection")
        self.direction = ak.zip({"x" : io.Get("reco_daughter_allShower_dirX"),
                                 "y" : io.Get("reco_daughter_allShower_dirY"),
                                 "z" : io.Get("reco_daughter_allShower_dirZ")})
        self.energy = io.Get("reco_daughter_allShower_energy")
        self.events = events # parent of TrueParticles

    @timer
    def GetPairValues(pairs, value):
        """get shower pair values, in pairs

        Args:
            pairs (list): shower pairs per event
            value (ak.Array): values to retrieve

        Returns:
            ak.Array: paired showers values per event
        """
        paired = []
        for i in range(len(pairs)):
            pair = pairs[i]
            evt = []
            for j in range(len(pair)):
                if len(pair[j]) > 0:
                    evt.append( [value[i][pair[j][0]], value[i][pair[j][1]]] )
                else:
                    evt.append([])
            paired.append(evt)
        return ak.Array(paired)

    @timer
    def AllShowerPairs(nd):
        """Get all shower pair combinations, excluding duplicates and self paired.

        Args:
            nd (Array): number of daughters in an event

        Returns:
            list: Jagged array of pairs per event
        """
        pairs = []
        for i in range(len(nd)):
            comb = itertools.combinations(range(nd[i]), 2)
            pairs.append(list(comb))
        return pairs

    @timer
    def ShowerPairsByHits(hits):
        """pair reconstructed showers in an event by the number of hits.
        pairs the two largest showers per event.
        TODO figure out a way to do this without sorting events (or re-sort events?)

        Args:
            hits (Array): number of collection plane hits of daughters per event

        Returns:
            [list]: shower pairs (maximum of one per event), note lists are easier to iterate through than np or ak arrays, hence the conversion
        """
        showers = ak.argsort(hits, ascending=False) # shower number sorted by nHits
        mask = ak.count(showers, 1) > 1
        showers = ak.pad_none(showers, 2, clip=True) # only keep two largest showers
        showers = ak.where( mask, showers, [[]]*len(mask) )
        pairs = ak.unflatten(showers, 1)
        return ak.to_list(pairs)



def Pi0MCFilter(events : Event, daughters : int = None):
    nDaughter = ak.count(events.recoParticles.nHits, 1)
    
    if daughters == None:
        r_mask = nDaughter > 1
    else:
        r_mask = nDaughter == 2
    
    photons = events.trueParticles.mother == 1 # get only primary daughters
    photons = events.trueParticles.pdg == 22 # get only photons
    t_mask = ak.num(photons[photons], -1) == 2 # exclude pi0 -> e+ + e- + photons

    valid = np.logical_and(r_mask, t_mask) # events which have 2 reco daughters and correct pi0 decay

    null = ak.any(events.recoParticles.direction.x == -999, 1) # exclude events where direction couldn't be calculated

    valid = np.logical_and(valid, np.logical_not(null))
    #? can we do without returning photons?
    return valid, photons
