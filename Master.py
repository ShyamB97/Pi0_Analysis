"""
Created on: 04/02/2022 12:26

Author: Shyam Bhuller

Description: Module containing core components of analysis code. 
"""
from __future__ import annotations

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
    def __init__(self, _filename : str) -> None:
        self.file = uproot.open(_filename)["pduneana/beamana"]
        self.nEvents = len(self.file["EventID"].array())
    def Get(self, item : str) -> ak.Array:
        """Load nTuple from root file as awkward array.

        Args:
            item (str): nTuple name in root file

        Returns:
            ak.Array: nTuple loaded
        """        
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
    def __init__(self, _filename : str = None) -> None:
        self.filename = _filename
        if self.filename != None:
            self.io = IO(self.filename)
            self.trueParticles = TrueParticles(self)
            self.recoParticles = RecoParticles(self)

    def SortByTrueEnergy(self) -> ak.Array:
        """returns index of shower pairs sorted by true energy (highest first).

        Returns:
            ak.Array: [description]
        """
        photonEnergy = self.trueParticles.energy[self.trueParticles.pdg == 22]
        return ak.argsort(photonEnergy, ascending=True)

    @timer
    def MatchMC(self, photon_dir, shower_dir, cut=0.25) -> tuple[ak.Array, ak.Array]:
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
    
    @timer
    def Filter(self, reco_filters : list = [], true_filters : list = []) -> Event:
        """Filter events.

        Args:
            reco_filters (list, optional): list of filters to apply to reconstructed data. Defaults to [].
            true_filters (list, optional): list of filters to apply to true data. Defaults to [].

        Returns:
            Event: filtered events
        """
        filtered = Event()
        filtered.trueParticles = self.trueParticles.Filter(true_filters)
        filtered.recoParticles = self.recoParticles.Filter(reco_filters)
        return filtered


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
    def __init__(self, events : Event) -> None:
        self.events = events # parent of TrueParticles
        if self.events.io != None:
            self.number = self.events.io.Get("g4_num")
            self.mother = self.events.io.Get("g4_mother")
            self.pdg = self.events.io.Get("g4_Pdg")
            self.energy = self.events.io.Get("g4_startE")
            self.momentum = ak.zip({"x" : self.events.io.Get("g4_pX"),
                                    "y" : self.events.io.Get("g4_pY"),
                                    "z" : self.events.io.Get("g4_pZ")})


    def Filter(self, filters : list) -> TrueParticles:
        """Filter true particle data.

        Args:
            filters (list): list of filters to apply to true data.

        Returns:
            TrueParticles: filtered data.
        """
        filtered = TrueParticles(self.events)
        filtered.number = self.number
        filtered.mother = self.mother
        filtered.pdg = self.pdg
        filtered.energy = self.energy
        filtered.momentum = self.momentum
        for f in filters:
            filtered.number = filtered.number[f]
            filtered.mother = filtered.mother[f]
            filtered.pdg = filtered.pdg[f]
            filtered.energy = filtered.energy[f]
            filtered.momentum = filtered.momentum[f]
        return filtered


class RecoParticles:
    leaves=(
        "reco_daughter_PFP_nHits_collection",
        "reco_daughter_allShower_dirX",
        "reco_daughter_allShower_dirY",
        "reco_daughter_allShower_dirZ",
        "reco_daughter_allShower_energy"
    )
    #? do something with leaves?

    def __init__(self, events : Event) -> None:
        self.events = events # parent of RecoParticles
        if self.events.io != None:
            self.nHits = self.events.io.Get("reco_daughter_PFP_nHits_collection")
            self.direction = ak.zip({"x" : self.events.io.Get("reco_daughter_allShower_dirX"),
                                     "y" : self.events.io.Get("reco_daughter_allShower_dirY"),
                                     "z" : self.events.io.Get("reco_daughter_allShower_dirZ")})
            self.energy = self.events.io.Get("reco_daughter_allShower_energy")


    def Filter(self, filters : list) -> RecoParticles:
        """Filter reconstructed data.

        Args:
            filters (list): list of filters to apply to reconstructed data.

        Returns:
            RecoParticles: filtered data.
        """
        filtered = RecoParticles(self.events)
        filtered.nHits = self.nHits
        filtered.direction = self.direction
        self.energy = self.energy
        for f in filters:
            filtered.nHits = filtered.nHits[f]
            filtered.direction = filtered.direction[f]
            filtered.energy = filtered.energy[f]
        return filtered

    @timer
    def GetPairValues(pairs, value) -> ak.Array:
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
    def AllShowerPairs(nd) -> list:
        """Get all shower pair combinations, excluding duplicates and self paired.

        Args:
            nd (ak.Array): number of daughters in an event

        Returns:
            list: Jagged array of pairs per event
        """
        pairs = []
        for i in range(len(nd)):
            comb = itertools.combinations(range(nd[i]), 2)
            pairs.append(list(comb))
        return pairs

    @timer
    def ShowerPairsByHits(hits) -> list:
        """pair reconstructed showers in an event by the number of hits.
        pairs the two largest showers per event.
        TODO figure out a way to do this without sorting events (or re-sort events?)

        Args:
            hits (ak.Array): number of collection plane hits of daughters per event

        Returns:
            list: shower pairs (maximum of one per event), note lists are easier to iterate through than np or ak arrays, hence the conversion
        """
        showers = ak.argsort(hits, ascending=False) # shower number sorted by nHits
        mask = ak.count(showers, 1) > 1
        showers = ak.pad_none(showers, 2, clip=True) # only keep two largest showers
        showers = ak.where( mask, showers, [[]]*len(mask) )
        pairs = ak.unflatten(showers, 1)
        return ak.to_list(pairs)



def Pi0MCFilter(events : Event, daughters : int = None) -> tuple[ak.Array, ak.Array]:
    """A filter for Pi0 MC dataset, selects events with a specific number of daughters
       and masks events where the direction has a null value -999. also returns indices
       of true photons to use when matching MC to reco.

    Args:
        events (Event): events being studied
        daughters (int): keep events with specific number of daughters. Defaults to None

    Returns:
        ak.Array: mask of events to filter
        ak.Array: mask of true photons to apply to true data
    """
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
