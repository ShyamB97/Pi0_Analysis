"""
Created on: 04/02/2022 12:26

Author: Shyam Bhuller

Description: Module containing core components of analysis code. 
"""

import warnings
import uproot
import awkward as ak
import time
import numpy as np
import itertools
# custom modules
import vector


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


def GenericFilter(particleData, filters):
    for f in filters:
            for var in vars(particleData):
                if hasattr(getattr(particleData, var), "__getitem__"):
                    try:
                        setattr(particleData, var, getattr(particleData, var)[f])
                    except:
                        warnings.warn(f"Couldn't apply filters to {var}.")


class IO:
    #? handle opening root file, and setting the ith event
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


class Data:
    def __init__(self, _filename : str = None) -> None:
        self.filename = _filename
        if self.filename != None:
            self.io = IO(self.filename)
            self.trueParticles = TrueParticleData(self)
            self.recoParticles = RecoParticleData(self)

    @property
    def SortedTrueEnergyMask(self) -> ak.Array:
        """ Returns index of shower pairs sorted by true energy (highest first).

        Args:
            primary (bool): sort index of primary pi0 decay photons (particle gun MC)

        Returns:
            ak.Array: sorted indices of true photons
        """
        mask = self.trueParticles.pdg == 22
        mask = np.logical_and(mask, self.trueParticles.mother == 1)
        return ak.argsort(self.trueParticles.energy[mask], ascending=True)

    @timer
    def GetMCMatchingFilters(self, cut=0.25, returnAngles=False):
        """ Matches Reconstructed showers to true photons and selected the best events
            i.e. ones which have both errors of less than 0.25 radians. Only works for
            events with two reconstructed showers and two true photons per event.

        Args:
            photon_dir (ak.Record created by vector): direction of true photons
            shower_dir (ak.Record created by vector): direction of reco showers

        Returns:
            ak.Array: shower indices in order of true photon number
            ak.Array: boolean mask of showers not matched
            ak.Array: mask which indicates which pi0 decays are "good"
            ak.Array (optional): minimum angle between showers and each true photon
        """
        # angle of all reco showers wrt to each true photon per event i.e. error
        photon_dir = vector.normalize(self.trueParticles.momentum)[self.trueParticles.truePhotonMask]
        angle_error_0 = vector.angle(self.recoParticles.direction, photon_dir[:, 0])
        angle_error_1 = vector.angle(self.recoParticles.direction, photon_dir[:, 1])

        # get smallest angle wrt to each true photon
        m_0 = ak.unflatten(ak.min(angle_error_0, -1), 1)
        m_1 = ak.unflatten(ak.min(angle_error_1, -1), 1)
        angles = ak.concatenate([m_0, m_1], -1)

        # make boolean mask of showers that aren't matched
        t_0 = ak.min(angle_error_0, -1) != angle_error_0
        t_1 = ak.min(angle_error_1, -1) != angle_error_1
        unmatched_mask = np.logical_and(t_0, t_1)

        # get shower which had the smallest angle
        m_0 = ak.unflatten(ak.argmin(angle_error_0, -1), 1)
        m_1 = ak.unflatten(ak.argmin(angle_error_1, -1), 1)
        matched_mask = ak.concatenate([m_0, m_1], -1)

        # get events where both reco MC angles are less than 0.25 radians
        selection = np.logical_and(angles[:, 0] < cut, angles[:, 1] < cut)

        # check how many showers had the same reco match to both true particles
        same_match = matched_mask[:, 0][selection] == matched_mask[:, 1][selection]
        print(f"number of events where both photons match to the same shower: {ak.count(ak.mask(same_match, same_match))}")

        if returnAngles is True:
            return matched_mask, unmatched_mask, selection, angles
        else:
            return matched_mask, unmatched_mask, selection
    

    def Filter(self, reco_filters : list = [], true_filters : list = [], returnCopy : bool = True):
        """ Filter events.

        Args:
            reco_filters (list, optional): list of filters to apply to reconstructed data. Defaults to [].
            true_filters (list, optional): list of filters to apply to true data. Defaults to [].
            returnCopy   (bool, optional): return a copy of filtered events, or change self

        Returns:
            Event: filtered events
        """
        if returnCopy is False:
            self.trueParticles.Filter(true_filters, returnCopy)
            self.recoParticles.Filter(reco_filters, returnCopy)
        else:
            filtered = Data()
            filtered.trueParticles = self.trueParticles.Filter(true_filters)
            filtered.recoParticles = self.recoParticles.Filter(reco_filters)
            return filtered


class TrueParticleData:
    def __init__(self, events : Data) -> None:
        self.events = events # parent of TrueParticles
        if hasattr(self.events, "io"):
            self.number = self.events.io.Get("g4_num")
            self.mother = self.events.io.Get("g4_mother")
            self.pdg = self.events.io.Get("g4_Pdg")
            self.energy = self.events.io.Get("g4_startE")
            self.momentum = ak.zip({"x" : self.events.io.Get("g4_pX"),
                                    "y" : self.events.io.Get("g4_pY"),
                                    "z" : self.events.io.Get("g4_pZ")})
            self.direction = vector.normalize(self.momentum)
            self.startPos = ak.zip({"x" : self.events.io.Get("g4_startX"),
                                    "y" : self.events.io.Get("g4_startY"),
                                    "z" : self.events.io.Get("g4_startZ")})
            self.endPos = ak.zip({"x" : self.events.io.Get("g4_endX"),
                                  "y" : self.events.io.Get("g4_endY"),
                                  "z" : self.events.io.Get("g4_endZ")})

    @property
    def truePhotonMask(self):
        photons = self.mother == 1 # get only primary daughters
        photons = np.logical_and(photons, self.pdg == 22)
        return photons


    def Filter(self, filters : list, returnCopy : bool = True):
        """Filter true particle data.

        Args:
            filters (list): list of filters to apply to true data.

        Returns:
            TrueParticles: filtered data.
        """
        if returnCopy is False:
            GenericFilter(self, filters)
        else:
            filtered = TrueParticleData(Data())
            filtered.number = self.number
            filtered.mother = self.mother
            filtered.pdg = self.pdg
            filtered.energy = self.energy
            filtered.momentum = self.momentum
            filtered.direction = self.direction
            filtered.startPos = self.startPos
            filtered.endPos = self.endPos
            GenericFilter(filtered, filters)
            return filtered


class RecoParticleData:
    def __init__(self, events : Data) -> None:
        self.events = events # parent of RecoParticles
        print(events.filename)
        if hasattr(self.events, "io"):
            self.beam_number = self.events.io.Get("beamNum")
            self.number = self.events.io.Get("reco_PFP_ID")
            self.mother = self.events.io.Get("reco_PFP_Mother")
            self.nHits = self.events.io.Get("reco_daughter_PFP_nHits_collection")
            self.direction = ak.zip({"x" : self.events.io.Get("reco_daughter_allShower_dirX"),
                                     "y" : self.events.io.Get("reco_daughter_allShower_dirY"),
                                     "z" : self.events.io.Get("reco_daughter_allShower_dirZ")})
            self.startPos = ak.zip({"x" : self.events.io.Get("reco_daughter_allShower_startX"),
                                    "y" : self.events.io.Get("reco_daughter_allShower_startY"),
                                    "z" : self.events.io.Get("reco_daughter_allShower_startZ")})
            self.energy = self.events.io.Get("reco_daughter_allShower_energy")
            self.momentum = self.GetMomentum()


    def GetMomentum(self):
        # TODO turn into attribute with @property
        mom = vector.prod(self.energy, self.direction)
        mom = ak.where(self.direction.x == -999, {"x": -999,"y": -999,"z": -999}, mom)
        mom = ak.where(self.energy < 0, {"x": -999,"y": -999,"z": -999}, mom)
        return mom


    def Filter(self, filters : list, returnCopy : bool = True):
        """Filter reconstructed data.

        Args:
            filters (list): list of filters to apply to reconstructed data.

        Returns:
            RecoParticles: filtered data.
        """
        if returnCopy is False:
            GenericFilter(self, filters)
        else:
            filtered = RecoParticleData(Data())
            filtered.nHits = self.nHits
            filtered.direction = self.direction
            filtered.startPos = self.startPos
            filtered.energy = self.energy
            filtered.momentum = self.momentum
            GenericFilter()
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


def Pi0MCMask(events : Data, daughters : int = None):
    #?do filtering within function i.e. return object of type Data or mutate events argument?
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
    elif daughters < 0:
        r_mask = nDaughter > abs(daughters)
    else:
        r_mask = nDaughter == daughters
    
    t_mask = ak.num(events.trueParticles.truePhotonMask[events.trueParticles.truePhotonMask], -1) == 2 # exclude pi0 -> e+ + e- + photons

    valid = np.logical_and(r_mask, t_mask) # events which have 2 reco daughters and correct pi0 decay

    null = ak.any(events.recoParticles.direction.x == -999, 1) # exclude events where direction couldn't be calculated

    valid = np.logical_and(valid, np.logical_not(null))
    return valid


def BeamFilteringMask(events : Data):
    hasBeam = events.recoParticles.beam_number != -999
    beamParticle = events.recoParticles.number == events.recoParticles.beam_number
    beamParticleDaughters = events.recoParticles.mother == events.recoParticles.beam_number
    particle_mask = np.logical_or(beamParticle, beamParticleDaughters)
    particle_mask = np.logical_and(particle_mask, hasBeam)
    return particle_mask

@timer
def MCTruth(events : Data, sortEnergy : ak.Array):
    #? move functionality into trueParticleData class and have each value as an @property?
    """Calculate true shower pair quantities.

    Args:
        events (Master.Event): events to process
        sortEnergy (ak.Array): mask to sort shower pairs by true energy

    Returns:
        tuple of ak.Array: calculated quantities
    """
    #* get the primary pi0
    mask_pi0 = np.logical_and(events.trueParticles.number == 1, events.trueParticles.pdg == 111)
    photons = events.trueParticles.truePhotonMask

    #* compute start momentum of dauhters
    p_daughter = events.trueParticles.momentum[photons]
    sum_p = ak.sum(p_daughter, axis=1)
    sum_p = vector.magntiude(sum_p)
    p_daughter_mag = vector.magntiude(p_daughter)
    p_daughter_mag = p_daughter_mag[sortEnergy]

    #* compute true opening angle
    angle = np.arccos(vector.dot(p_daughter[:, 1:], p_daughter[:, :-1]) / (p_daughter_mag[:, 1:] * p_daughter_mag[:, :-1]))

    #* compute invariant mass
    e_daughter = events.trueParticles.energy[photons]
    inv_mass = (2 * e_daughter[:, 1:] * e_daughter[:, :-1] * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    p_pi0 = events.trueParticles.momentum[mask_pi0]
    p_pi0 = vector.magntiude(p_pi0)
    return inv_mass, angle, p_daughter_mag[:, 1:], p_daughter_mag[:, :-1], p_pi0

@timer
def RecoQuantities(events : Data, sortEnergy : ak.Array):
    #? move functionality into recoParticleData class and have each value as an @property?
    """Calculate reconstructed shower pair quantities.

    Args:
        events(Master.Event): events to process
        sortEnergy (ak.Array): mask to sort shower pairs by true energy

    Returns:
        tuple of ak.Array: calculated quantities + array which masks null shower pairs
    """
    sortedPairs = ak.unflatten(events.recoParticles.energy[sortEnergy], 1, 0)
    leading = sortedPairs[:, :, 1:]
    secondary = sortedPairs[:, :, :-1]

    #* opening angle
    direction_pair = ak.unflatten(events.recoParticles.direction[sortEnergy], 1, 0)
    direction_pair_mag = vector.magntiude(direction_pair)
    angle = np.arccos(vector.dot(direction_pair[:, :, 1:], direction_pair[:, :, :-1]) / (direction_pair_mag[:, :, 1:] * direction_pair_mag[:, :, :-1]))

    #* Invariant Mass
    inv_mass = (2 * leading * secondary * (1 - np.cos(angle)))**0.5

    #* pi0 momentum
    pi0_momentum = vector.magntiude(ak.sum(events.recoParticles.momentum, axis=-1))/1000

    null_dir = np.logical_or(direction_pair[:, :, 1:].x == -999, direction_pair[:, :, :-1].x == -999) # mask shower pairs with invalid direction vectors
    null = np.logical_or(leading < 0, secondary < 0) # mask of shower pairs with invalid energy
    
    #* filter null data
    pi0_momentum = np.where(null_dir, -999, pi0_momentum)
    pi0_momentum = np.where(null, -999, pi0_momentum)

    leading = leading/1000
    secondary = secondary/1000

    leading = np.where(null, -999, leading)
    leading = np.where(null_dir, -999, leading)
    secondary = np.where(null, -999, secondary)
    secondary = np.where(null_dir, -999, secondary)

    angle = np.where(null, -999, angle)
    angle = np.where(null_dir, -999, angle)

    inv_mass = inv_mass/1000
    inv_mass = np.where(null, -999, inv_mass)
    inv_mass = np.where(null_dir, -999, inv_mass)

    return inv_mass, angle, leading, secondary, pi0_momentum, ak.unflatten(null, 1, 0)


def FractionalError(reco : ak.Array, true : ak.Array, null : ak.Array):
    """Calcuate fractional error, filter null data and format data for plotting.

    Args:
        reco (ak.Array): reconstructed quantity
        true (ak.Array): true quantity
        null (ak.Array): mask for events without shower pairs, reco direction or energy

    Returns:
        tuple of np.array: flattened numpy array of errors, reco and truth
    """
    true = true[null]
    true = ak.where( ak.num(true, 1) > 0, true, [np.nan]*len(true) )
    reco = ak.flatten(reco, 1)[null]
    print(f"number of reco pairs: {len(reco)}")
    print(f"number of true pairs: {len(true)}")
    error = (reco / true) - 1
    return ak.to_numpy(ak.ravel(error)), ak.to_numpy(ak.ravel(reco)), ak.to_numpy(ak.ravel(true))
