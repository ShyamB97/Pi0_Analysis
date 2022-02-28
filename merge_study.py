"""
Created on: 15/02/2022 17:37

Author: Shyam Bhuller

Description: A script studying pi0 decay geometry and shower merging.
"""
import os
import awkward as ak
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# custom modules
import Plots
import Master
import vector

def AnalyzeReco(events : Master.Event, matched : ak.Array, unmatched : ak.Array):
    """Study relationships between angles and distances between matched and unmatched showers.

    Args:
        events (Master.Event): events to study
        matched (ak.Array): indicies of matched showers
        unmatched (ak.Array): boolean mask of unmatched showers
    """
    matched_reco = events.Filter([matched]).recoParticles # filter reco for matched/unmatched only
    unmatched_reco = events.Filter([unmatched]).recoParticles

    #* calculate separation of matched to unmatched
    separation_0 = vector.dist(unmatched_reco.startPos, matched_reco.startPos[:, 0])
    separation_1 = vector.dist(unmatched_reco.startPos, matched_reco.startPos[:, 1])
    separation = ak.concatenate([separation_0, separation_1], -1)
    minMask_dist = ak.min(separation, -1) == separation # get closest matched shower to matched to study various combinations

    #* same as above but for angular distance
    angle_0 = vector.angle(unmatched_reco.direction, matched_reco.direction[:, 0])
    angle_1 = vector.angle(unmatched_reco.direction, matched_reco.direction[:, 1])
    angle = ak.concatenate([angle_0, angle_1], -1)
    minMask_angle = ak.min(angle, -1) == angle

    #* get various combinations of distances and angles to look at
    min_separation_by_dist = separation[minMask_dist]
    min_separation_by_angle = separation[minMask_angle]
    min_angle_by_dist = angle[minMask_dist]
    min_angle_by_angle = angle[minMask_angle]

    #* ravel for plotting
    separation_0 = ak.ravel(separation_0)
    separation_1 = ak.ravel(separation_1)
    angle_0 = ak.ravel(angle_0)
    angle_1 = ak.ravel(angle_1)
    min_separation_by_dist = ak.ravel(min_separation_by_dist)
    min_separation_by_angle = ak.ravel(min_separation_by_angle)
    min_angle_by_dist = ak.ravel(min_angle_by_dist)
    min_angle_by_angle = ak.ravel(min_angle_by_angle)
    #separation = ak.ravel(vector.dist(matched_reco.startPos[:, 0], matched_reco.startPos[:, 1]))
    #opening_angle = ak.ravel(vector.dist(matched_reco.direction[:, 0], matched_reco.direction[:, 1]))

    #* plots
    directory = outDir + "merging/"
    if save is True: os.makedirs(directory, exist_ok=True)
    
    Plots.PlotHistComparison(separation_0, separation_1[separation_1 < 300], bins, xlabel="Spatial separation between matched showers and shower 2 (cm)", label_1="shower 0", label_2="shower 1", histtype="step")
    if save is True: Plots.Save( "spatial" , directory)
    Plots.PlotHistComparison(angle_0, angle_1, bins, xlabel="Angular separation between matched showers and shower 2 (rad)", label_1="shower 0", label_2="shower 1", histtype="step")
    if save is True: Plots.Save( "angular" , directory)
    
    plt.rcParams["figure.figsize"] = (6.4*2,4.8)
    plt.figure()
    plt.subplot(1, 2, 1)
    _, edges = Plots.PlotHist2D(separation_0, angle_0, bins, xlabel="Spatial separation between shower 0 and shower 2 (cm)", ylabel="Angular separation between shower 0 and shower 2 (rad)", newFigure=False)
    plt.subplot(1, 2, 2)
    Plots.PlotHist2D(separation_1, angle_1, edges, xlabel="Spatial separation between shower 1 and shower 2 (cm)", ylabel="Angular separation between shower 1 and shower 2 (rad)", newFigure=False)
    if save is True: Plots.Save( "spatial_vs_anglular" , directory)
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    min_spatial_l = "Spatial separation between closest matched shower and shower 2 (cm)"
    min_angular_l = "Angular separation between closest shower and shower 2 (cm)"
    merge_dist_l = "merge by distance"
    merge_angle_l = "merge by angle"

    Plots.PlotHistComparison(min_separation_by_dist, min_separation_by_angle, bins, xlabel=min_spatial_l, label_1=merge_dist_l, label_2=merge_angle_l, histtype="step")
    if save is True: Plots.Save("min_spatial" , directory)
    Plots.PlotHistComparison(min_angle_by_dist, min_angle_by_angle, bins, xlabel=min_angular_l, label_1=merge_dist_l, label_2=merge_angle_l, histtype="step")
    if save is True: Plots.Save("min_angle" , directory)

    plt.rcParams["figure.figsize"] = (6.4*2,4.8)
    plt.figure()
    plt.subplot(1, 2, 1)
    _, edges = Plots.PlotHist2D(min_separation_by_dist, min_angle_by_dist, bins, xlabel=min_spatial_l, ylabel=min_angular_l, title=merge_dist_l, newFigure=False)
    plt.subplot(1, 2, 2)
    Plots.PlotHist2D(min_separation_by_angle, min_angle_by_angle, edges, xlabel=min_spatial_l, ylabel=min_angular_l, title=merge_angle_l, newFigure=False)
    if save is True: Plots.Save( "spatial_vs_anglular" , directory)
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def AnalyzeTruth(events : Master.Event, photons : ak.Array):
    """Plot distances of true particles wrt to eachother + the decay vertex to gauge size of pi0 decays

    Args:
        events (Master.Events): events to look at
        photons (ak.Array): mask to get photons produced from pi0 decays
    """
    pi0_vertex = events.Filter(true_filters=[ak.to_list(events.trueParticles.number == 1)]).trueParticles.endPos # decay vertex is where the particle trajectory ends

    photons_true = events.Filter(true_filters=[photons]).trueParticles

    #* distance from each photon end point to the deacy vertex
    dist_to_vertex_0 = vector.dist(pi0_vertex, photons_true.endPos[:, 0]) # photon end points are the start of showers
    dist_to_vertex_1 = vector.dist(pi0_vertex, photons_true.endPos[:, 1])

    #* distance between each photon end point
    true_shower_separation = vector.dist(photons_true.endPos[:, 0], photons_true.endPos[:, 1])

    #* plots
    if save is True: os.makedirs(outDir + "truth/", exist_ok=True)
    Plots.PlotHist(ak.ravel(dist_to_vertex_0), bins, "Distance from photon 0 end point to decay vertex (cm)")
    if save is True: Plots.Save("dist_vertex_0" , outDir + "truth/")
    Plots.PlotHist(ak.ravel(dist_to_vertex_1), bins, "Distance from photon 1 end point to decay vertex (cm)")
    if save is True: Plots.Save("dist_vertex_1" , outDir + "truth/")
    Plots.PlotHist(true_shower_separation, bins, "True photon separation (cm)")
    if save is True: Plots.Save("separation" , outDir + "truth/")


def MakePlots(dist : ak.Array, angle : ak.Array, dist_label : str, angle_label : str, subdirectory : str, title : str = None):
    """Make plots of distance, angle and distance vs angle.

    Args:
        dist (ak.Array): distance between two showers
        angle (ak.Array): angle between two showers
        dist_label (str): distance plot label
        angle_label (str): angle plot label
        subdirectory (str): output subdirectory
    """
    _dir = outDir + subdirectory
    if save is True: os.makedirs(_dir, exist_ok=True)
    Plots.PlotHist(dist, bins, dist_label, title=title)
    if save is True: Plots.Save( "distance" , _dir)
    Plots.PlotHist(angle, bins, angle_label, title=title)
    if save is True: Plots.Save( "angle" , _dir)
    Plots.PlotHist2D(dist, angle, bins, x_range=[0, 150], xlabel=dist_label, ylabel=angle_label, title=title)
    if save is True: Plots.Save( "2D" , _dir)

@Master.timer
def mergeShower(events : Master.Event, matched : ak.Array, unmatched : ak.Array, mergeMethod : int = 1, energyScalarSum : bool = False):
    """Merge shower not matched to MC to the spatially closest matched shower.
       Only works for 3 daughters per event.

    Args:
        events (Master.Event): events to study
        matched (ak.Array): matched shower indicies
        unmatched (ak.Array): boolean mask of unmatched showers
        mergeMethod (int): method 1 merges by closest angular distance, method 2 merges by closest spatial distance
        energyScalarSum (bool): False does a sum of momenta, then magnitude, True does magnitude of momenta, then sum

    Returns:
        Master.Events: events with matched reco showers after merging
    """
    events_matched = events.Filter([matched])
    unmatched_reco = events.Filter([unmatched]).recoParticles # filter reco for matched/unmatched only

    if mergeMethod == 2:
        #* distance from each matched to unmatched
        separation_0 = vector.dist(unmatched_reco.startPos, events_matched.recoParticles.startPos[:, 0])
        separation_1 = vector.dist(unmatched_reco.startPos, events_matched.recoParticles.startPos[:, 1])
        separation = ak.concatenate([separation_0, separation_1], -1)
        mergeMask = ak.min(separation, -1) == separation # get boolean mask to which matched shower to merge to

    if mergeMethod == 1:
        angle_0 = vector.angle(unmatched_reco.direction, events_matched.recoParticles.direction[:, 0])
        angle_1 = vector.angle(unmatched_reco.direction, events_matched.recoParticles.direction[:, 1])
        angle = ak.concatenate([angle_0, angle_1], -1)
        mergeMask = ak.min(angle, -1) == angle

    #* create Array which contains the amount of energy to merge to the showers
    #* will be zero for the shower we don't want to merge to
    momentumToMerge = ak.where(mergeMask, ak.flatten(unmatched_reco.momentum), {"x": 0, "y": 0, "z": 0}) # create array, where if mergeMask is true, then add appropriate energy, else 0
    momentumToMerge = ak.where(events_matched.recoParticles.momentum.x != -999, momentumToMerge, {"x": 0, "y": 0, "z": 0})  # ensure when mergeing we ignore matched showers with undefined energy
    momentumToMerge = ak.where(momentumToMerge.x != -999, momentumToMerge, {"x": 0, "y": 0, "z": 0}) # ensure when mergeing we ignore unmatched showers with undefined energy

    events_matched.recoParticles.momentum = vector.Add(events_matched.recoParticles.momentum, momentumToMerge)

    new_direction = vector.normalize(events_matched.recoParticles.momentum)
    events_matched.recoParticles.direction = ak.where(events_matched.recoParticles.momentum.x != -999, new_direction, {"x": -999, "y": -999, "z": -999})

    if energyScalarSum is True:
        energyToMerge = ak.where(mergeMask, ak.flatten(unmatched_reco.energy), 0) # create array, where if mergeMask is true, then add appropriate energy, else 0
        energyToMerge = ak.where(events_matched.recoParticles.energy != -999, energyToMerge, 0) # ensure when mergeing we ignore matched showers with undefined energy
        energyToMerge = ak.where(energyToMerge != -999, energyToMerge, 0) # ensure when mergeing we ignore unmatched showers with undefined energy
        events_matched.recoParticles.energy = events_matched.recoParticles.energy + energyToMerge # merge energies
    else:
        new_energy = vector.magntiude(events_matched.recoParticles.momentum)
        events_matched.recoParticles.energy = ak.where(events_matched.recoParticles.momentum.x != -999, new_energy, -999)

    return events_matched
 
@Master.timer
def CalculateQuantities(events : Master.Event, photons : ak.Array, names : str):

    mct = Master.MCTruth(events, events.SortByTrueEnergy(), photons)
    rmc = Master.RecoQuantities(events, events.SortByTrueEnergy())

    # keep track of events with no shower pairs
    null = ak.flatten(rmc[-1], -1)
    null = ak.num(null, 1) > 0

    error = []
    reco = []
    true = []
    for i in range(len(names)):
        print(names[i])
        e, r, t = Master.Error(rmc[i], mct[i], null)
        error.append(e)
        reco.append(r)
        true.append(t)

    error = np.nan_to_num(error, nan=-999)
    reco = np.nan_to_num(reco, nan=-999)
    true = np.nan_to_num(true, nan=-999)
    return true, reco, error


def CreateFilteredEvents(events : Master.Event, nDaughters=None):
    valid, photons = Master.Pi0MCFilter(events, nDaughters)

    shower_dir = events.recoParticles.direction[valid]
    print(f"Number of showers events: {ak.num(shower_dir, 0)}")
    photon_dir = vector.normalize(events.trueParticles.momentum)[photons][valid]

    showers, _, selection_mask, angles = events.MatchMC(photon_dir, shower_dir, returnAngles=True)

    reco_filters = [valid, showers, selection_mask]
    true_filters = [valid, selection_mask]

    return events.Filter(reco_filters, true_filters), photons[valid][selection_mask]


def AnalyseQuantities():
    if save is True: os.makedirs(outDir + "reco/", exist_ok=True)
    l_loc = ["right", "right", "left", "right", "left"]
    for j in range(len(names)):
        plt.figure()
        for i in range(len(f_l)):
            data = rs[i][j]
            data = data[data > -900]
            if i == 0:
                _, edges = Plots.PlotHist(data, bins=bins, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
            else:
                Plots.PlotHist(data, bins=edges, xlabel=r_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
        plt.legend(loc=f"upper {l_loc[j]}", fontsize="small")
        if save is True: Plots.Save( names[j] , outDir + "reco/")

    if save is True: os.makedirs(outDir + "fractional_error/", exist_ok=True)
    l_loc = ["left", "left", "left", "right", "left"]
    for j in range(len(names)):
        plt.figure()
        for i in range(len(f_l)):
            data = es[i][j]
            data = data[data > fe_range[0]]
            data = data[data < fe_range[1]]
            if i == 0:
                _, edges = Plots.PlotHist(data, bins=bins, xlabel=e_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
            else:
                Plots.PlotHist(data, bins=edges, xlabel=e_l[j], histtype="step", newFigure=False, label=f_l[i], density=True)
        plt.legend(loc=f"upper {l_loc[j]}", fontsize="small")
        if save is True: Plots.Save( names[j] , outDir + "fractional_error/")

    if save is True: os.makedirs(outDir + "2D/", exist_ok=True)
    plt.rcParams["figure.figsize"] = (6.4*2,4.8*2)
    for j in range(len(names)):
        plt.figure()
        for i in range(len(f_l)):
            plt.subplot(2, 2, i+1)
            if i == 0:
                _, edges = Plots.PlotHist2D(ts[i][j], es[i][j], bins, y_range=fe_range, xlabel=t_l[j], ylabel=e_l[j], title=f_l[i], newFigure=False)
            else:
                Plots.PlotHist2D(ts[i][j], es[i][j], edges, y_range=fe_range, xlabel=t_l[j], ylabel=e_l[j], title=f_l[i], newFigure=False)
        if save is True: Plots.Save( names[j] , outDir + "2D/")
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def Plot2DTest(ind):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.4*2,4.8*2))

    for i in range(len(axes.flat)):
        x = ts[i][ind]
        y = es[i][ind]

        if len(np.unique(x)) == 1:
            print(len(np.unique(x)))
            x_range = [min(x)-0.01, max(x)+0.01]
        else:
            x_range = [min(x), max(x)]
        if i == 0:
            h0, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, fe_range ], density=True)
            #scale_factor = np.max(h0)
            cmap = plt.get_cmap()
            #norm = matplotlib.colors.Normalize(0, 2)
            #norm = matplotlib.colors.Normalize(np.min(h0), np.max(h0))
            h0[h0==0] = np.nan
            h0T = h0.T / h0.T
            im = axes.flat[i].imshow(np.flip(h0T, 0), extent=[x_range[0], x_range[1], fe_range[0], fe_range[1]], norm=matplotlib.colors.LogNorm())#, norm=norm, cmap=cmap)
            fig.colorbar(im, ax=axes.flat[i])
        else:
            h, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], range=[x_range, fe_range], density=True)
            h = h / h0
            h[h==0] = np.nan
            im = axes.flat[i].imshow(np.flip(h.T, 0), extent=[x_range[0], x_range[1], fe_range[0], fe_range[1]], norm=matplotlib.colors.LogNorm())#, norm=norm, cmap=cmap)
            fig.colorbar(im, ax=axes.flat[i])
        axes.flat[i].set_aspect("auto")
        axes.flat[i].set_title(f_l[i])

    fig.add_subplot(1, 1, 1, frame_on=False)
    plt.tight_layout()

    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    # Adding the x-axis and y-axis labels for the bigger plot
    plt.xlabel(t_l[2], fontsize=14)
    plt.ylabel(e_l[2], fontsize=14)


#* user parameters
save = False
outDir = "pi0_0p5GeV_100K/shower_merge/"
bins = 50
s = time.time()
events = Master.Event("ROOTFiles/pi0_0p5GeV_100K_5_7_21.root")
events_2, photons_2 = CreateFilteredEvents(events, 2)
valid, photons = Master.Pi0MCFilter(events, 3)

shower_dir = events.recoParticles.direction[valid]
print(f"Number of showers events: {ak.num(shower_dir, 0)}")
photon_dir = vector.normalize(events.trueParticles.momentum)[photons][valid]

matched, unmatched, selection_mask = events.MatchMC(photon_dir, shower_dir)

events = events.Filter([valid, selection_mask], [valid, selection_mask]) # filter events based on MC matching

# filter masks
matched = matched[selection_mask]
unmatched = unmatched[selection_mask]
photons = photons[valid][selection_mask]

events_merged_a = mergeShower(events, matched, unmatched, 1, False)
events_merged_s = mergeShower(events, matched, unmatched, 2, False)
events_unmerged = events.Filter([matched])


names = ["inv_mass", "angle", "lead_energy", "sub_energy", "pi0_mom"]
t_l = ["True invariant mass (GeV)", "True opening angle (rad)", "True leading shower energy (GeV)", "True secondary shower energy (GeV)", "True $\pi^{0}$ momentum (GeV)"]
e_l = ["Invariant mass fractional error (GeV)", "Opening angle fractional error (rad)", "Leading shower energy fractional error (GeV)", "Secondary shower energy fractional error (GeV)", "$\pi^{0}$ momentum fractional error (GeV)"]
r_l = ["Invariant mass (GeV)", "Opening angle (rad)", "Leading shower energy (GeV)", "Subleading shower energy (GeV)", "$\pi^{0}$ momentum (GeV)"]

t_2, r_2, e_2 = CalculateQuantities(events_2, photons_2, names)
t, r, e = CalculateQuantities(events_unmerged, photons, names)
t_a, r_a, e_a = CalculateQuantities(events_merged_a, photons, names)
t_s, r_s, e_s = CalculateQuantities(events_merged_s, photons, names)


f_l = ["2 showers", "3 showers, unmerged", "3 showers, merge by angle", "3 showers, merge by distance"]
fe_range = [-1, 1]
ts = [t_2, t, t_a, t_s]
rs = [r_2, r, r_a, r_s]
es = [e_2, e, e_a, e_s]

# Plot2DTest(0)
# Plot2DTest(1)
# Plot2DTest(2)
# Plot2DTest(3)
# Plot2DTest(4)

# hists = []
# h0, xedges, yedges = np.histogram2d(ts[0][2], es[0][2], bins=bins, range=[[min(ts[0][2]), max(es[0][2])], fe_range], density=True)
# plt.figure()
# for i in range(len(f_l)):
#     x = ts[i][2]
#     y = es[i][2]
#     x_range = [min(x), max(x)]
#     h, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], range=[x_range, fe_range], density=True)
#     h = np.ravel(np.nan_to_num(h / h0, posinf=0, neginf=0))
#     h = h[h != 0]
#     hists.append(h)

# hists.reverse()
# f_l.reverse()
# for i in range(len(f_l)-1):
#     if i == 0:
#         _, edges = Plots.PlotHist(hists[i], bins=50, xlabel="bin height ratio wrt 2 shower sample", histtype="step", density=False, newFigure=False, label=f_l[i])
#     else:
#         Plots.PlotHist(hists[i], bins=edges, xlabel="bin height ratio wrt 2 shower sample", histtype="step", density=False, newFigure=False, label=f_l[i])

#AnalyseQuantities()
#AnalyzeReco(events, matched, unmatched)
#AnalyzeTruth(events, photons)
print(f'time taken: {(time.time()-s):.4f}')
