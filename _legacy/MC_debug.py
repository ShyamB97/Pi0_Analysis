#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:04:42 2021

@author: sb16165
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Master
from Master import Unwrap, Vector3, Conditional
from Plots import PlotHist, PlotHist2D
import SelectionQuantities


particle_dict = { "p": 2212, "$\pi^{+}$": 211, "$\pi^{-}$": -211, "$e^{+}$": -11, 
                  "$e^{-}$": 11, "$\mu^{+}$": -13, "$\mu^{-}$": 13, "$\gamma$": 22, 
                  "$K^{+}$": 321, "$K^{-}$": -321, r'$\nu_{\mu}$': 14, r'$\bar{\nu}_{\mu}$': -14,
                  r'$\nu_{e}$': 12, r'$\bar{\nu}_{e}$': -12, "$\pi^{0}$": 111, "n": 2112,
                  }


def PlotParticleTypes(pdg):
    """
    Plot particle PDG's in MC truth list
    """
    unique, amount = np.unique(pdg, return_counts=True)
    unique = list(unique)
    unique = [str(i) for i in unique]
    
    nucleons = 0
    particles = []
    n_particles = []
    for i in range(len(unique)):
        if len(unique[i]) == 10:
            nucleons += amount[i]
        else:
            particles.append(list(particle_dict.keys())[list(particle_dict.values()).index(int(unique[i]))])
            n_particles.append(amount[i])
    
    if nucleons > 0:
        particles.append("nucleons")
        n_particles.append(nucleons)
    
    plt.bar(particles, n_particles)
    plt.xlabel("particle type")
    plt.ylabel("count")
    #plt.yscale("log")
    plt.xticks(rotation=20)
    plt.tight_layout()


def CalculateMass():
    """
    Calculate opening angles and inverant masses of shower pairs in MC truth
    """
    p = (p_x**2 + p_y**2 + p_z**2)**0.5
    dir_x = p_x / p
    dir_y = p_y / p
    dir_z = p_z / p

    pion_ID_mask = Master.SelectionMask()
    pion_ID_mask.InitiliseMask(pdg)
    pion_ID_mask.CutMask(pdg, 111, Conditional.EQUAL)
    pion_IDs = pion_ID_mask.Apply(IDs)
    pion_mass = pion_ID_mask.Apply(mass)

    opening_angles_by_P = []
    opening_angles_by_mass = []
    inv_masses = []
    residuals = []
    for i in range(len(pion_IDs)):
        pion_IDs_evt = pion_IDs[i]
        evt_opening_angles_by_P = []
        evt_opening_angles_by_mass = []
        evt_inv_mass = []
        evt_residual = []
        for j in range(len(pion_IDs_evt)):
            mask = Master.SelectionMask()
            mask.InitiliseMask(pdg)
            mask.CutMask(mother, pion_IDs_evt[j], Conditional.EQUAL)
            x = Unwrap(mask.Apply(dir_x))

            if(len(x) != 2):
                print("not two body decay!")
                continue
            y = Unwrap(mask.Apply(dir_y))
            z = Unwrap(mask.Apply(dir_z))
            e = Unwrap(mask.Apply(energy))
        
            opening_angle_by_P = SelectionQuantities.Angle(x[0], y[0], z[0], x[1], y[1], z[1])
            evt_opening_angles_by_P.append(opening_angle_by_P)

            opening_angle_by_mass = OpeningAngleByMass(mass, e[0], e[1])
            evt_opening_angles_by_mass.append(opening_angle_by_mass)

            inv_mass = np.sqrt(2 * e[0] * e[1] * (1 - np.cos(opening_angle_by_P)) )
            evt_inv_mass.append(inv_mass)

            residual = (inv_mass - pion_mass[i][j]) / pion_mass[i][j]
            evt_residual.append(residual)

        opening_angles_by_P.append(evt_opening_angles_by_P)
        opening_angles_by_mass.append(evt_opening_angles_by_mass)
        inv_masses.append(evt_inv_mass)
        residuals.append(evt_residual)

    opening_angles_by_P = Unwrap(opening_angles_by_P)
    opening_angles_by_mass = Unwrap(opening_angles_by_mass)
    inv_masses = Unwrap(inv_masses)
    residuals = Unwrap(residuals)


    plt.figure(4, (12, 10))
    plt.subplot(221)
    PlotHist(opening_angles_by_P, xlabel="opening angle (rad)")
    plt.subplot(222)
    PlotHist(inv_masses, xlabel="pion mass (GeV)")
    plt.subplot(223)
    PlotHist(residuals, xlabel="$(m_{true} - m_{calc} )/ m_{true}$")
    plt.subplot(221)
    PlotHist(opening_angles_by_P, xlabel="opening angle (rad)")    

    print("pion mass average: " + str(np.mean(inv_mass)))
    print("pion mass standard deviation: " + str(np.std(inv_mass)))


def OpeningAngleByMass(pi0_mass, E_1, E_2):
    cos_angle = 1 - ( pi0_mass**2 / (2 * E_1 * E_2) )
    return np.arccos(cos_angle)



file = "ROOTFiles/pi0Test_output_0p5GeV_allPFP.root"
#file = "ROOTFiles/pi0Test_output_PDSPProd4_MC_1GeV_SCE_DataDriven_reco_100_10_05_21.root"
pdg = Master.GetData(file, "g4_Pdg")
IDs = Master.GetData(file, "g4_num")
evd_ID = Master.GetData(file, "EventID")
mother = Master.GetData(file, "g4_mother")
mass = Master.GetData(file, "g4_mass")
energy = Master.GetData(file, "g4_startE")

start_posX = Master.GetData(file, "g4_startX")
start_posY = Master.GetData(file, "g4_startY")
start_posZ = Master.GetData(file, "g4_startZ")

end_posX = Master.GetData(file, "g4_endX")
end_posY = Master.GetData(file, "g4_endY")
end_posZ = Master.GetData(file, "g4_endZ")

p_x = Master.GetData(file, "g4_pX")
p_y = Master.GetData(file, "g4_pY")
p_z = Master.GetData(file, "g4_pZ")

p = Vector3(p_x, p_y, p_z)


#pdg = Unwrap(pdg)
#IDs = Unwrap(IDs)
#mother = Unwrap(mother)
#mass = Unwrap(mass)


#PlotParticleTypes(Unwrap(pdg))

"""
def PionValues(mc_value, mc_pdg):
    pion_energy = [mc_value[i][mc_pdg[i] == 111] for i in range(len(mc_value))]
    return pion_energy


pion_momentum = Unwrap(PionValues(p.Magnitude(), pdg))
pion_energy = Unwrap(PionValues(energy, pdg))
pion_mass = Unwrap(PionValues(mass, pdg))

calculated_mass = ( pion_energy**2 - pion_momentum**2 )**0.5

PlotHist(pion_energy[pion_energy < 1 + pion_mass], density=False)
"""

# study pions by momentum
pion_p = p.Magnitude()
#p_unique = np.unique(pion_p)


photon_opening_angles = []
opening_angles_by_mass = []
photon_energies = []
pion_masses = []

higher_order = 0

for i in range(len(pion_p)):
    
    photon_p = p[i][pdg[i] == 22]
    
    if len(photon_p.x) != 2:
        print("two photon object only")
        photon_opening_angles.append(-999)
        photon_energies.append([-999])
        pion_masses.append(-999)
        higher_order += 1
        continue
    
    opening_angle_by_P = SelectionQuantities.Angle( photon_p[0], photon_p[1] )
    
    energies = [vec.Magnitude() for vec in photon_p]
    
    opening_angle_by_mass = OpeningAngleByMass(mass[i][pdg[i] == 111], energies[0], energies[1])
    pion_mass = np.sqrt(2 * energies[0] * energies[1] * (1 - np.cos(opening_angle_by_P)) )
    
    photon_opening_angles.append(opening_angle_by_P)
    opening_angles_by_mass.append(opening_angle_by_mass)
    photon_energies.append(energies)
    pion_masses.append(pion_mass)

photon_energies = np.array(photon_energies, dtype=object)
photon_opening_angles = np.array(photon_opening_angles, dtype=object)
pion_masses = np.array(pion_masses, dtype=object)

opening_angles_by_mass = np.array(opening_angles_by_mass, dtype=object)

#print(evd_ID)
#print(pdg)


plt.figure(4, (12, 10))
plt.subplot(221)
PlotHist(photon_opening_angles[photon_opening_angles != -999], xlabel="opening angle by momentum (rad)")

plt.subplot(222)
PlotHist(opening_angles_by_mass, xlabel="opening angle by mass (rad)")


plt.subplot(223)
#PlotHist(pion_masses[pion_masses != -999], xlabel="pion mass (GeV)")
plt.plot(pion_masses[pion_masses != -999])
plt.xlabel("event number")
plt.ylabel("pion mass (GeV)")
plt.tight_layout()

plt.subplot(224)

photon_energies = Unwrap(photon_energies)
photon_energies = photon_energies[photon_energies != -999]

PlotHist(photon_energies, xlabel="photon energies (GeV)")
