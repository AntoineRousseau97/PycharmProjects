import numpy as np
import matplotlib.pyplot as plt

#Definition of known costants
avogadro = 6.022e23 #particle/mol
c = 299792458 #m/s
me = 9.1094E-31*6.242e18 #eV
mp = 1.6727E-27*6.242e18 #eV
re = 2.8179403227e-15 #m

#Definition of materials (water and compact bone)

#Water
list_atom_H20 = [1, 8] #atomic number of atoms in H20
list_fraction_H20 = [0.11894, 0.888194] #massic fraction of constituents
list_molar_mass_H20_g = [1.0079, 15.999] #g/mol (molar mass in g)
list_molar_mass_H20 = [x / 1000 for x in list_molar_mass_H20_g] #kg/mol (in kg)
rho_water = 1000 #kg/m^3 (density)
I_H20 = 75 #eV

#Compact bone
list_atom_bone = [1, 6, 7, 8, 12, 15, 16, 20]
list_fraction_bone = [0.063984, 0.278, 0.027, 0.410016, 0.002, 0.07, 0.002, 0.147]
list_molar_mass_bone_g = [1.007, 12.01, 14.007, 15.999, 24.305, 30.974, 32.06, 40.078] #g/mol
list_molar_mass_bone = [x / 1000 for x in list_molar_mass_bone_g] #kg/mol
rho_bone = 1850 #kg/m^3
I_bone = 91.9 #

#calculation of electron density of materials
def calc_ne(list_atom_no, list_fraction, list_molar_mass, rho):
    i = 0
    ne = 0
    for atom_no in list_atom_no:
        nb_electron = atom_no*(list_fraction[i]/list_molar_mass[i])*avogadro
        ne += rho*nb_electron
        i += 1
    return ne


def Scol(T, ne, I):
    gamma = T/(mp*c**2) + 1
    beta = np.sqrt(1- 1/gamma**2)
    Te_max = (2 * me * c**2 * (gamma**2 - 1)) / (1 + 2 * gamma * (me / mp) + (me / mp)**2)

    return 2*np.pi*(re**2)*me*(c**2)*ne*(1/(beta**2))*(np.log(2*me*(c**2)*(beta**2)*(gamma**2)*Te_max/(I**2))-2*(beta**2))


g_eau = lambda T: 1/(Scol(T, calc_ne(list_atom_H20, list_fraction_H20, list_molar_mass_H20, rho_water), I_H20))  # s: distance parcourue par un proton, T_i et T_f: energies cinétiques respectives avant et apres que le proton ait subi l'atténuation d'épaisseur s du matériau

g_bone = lambda T: 1 / (Scol(T, calc_ne(list_atom_bone, list_fraction_bone, list_molar_mass_bone, rho_bone), I_bone))
#def g_bone(T):
 #   return 1 / (Scol(T, calc_ne(list_atom_bone, list_fraction_bone, list_molar_mass_bone, rho_bone), I_bone))

def integ(g, b, a, N):
    h = (b-a)/N
    d = 0.5*g(b)+0.5*g(a)
    for i in range(1,N):
        d += g(a+i*h)

    return d*h

T_i = 150*(10**6)
T_f = 3*(10**6)
n = 1000

E_p = np.linspace(T_i, T_f, 100)

dist_eau = 0
dist_bone = 0

lst_s_eau = [dist_eau]
lst_s_bone = [dist_bone]

E_dep_eau = []
E_dep_bone = []

for i in range(len(E_p)-1):
    s_eau = integ(g_eau, E_p[i], E_p[i+1], n)
    dist_eau += s_eau
    lst_s_eau.append(dist_eau)
    E_dep_eau.append((E_p[i]-E_p[i+1])/s_eau/(10**6))

    s_bone = integ(g_bone, E_p[i], E_p[i+1], n)
    dist_bone += s_bone
    lst_s_bone.append(dist_bone)
    E_dep_bone.append((E_p[i]-E_p[i+1])/s_bone/(10**6))

# Graphique de l'énergie d'un proton selon sa distance parcourue

E_p = (E_p/(10**6))
plt.plot(lst_s_eau, E_p, label='Distance finale eau = ' + str(np.round(dist_eau, 6)))
plt.plot(lst_s_bone, E_p, label='Distance finale os = ' + str(np.round(dist_bone,6)))
plt.xlabel("Distance parcourue par un proton à 150 MeV [m]")
plt.ylabel("Énergie cinétique d'un proton [MeV]")
plt.title("Énergie cinétique d'un protons selon \n sa profondeur", fontsize=12)
plt.legend(loc='best')
plt.show()

# Graphique de l'énergie déposée par un proton par mètre selon sa distance parcourue
plt.plot(lst_s_eau[1:], E_dep_eau, label='eau')
plt.plot(lst_s_bone[1:], E_dep_bone, label='os')
plt.yscale('log')
plt.xlabel("Distance parcourue par un proton [m]")
plt.ylabel("Énergie déposée par mètre [MeV/m]")
plt.title("Énergie déposée par un proton par mètre en \n fonction de sa profondeur dans le milieu", fontsize=12)
plt.legend(loc='best')
plt.show()

# Portée CSDA/rho des protons [m]
R_CSDA_eau = integ(g_eau, T_i, T_f, n)
R_CSDA_bone = integ(g_bone, T_i, T_f, n)

print("R_CSDA_eau = " + str(R_CSDA_eau) + "m", "R_CSDA_bone = " + str(R_CSDA_bone) + "m")