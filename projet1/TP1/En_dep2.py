import matplotlib.pyplot as plt
import numpy as np

# Distance entre la surface et le mélanome [m]
d = 0.04

# Fonction qui permet d'integrer avec la méthode des trapèzes
def integ(g, b, a, N):
    h = (b-a)/N
    d = 0.5*g(b)+0.5*g(a)
    for i in range(1,N):
        d += g(a+i*h)

    return d*h
# Constantes physiques utiles
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

pi = np.pi


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


g = lambda T: 1/(Scol(T, calc_ne(list_atom_H20, list_fraction_H20, list_molar_mass_H20, rho_water), I_H20))

# Courbe de la distance en selon l'énergie

n = 1000
E_i = np.linspace(200*10**6, 3*10**6, n)# liste des énergies initiales des protons
E_f = 3*(10**6)                         # énergie finale des protons
n = 10000                               # nombre de segments pour l'intégration
R_CSDA = []                             # liste de la portée des protons
for i in E_i:
    R_CSDA.append(integ(g, i, E_f, n))


# interpolation de l'énergie selon la portée
interpol = np.poly1d(np.polyfit(R_CSDA, E_i, 9))
val_fit = interpol(d)

# Graphique de vérification de l'interpolation
plt.plot(R_CSDA, E_i/(10**6))
plt.plot(R_CSDA, interpol(R_CSDA)/(10**6), label='interpolation')
plt.xlabel("Distance à parcourir [m]")
plt.ylabel("Énergie cinétique d'un photon [MeV]")
plt.yscale('log')
plt.legend(loc='best')
plt.show()

# fonction qui permet de réduire l'erreur à 0.01 MeV
def optim(E_optim, n_iter, k = 1000):
    sup = integ(g, E_optim + k, E_f, n)
    inf = integ(g, E_optim - k, E_f, n)

    if sup < d:
        optim(E_optim + k, n_iter + 1)
    if inf > d:
        optim(E_optim - k, n_iter + 1)
    if sup >= d >= inf:
        print(E_optim/10**6, 'MeV')

optim(val_fit, 0)