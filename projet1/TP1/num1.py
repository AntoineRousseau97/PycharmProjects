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
I_bone = 91.9 # eV

#calculation of electron density of materials
def calc_ne(list_atom_no, list_fraction, list_molar_mass, rho):
    i = 0 
    ne = 0
    for atom_no in list_atom_no:
        nb_electron = atom_no*(list_fraction[i]/list_molar_mass[i])*avogadro
        ne += rho*nb_electron
        i += 1
    return ne

#Calculation of lorentz factor and stopping power of proton in materials
def f(T, ne, I):
    gamma = T/(mp*c**2) + 1
    beta = np.sqrt(1- 1/gamma**2)
    Te_max = (2 * me * c**2 * (gamma**2 - 1)) / (1 + 2 * gamma * (me / mp) + (me / mp)**2)
    return 2*np.pi*(re**2)*me*(c**2)*ne*(1/(beta**2))*(np.log(2*me*(c**2)*(beta**2)*(gamma**2)*Te_max/(I**2))-2*(beta**2))

#Plotting
def plot_Scol(ne, I, figname):
    T = np.linspace(3e6,250e6, 1000000)
    Scol = f(T, ne, I)
    
    g = plt.figure()
    plt.plot(T, Scol, "k")
    plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("Énergie (eV)")
    plt.ylabel(r"Pouvoir d’arrêt collisionnel ($MeV \cdot cm^2 / g$)")
    plt.show()
    g.savefig(figname, dpi = 600, bbox_inches = "tight")

#Calling the electron density function
ne_H20 = calc_ne(list_atom_H20, list_fraction_H20, list_molar_mass_H20, rho_water)
ne_bone = calc_ne(list_atom_bone, list_fraction_bone, list_molar_mass_bone, rho_bone)
    
#Printing electron density
print(ne_H20)
print(ne_bone)  


#Calling the plotting function
plot_Scol(ne_H20, I_H20, "Scol_H20.png")
plot_Scol(ne_bone, I_bone, "Scol_bone.png")



