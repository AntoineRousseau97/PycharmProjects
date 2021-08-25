import numpy as np
from scipy.stats import moyal
import matplotlib.pyplot as plt
from scipy import integrate
import timeit

Ti = 150e6
rho_H2O = 1000
rho_bone = 1850
I_H2O = 75 #eV
I_bone = 91.9 #eV
c = 299792458
me = 9.1094E-31*6.242e18 #eV
mp = 1.6727E-27*6.242e18 #eV 
re = 2.8179403227e-15 #m

ne_H2O = 3.385161894888597e+29
ne_bone = 5.906570630843504e+29

energy_list = []

for i in range(10000):
    energy_list += [moyal.rvs(150, 4)]
    
#definition of function
def f(T, ne, I, rho):
    gamma = T/(mp*c**2) + 1
    beta = np.sqrt(1- 1/gamma**2)
    Te_max = (2 * me * c**2 * (gamma**2 - 1)) / (1 + 2 * gamma * (me / mp) + (me / mp)**2)
    Scol = 2*np.pi*(re**2)*me*(c**2)*ne*(1/(beta**2))*(np.log(2*me*(c**2)*(beta**2)*(gamma**2)*Te_max/(I**2))-2*(beta**2))
        
    return rho/Scol

#definition of function's derivative
def f_prime(T, ne, I):
    gamma = T/(mp*c**2) + 1
    a = 2*me
    b = 1+(me/mp)**2
    d = me/mp
    U = 2*np.pi*(re**2)*me*(c**2)*ne
    k = (a/I)**2
    
    return 1/(U*gamma*(gamma*(4*b*gamma+3*d*(gamma**2))-2*(b+d*gamma)*np.log((k*(c**2)*((gamma**2)-1)**2)/(b+d*gamma))))/((((gamma**2)-1)**2)*(b+d*gamma)*(mp*(c**2)))

#function to find if the calculated error is below target precision 
def test_error(Ri, Ri_old, m, target, found):
    err = (1/((4**m)-1))*np.abs(Ri-Ri_old)
    if found == False:
        if err <= Ri*target:
            found = True
            
    return [err, found]

def trapeze(N, a, b, ne, I, rho): 
    h = (b-a)/N
    s = 0.5*f(a, ne, I, rho) + 0.5*f(b, ne, I, rho)
    
    for k in range(1,N):
        s += f(a+k*h, ne, I, rho)

    theo_error = (1/12)*(h**2)*(f_prime(a, ne, I) - f_prime(b, ne, I))
        
    return h*s, theo_error
    
def simpson(N, a, b, ne, I, rho):
    h = (b-a)/N
    s1 = 0.0
    for k in range(1,N,2):
        s1 += f(a+k*h, ne, I, rho)
    s2 = 0.0
    for k in range(2,N,2):
        s2 += f(a+k*h, ne, I, rho)
        
    theo_error = 0 #could be implemented in the future
        
    return (f(a, ne, I, rho)+f(b, ne, I, rho)+4.0*s1+2.0*s2)*h/3.0, theo_error

#Algorithm calculating the integral with trapeze and simpson methods
def algo(Ti, material, N_i, target, int_type):

    if material == "water":
        ne = ne_H2O
        I = I_H2O
        rho = rho_H2O
        
    elif material == "bone":
        ne = ne_bone
        I = I_bone
        rho = rho_bone
        
    if int_type == "trapeze":
        integral, theo_error = trapeze(N_i, 3e6, Ti, ne, I, rho)
        R = [integral]
        
    elif int_type == "simpson" :
        integral, theo_error = simpson(N_i, 3e6, Ti, ne, I, rho)
        R = [integral]
        
    i = 1
    found = False
    
    while found == False:
        N_i = N_i * 2
        
        if int_type == "trapeze":
            integral, theo_error = trapeze(N_i, 3e6, Ti, ne, I, rho)
            R += [integral]
            
        elif int_type == "simpson" :
            integral, theo_error = simpson(N_i, 3e6, Ti, ne, I, rho)
            R += [integral]
        
        Ri = R[-1]
        Ri_old = R[-1-i]
        error = test_error(Ri, Ri_old, 1, target, found) #calculating error
        found = error[1] #confirming if error < target
        
        
        #loop for calculation of Ri,m integrals (m != 1)
        for m in range(2,i+2):
            
            if found == False:
                Ri = R[-1] #integral Ri,m
                Ri_old = R[-1-i] #integral Ri-1,m
                
                error = test_error(Ri, Ri_old, m, target, found) #calculating error
                found = error[1] #confirming if error < target

                R += [Ri + error[0]] # calculating integral Ri,m+1
        
        i += 1    
    
    return Ri, error[0], theo_error, N_i

#function to evaluate the speed of algorithms
def time_algorithm(int_type, target):
    protons = []
    start_time = timeit.default_timer()
    
    n = 0
    for i in energy_list:
        if int_type == "quad":
            integral, error = integrate.quad(lambda x :f(x, ne_H2O, I_H2O, rho_H2O), 3e6, i*1e6)
            protons += [integral]
        else:
            integral, error, theo_error, Nmax = algo(i*1e6, "water", 1, target, int_type)
            protons += [integral]
        
        end_time = timeit.default_timer()
        n += 1
            
    return protons, (end_time-start_time)
    
scope_list = []

#============================================================================
# #Calculating the scope of proton in bones and water with both integration methods
# for i in ["trapeze", "simpson"]:
#     for j in ["water", "bone"]:
#         scope, error, theo_error, Nmax = algo(Ti, j, 1, 1e-16, i)
#         print(i + ", " +  j + " :")
#         print("result = " + str(scope) + " cm")
#         print("error = " + str(error))
#         print("Nmax = " + str(Nmax))
#         print("theo_error = " + str(theo_error) + "\n")

#============================================================================
#Calculating speed of algorithms and comparing with scipy.integrate.quad
proton_scope, time = time_algorithm("trapeze", 1e-8)
scope_list += [proton_scope]
proton_per_time = np.round(10000/time, 2)
print("trapeze : " + str(proton_per_time) + " protons/s")

proton_scope, time = time_algorithm("simpson", 1e-8)
scope_list += [proton_scope]
proton_per_time = np.round(10000/time, 2)
print("simpson : " + str(proton_per_time) + " protons/s")

proton_scope, time = time_algorithm("quad", 1e-8)
scope_list += [proton_scope]
proton_per_time = np.round(10000/time, 2)
print("quad : " + str(proton_per_time) + " protons/s")

plt.hist(scope_list[0], 50, color = "k")
plt.xlabel("Énergie (MeV)")
plt.ylabel("Nombre de protons")
plt.show()
                   
#============================================================================
# Plotting the results according to N 
# energy_trapeze = []
# energy_simpson = []
# N_simpson = []

# N_trapeze = np.linspace(200, 1892, 1693)
# N_simpson = [2*x for x in range(100,257)]

# for N in range(100, 1893):
#     integral, theo_error = trapeze(N, 3e6, Ti, ne_H2O, I_H2O, rho_H2O)
#     energy_trapeze += [integral]
        
# for N in N_simpson:

#     integral, theo_error = simpson(N, 3e6, Ti, ne_H2O, I_H2O, rho_H2O)
#     energy_simpson += [integral]
    
    
# plt.plot(N_trapeze, energy_trapeze, label = "Trapèze", color = "k")
# plt.xlabel("Nombre de tranches (N)")
# plt.ylabel("Portée (cm)")
# plt.show()
# plt.plot(N_simpson, energy_simpson, label = "Simpson", color = "k")
# plt.xlabel("Nombre de tranches (N)")
# plt.ylabel("Portée (cm)")
# plt.show()
#============================================================================   

                       
    