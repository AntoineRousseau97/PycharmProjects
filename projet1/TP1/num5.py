import numpy as np
import matplotlib.pyplot as plt

Ti = 150e6
rho_H2O = 1000
rho_bone = 1.85 
I_H20 = 75 #eV
I_bone = 91.9 #eV
c = 299792458
me = 9.1094E-31*6.242e18 #eV
mp = 1.6727E-27*6.242e18 #eV 
re = 2.8179403227e-15 #m

ne_H20 = 3.385161894888597e+29
ne_bone = 5.906570630843504e+29

def f(T, ne, I, rho):
    gamma = T/(mp*c**2) + 1
    beta = np.sqrt(1- 1/gamma**2)
    Te_max = (2 * me * c**2 * (gamma**2 - 1)) / (1 + 2 * gamma * (me / mp) + (me / mp)**2)
    Scol = 2*np.pi*(re**2)*me*(c**2)*ne*(1/(beta**2))*(np.log(2*me*(c**2)*(beta**2)*(gamma**2)*Te_max/(I**2))-2*(beta**2))
        
    return rho/Scol

def error(Ri, Ri_old, m, found):
    #err = np.abs((1/(4**m -1)) * (Ri - Ri_old))
    err = (1/((4**m)-1))*np.abs(Ri-Ri_old)
    if found == False:
        if err <= Ri*1e-16:
            found = True
            
    return [err, found]

def trapeze(N, a, b, ne, I, rho): 
    h = (b-a)/N
    #méthode du trapèze
    s = 0.5*f(a, ne, I, rho) + 0.5*f(b, ne, I, rho)
    
    for k in range(1,N):
        s += f(a+k*h, ne, I, rho)
        
    return h*s
    
def simpson(N, a, b, ne, I, rho):
    h = (b-a)/N
    #méthode de Simpson
    s1 = 0.0
    for k in range(1,N,2):
        s1 += f(a+k*h, ne, I, rho)
    s2 = 0.0
    for k in range(2,N,2):
        s2 += f(a+k*h, ne, I, rho)
        
    return (f(a, ne, I, rho)+f(b, ne, I, rho)+4.0*s1+2.0*s2)*h/3.0



#===========================================
N_i = 1
R = [trapeze(N_i, 3e6, Ti, ne_H20, I_H20, rho_H2O)]
i = 1
found = False


while found == False:
    N_i = N_i * 2
    R += [trapeze(N_i, 3e6, Ti, ne_H20, I_H20, rho_H2O)] #integral Ri,1
    Ri = R[-1]
    Ri_old = R[-1-i]
    error_func = error(Ri, Ri_old, 1, found) #calculating error
    found = error_func[1] #confirming if error < target
    
    #print("(i,m) = (" + str(i+1) + "," + "1)")
    
    #loop for calculation of Ri,m integrals (m != 1)
    for m in range(2,i+2):
        
        if found == False:
            Ri = R[-1] #integral Ri,m
            Ri_old = R[-1-i] #integral Ri-1,m
            
            error_func = error(Ri, Ri_old, m, found) #calculating error
            found = error_func[1] #confirming if error < target
        
            #print("(i,m) = (" + str(i+1) + "," + str(m) + ")")
            
            R += [Ri + error_func[0]] # calculating integral Ri,m+1
        
            if found == True:
                print("Trapeze")
                print("R " + str(i+1) + " " + str(m))
                print("N = " + str(N_i))
                print("error = " + str(error_func[0]))
                #print(Ri_old)
                print(Ri)
                print(" ")
    
    i += 1
#=======================================================================    
N_i = 1
R = [simpson(N_i, 3e6, Ti, ne_H20, I_H20, rho_H2O)]
i = 1
found = False


while found == False:
    N_i = N_i*2
    R += [simpson(N_i, 3e6, Ti, ne_H20, I_H20, rho_H2O)] #integral Ri,1
    Ri = R[-1]
    Ri_old = R[-1-i]
    #Ri_old = R[-1-i]
    
    error_func = error(Ri, Ri_old, 1, found) #calculating error
    #print(error_func[0])
    found = error_func[1]#confirming if error < target
    
    #print("(i,m) = (" + str(i+1) + "," + "1)")
    
    #loop for calculation of Ri,m integrals (m != 1)
    for m in range(2,i+2):
        
        if found == False:
            Ri = R[-1] #integral Ri,m-1
            Ri_old = R[-1-i] #integral Ri-1,m-1
            
            error_func = error(Ri, Ri_old, m, found) #calculating error
            found = error_func[1] #confirming if error < target
        
            #print("(i,m) = (" + str(i+1) + "," + str(m) + ")")
            
            R += [Ri + error_func[0]] # calculating integral Ri,m
            if found == True:
                print("Simpson")
                print("R " + str(i+1) + " " + str(m))
                print("N = " + str(N_i))
                print("error = " + str(error_func[0]))
                #print(Ri_old)
                print(Ri)
                print(" ")
    
    i += 1