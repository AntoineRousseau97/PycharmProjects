import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#function that defines V(x)
def V(x):
    return 4*((1/x)**12-(1/x)**6)

#function that defines the function to integrate i.e. (E-v(x))^2
def g(E,x):
    v = V(x)
    return (E - v)**(1/2), v

#function defining the turning points
def x_plusminus(E):
    x_minus = 2**(1/6) * (np.sqrt(E+1)/E-1/E)**(1/6)
    x_plus = 2**(1/6) * (-((np.sqrt(E+1)+1)/E))**(1/6)
    return x_plus, x_minus

#function that calculates the integral
def f(E, gamma, n):
    x_plus, x_minus = x_plusminus(E) 
    integral, error = integrate.quad(lambda x :g(E,x)[0], x_minus, x_plus)
    return gamma * integral - (n+1/2)*np.pi

#function that computes epsilon with gamma and n as an input
def epsilon(gamma, n, error):
    x1 = -1
    x2 = -0.0000000001
    fx1 = f(x1, gamma, n)
    fx2 = f(x2, gamma, n)
    N = 0
    
    while np.abs(x2-x1) > error:
        x_prime = 0.5*(x1 + x2)
        new_f = f(x_prime, gamma, n)
        
        if new_f == 0: 
            break
        elif new_f*fx1 > 0:
             x1 = x_prime
             fx1 = new_f
        elif new_f*fx2 > 0:
            x2 = x_prime
            fx2 = new_f

        N += 1
            
    return 0.5*(x1+x2), N

E_list = []
V_list = []
N_list = []
n_list = list(range(20))
x_list = np.linspace(1, 1.8, 100)
x_plus_list = []
x_minus_list = []

#for loop that gets the energy for n < 20
for i in n_list:
    E, N = epsilon(150, i, 1e-8)
    x_plus, x_minus = x_plusminus(E)
    E_list += [E]
    x_plus_list += [x_plus]
    x_minus_list += [x_minus]
    
for i in x_list:
    V_list += [V(i)]
    
#Plotting of V(x) that shows the computed energies and turning points
plt.plot(x_list, V_list, "k")
plt.xlim(0.9, 1.8)

for i in range(len(E_list)):
    plt.hlines(E_list[i], x_minus_list[i], x_plus_list[i], linestyles = '--')

    if i % 2 == 0:
        string = "E" + "$_{" + str(i + 1) + "}$"
        plt.text(x_plus_list[i] + 0.025, E_list[i] - 0.05, string)
plt.plot(x_minus_list, E_list, "ko")
plt.plot(x_plus_list, E_list, "ko") 
plt.xlabel("Position (-)")
plt.ylabel("Énergie $\epsilon_{n}$ (-) ")
plt.show()