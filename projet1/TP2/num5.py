import numpy as np
import matplotlib.pyplot as plt

R0_list = np.linspace(0, 5.7, 58)
p_list = []
p_initial = 1

def RHS(p,R0):
    next_p = 1-np.exp(-R0*p) 
    return next_p

def relaxation(R0, target = 1e-8):
    p = p_initial
    p_old = 1000
    
    while np.abs(p-p_old) > target:        
        p_old = p
        p = RHS(p, R0)    
    
    return p

for R0 in R0_list:
    p_list += [relaxation(R0)]
    
plt.plot(R0_list, p_list, "ko");
plt.xlabel('$R_{0}$')
plt.ylabel('p évalué')
plt.show()



