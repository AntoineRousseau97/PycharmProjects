import numpy as np
import matplotlib.pyplot as plt

#R0_list = np.linspace(0, 5.7, 58).
R0 = 2
p_list = []
n_list = []
p_initial = 1

def RHS(p,R0):
    next_p = 1-np.exp(-R0*p) 
    return next_p

def relaxation(R0, target = 1e-8):
    p = p_initial
    p_old = 1000
    n = 0
    
    while np.abs(p-p_old) > target:        
        p_old = p
        p = RHS(p, R0)    
        n += 1
    
    return p,n

def relaxation_accel(R0, w, target = 1e-6):
    p = p_initial
    p_old = 1000
    n = 0
    
    while np.abs(p-p_old) > target:        
        p_old = p
        p = (1+w)*RHS(p, R0) - w*p
        n += 1
    
    return p, n

# for R0 in R0_list:
#     relax = relaxation(R0)
#     p_list += [relax[0]]
    
w_list = np.linspace(0.5, 0.8, 400)

for w in w_list:
    n_list += [relaxation_accel(R0, w)[1]]

print(relaxation(R0)[1])
print(relaxation_accel(R0, 0.685)[1])
    
plt.plot(w_list, n_list, "k");
plt.xlabel('$R_{0}$')
plt.ylabel('p évalué')
plt.show()



