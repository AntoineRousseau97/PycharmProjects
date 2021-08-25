from astropy.constants import G, M_sun, M_earth #(m3 / (kg s2) and kg)
from numpy import array, sqrt, arange
import matplotlib.pyplot as plt
   
#function that calculates the radius with x and y         
def r_func(x,y):
    return sqrt(x**2 + y**2)
    
def f(r,t):
    x = r[0]
    y = r[1]
    v_x = r[2]
    v_y= r[3]
    radius = r_func(x,y)    
    fx = v_x
    fv_x = -GM*x/radius**3
    fy = v_y
    fv_y = -GM*y/radius**3
    
    return array([fx, fy, fv_x, fv_y], float)

def append_all(r):
    x_points.append(r[0])
    y_points.append(r[1])
    vx_points.append(r[2])
    vy_points.append(r[3])
    
#Using astropy to define constants
GM = G.value*M_sun.value
m = M_earth.value

Ke = []
U = []
x_points = []
y_points = []
vx_points = []
vy_points = []

#Definition of intial values for all variables
x_0 = 1.4710e11 #intial x posirion (m)
y_0 = 0 #inital y position (m)
vx_0 = 0 #initial x velocity (m/s)
vy_0 = 3.0287e4 #intial y velocity (m/s)

r = array([x_0, y_0, vx_0, vy_0])
a = 0 #intial time (s)
b = 31536000*3 #final time (s) (3 years)
h = 10000
tpoints = arange(a,b,h) 

for t in tpoints:
    append_all(r)
    Ke += [0.5 * m * (r[2] ** 2 + r[3] ** 2)] # Kinectic Energy (J)
    U += [-GM*m/r_func(r[0], r[1])] # Potential Energy (J)
    k1 = h*f(r,t)
    k2 = h*f(r + 0.5*k1, t+0.5*h)
    r += k2

#Changing energy list into arrays to sum them
Ke = array(Ke)
U = array(U)
E_tot = Ke+U #Total energy (sum of potential and kinetic energy) (J)

#Plot of E_tot alone
plt.plot(tpoints, E_tot, "k")
plt.xlabel("Temps (s)")
plt.ylabel("Énergie (J)")
plt.ticklabel_format(useOffset=False)
plt.title("Énergie totale en fonction du temps")
plt.show()