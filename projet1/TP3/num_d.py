from astropy.constants import G, M_sun, M_earth #(m3 / (kg s2) and kg)
from numpy import array, sqrt, arange
import matplotlib.pyplot as plt
   
#function that calculates the radius with x and y         
def r_func(x,y):
    return sqrt(x**2 + y**2)

#function calculating the RHS of all 4 1st order ODEs
def f(r, v, t):
    x = r[0]
    y = r[1]
    v_x = v[0]
    v_y= v[1]
    radius = r_func(x,y)
    fx = v_x
    fv_x = -GM*x/radius**3
    fy = v_y
    fv_y = -GM*y/radius**3
    
    return array([fx, fy], float), array([fv_x, fv_y], float)

#function appending each part of vector r and v to the correct list
def append_all(r, v):
    x_points.append(r[0])
    y_points.append(r[1])
    vx_points.append(v[0])
    vy_points.append(v[1])
    vx2_points.append(v_2[0])
    vy2_points.append(v_2[1])
    x2_points.append(r_2[0])
    y2_points.append(r_2[1])
    
#Using astropy to define constants
GM = G.value*M_sun.value
m = M_earth.value

x_points = []
y_points = []
vx_points = []
vy_points = []
vx2_points = []
vy2_points = []
x2_points = []
y2_points = []

#Definition of intial values for all variables
x_0 = 1.4710e11 #intial x posirion (m)
y_0 = 0 #inital y position (m)
vx_0 = 0 #initial x velocity (m/s)
vy_0 = 3.0287e4 #intial y velocity (m/s)

r = array([x_0, y_0], float)  #array containing x and y position
v = array([vx_0, vy_0], float) #array containing x and y velocity
a = 0 #intial time (s)
b = 31536000*3 #final time (s) (3 years)
h = 3600 #step size (1 hour in seconds)
tpoints = arange(a,b,h) 
v_int = v + 0.5*h*f(r,v,0)[1] #initialization of v(t+1/2h)
v_2 = v + h*f(r,v,0)[1]
Ke = []
U = []
error = []
r_2 = [0, 0]

for t in tpoints:
    append_all(r,v)
    Ke += [0.5 * m * (v[0] ** 2 + v[1] ** 2)] # Kinectic Energy (J)
    U += [-GM*m/r_func(r[0], r[1])] # Potential Energy (J)
    r_2 = r.copy()
    v_2 = v.copy()
    r += h*v_int #r(t+h)
    k = h*f(r,v,t)[1] #k
    v = v_int + 0.5*k #v(t+h)
    v_2 += k
    r_2 += 2*h*v_2
    v_int += k #v(t+3/2h)
    
for i in range(2, len(x_points)):
    err_x = ((x_points[i]-x2_points[i-1])/6)
    err_y = ((y_points[i]-y2_points[i-1])/6)
    error += [r_func(err_x, err_y)]

error_year = (sum(error)/3)/1000
print(error_year)



#Changing energy list into arrays to sum them
Ke = array(Ke)
U = array(U)
E_tot = Ke+U #Total energy (sum of potential and kinetic energy) (J)

#Plot of Ke, U and E_tot
plt.plot(tpoints, Ke, "k", label = "Énergie cinétique")
plt.plot(tpoints, U, "r", label = "Énergie potentielle")
plt.plot(tpoints, E_tot, "k--", label = "Énergie totale")
plt.xlabel("Temps (s)")
plt.ylabel("Énergie (J)")
plt.legend()
plt.title("Bilan d'énergie")
plt.show()

#Plot of E_tot alone
plt.plot(tpoints, E_tot, "k")
plt.xlabel("Temps (s)")
plt.ylabel("Énergie (J)")
plt.ticklabel_format(useOffset=False)
plt.title("Énergie totale en fonction du temps")
plt.show()
    
#Plot of x position and y position in function of time
plt.plot(tpoints, x_points, color = "k", label = "Position en x")
plt.plot(tpoints, y_points, color = "r", label = "Position en y")
plt.xlabel("temps (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.show()

#Plot of x position in function of y position
plt.plot(x_points, y_points, "k")
plt.plot(0,0,"ko")
plt.xlabel("Position en x (m)")
plt.ylabel("Position en y (m)")
plt.show()