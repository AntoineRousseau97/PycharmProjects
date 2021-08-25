from astropy.constants import G, M_sun #(m3 / (kg s2) and kg)
from numpy import array, sqrt, arange
import matplotlib.pyplot as plt
           
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
    

GM = G.value*M_sun.value

x_points = []
y_points = []
vx_points = []
vy_points = []

x_0 = 4e12 #m
y_0 = 0 #m
vx_0 = 0 #m/s
vy_0 = 500 #m/s
r = array([x_0, y_0, vx_0, vy_0], float)    

a = 0
b = 31536000*100    
N = 100000
h = (b-a)/N   
tpoints = arange(a,b,h)

for t in tpoints:
    append_all(r)
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h) 
    r += (k1+2*k2+2*k3+k4)/6 
    
plt.plot(tpoints, x_points)
plt.show()



# #x' = v_x
# #v_x' = -GM*x/r**3
# #y' = v_y
# #v_y' = -GM*y/r**3
