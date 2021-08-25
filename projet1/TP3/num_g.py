from math import sin,pi 
from numpy import empty,array,arange, sqrt
import matplotlib.pyplot as plt
from astropy.constants import G, M_sun #(m3 / (kg s2) and kg)

GM = G.value*M_sun.value
x_points = []
y_points = []
vx_points = []
vy_points = []

#Definition of intial values for all variables
x_0 = 1.4710e11 #intial x posirion (m)
y_0 = 0 #inital y position (m)
vx_0 = 0 #initial x velocity (m/s)
vy_0 = 3.0287e4 #intial y velocity (m/s)


a = 0 #intial time (s)
b = 31536000*3 #final time (s) (3 years)
#N = 10000 #step size (1 hour in seconds)
#H = (b-a)/N 
H = 604800 # 1 week in seconds
delta = 3.1709792e-5 #m/s

def r_func(x,y):
    return sqrt(x**2 + y**2)

def f(r):
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

tpoints = arange(a,b,H) 
r = array([x_0, y_0, vx_0, vy_0], float)  # array containing x and y position
# Do the "big steps" of size H 
for t in tpoints: 
    x_points.append(r[0])
    y_points.append(r[1]) 
    vx_points.append(r[2])
    vy_points.append(r[3]) 
    
    # Do one modified mipoint step of size H 
    # to get things started 
    n = 1 
    r1 = r + 0.5*H*f(r) 
    r2 = r + H*f(r1) 
    # The array R1 stores the first row of the 
    # extrapolation table, which contains only the single 
    # modified midpoint estimate of the solution at the 
    # end of the interval 
    R1 = empty([1,4] ,float) 
    R1[0] = 0.5*(r1 + r2 + 0.5*H*f(r2)) 
# Now increase n until the required accuracy is reached 
    error = 2*H*delta 
    while error>H*delta: 
        n += 1 
        h = H/n 
        # Modified midpoint method 
        r1 = r + 0.5*h*f(r) 
        r2 = r + h*f(r1) 
        for i in range(n-1): 
            r1 += h*f(r2) 
            r2 += h*f (r1) 
        # Calculate extrapolation estimates. Arrays R1 and R2 
        # hold the two most recent lines of the table 
        R2 = R1     
        R1 = empty([n,4] ,float) 
        R1[0] = 0.5*(r1 + r2 + 0.5*h*f(r2)) 
        for m in range(1,n): 
            epsilon= (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1) 
            R1[m] = R1[m-1] + epsilon 
        error= abs(epsilon[0]) 
        
    # Set r equal to the most accurate estimate we have, 
    # before moving on to the next big step 
    r = R1[n-1] 

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