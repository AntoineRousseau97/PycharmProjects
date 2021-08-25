from astropy.constants import G, M_sun #(m3 / (kg s2) and kg)
from numpy import array, sqrt, arange, empty, transpose
import matplotlib.pyplot as plt
from astropy.time import Time# ephemerides
import de421
from jplephem import Ephemeris

eph = Ephemeris(de421) 
# dates
lancement=Time("2020-07-30")
atterissage=Time("2021-02-18")
# un nombre de jours juliens est attendu par la routine, d'ou le .jd position en km, vitesse en km par jour
position, velocity = eph.position_and_velocity('mars',lancement.jd)
p = []
v = []

GM = G.value*M_sun.value
x_points = []
y_points = []
z_points = []
vx_points = []
vy_points = []
vz_points = []
#Definition of intial values for all variables
x_0 = p[0]*1000 #intial x position (m)
y_0 = p[1]*1000 #inital y position (m)
z_0 = p[2]*1000#initial z position (m)
vx_0 = v[0]*1000/86400 #initial x velocity (m/s)
vy_0 = v[1]*1000/86400 #intial y velocity (m/s)
vz_0 = v[2]*1000/86400#initial z velocity (m/s)

a = 0 #intial time (s)
b = 86400*1000 #final time (s) (203 days)
N = 10000 #step size
H = (b-a)/N 
delta = 3.1709792e-5 #conversion of km/year into m/s

def r_func(x,y,z):
    return sqrt(x**2 + y**2 + z**2)

def f(r):
    x = r[0]
    y = r[1]
    z = r[2]
    v_x = r[3]
    v_y= r[4]
    v_z = r[5]
    radius = r_func(x,y,z)
    fx = v_x
    fv_x = -GM*x/radius**3
    fy = v_y
    fv_y = -GM*y/radius**3
    fz = v_z
    fv_z = -GM*z/radius**3
    return array([fx, fy, fz, fv_x, fv_y, fv_z], float)

tpoints = arange(a,b,H) 
thetapoints = [] 
r = array([x_0, y_0, z_0, vx_0, vy_0, vz_0], float)  #array containing positions and velocities
# Do the "big steps" of size H 
for t in tpoints: 
    thetapoints.append(r[0]) 
    # Do one modified mipoint step of size H 
    # to get things started 
    n = 1 
    r1 = r + 0.5*H*f(r) 
    r2 = r + H*f(r1) 
    # The array R1 stores the first row of the 
    # extrapolation table, which contains only the single 
    # modified midpoint estimate of the solution at the 
    # end of the interval 
    R1 = empty([1,6], float) 
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
        R1 = empty([n,6], float) 
        R1[0] = 0.5*(r1 + r2 + 0.5*h*f(r2)) 
        for m in range(1,n): 
            epsilon= (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1) 
            R1[m] = R1[m-1] + epsilon 
        error= abs(epsilon[0]) 
    # Set r equal to the most accurate estimate we have, 
    # before moving on to the next big step 
    r = R1[n-1] 
# Plot the results 
plt.plot(tpoints,thetapoints) 
#plot(tpoints,thetapoints,"b.") 
plt.show() 