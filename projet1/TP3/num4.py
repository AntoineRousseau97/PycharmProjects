from astropy.constants import G, M_sun  # (m3 / (kg s2) and kg)
from numpy import array, sqrt, arange
import matplotlib.pyplot as plt


def r_func(x, y):
    return sqrt(x ** 2 + y ** 2)


def f(r):
    x = r[0]
    y = r[1]
    v_x = r[2]
    v_y = r[3]
    radius = r_func(x, y)
    fx = v_x
    fv_x = -GM * x / radius ** 3
    fy = v_y
    fv_y = -GM * y / radius ** 3

    return array([fx, fy, fv_x, fv_y], float)


def append_all(r):
    x_points.append(r[0])
    y_points.append(r[1])
    vx_points.append(r[2])
    vy_points.append(r[3])


GM = G.value * M_sun.value

x_points = []
y_points = []
vx_points = []
vy_points = []

x_0 = 4e12  # m
y_0 = 0  # m
vx_0 = 0  # m/s
vy_0 = 500  # m/s
r = array([x_0, y_0, vx_0, vy_0], float)

def f_tpts(a, b, N):
    h = (b - a) / N
    tpoints = arange(a, b, h)
    return h, tpoints

#tpoints = arange(a, b, h)

#    k1 = h * f(r)
#    k2 = h * f(r + 0.5 * k1)
#    k3 = h * f(r + 0.5 * k2)
#    k4 = h * f(r + k3)
#    r += (k1 + 2 * k2 + 2 * k3 + k4) / 6


def perf_h(target, r, h_init):
    rho = 0
    while rho < 1:

        #calcule de x_2
        k1_2 = 2*h_init * f(r)
        k2_2 = 2*h_init * f(r + 0.5 * k1_2)
        k3_2 = 2*h_init * f(r + 0.5 * k2_2)
        k4_2 = 2*h_init * f(r + k3_2)
        x_2 = (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2) / 6
        #calcul de x_1
        k1 = h_init * f(r)
        k2 = h_init * f(r + 0.5 * k1)
        k3 = h_init * f(r + 0.5 * k2)
        k4 = h_init * f(r + k3)
        x = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        k1_1 = h_init * f(x)
        k2_1 = h_init * f(x + 0.5 * k1_1)
        k3_1 = h_init * f(x + 0.5 * k2_1)
        k4_1 = h_init * f(x + k3_1)
        x_1 = (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6

        e_x = (x_1[2] - x_2[2])/30
        e_y = (x_1[3] - x_2[3])/30
        rho = ((h_init * target) / r_func(e_x, e_y))
        h_init = h_init/2
        print(rho)

    best_h = 2*h_init*(rho**(1/4))

    return best_h, rho

b = 3153600000
t = 0
tpoints = []
h = 31536
target = 0.000031709

while t < b:
    h = perf_h(target, r, h)[0]
    append_all(r)
    tpoints.append(t)
#    print(h)
    k1 = h * f(r)
    k2 = h * f(r + 0.5 * k1)
    k3 = h * f(r + 0.5 * k2)
    k4 = h * f(r + k3)
    r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t += h
    h = 2*h


plt.plot(tpoints, x_points)
plt.plot(tpoints, y_points)
plt.show()

# x' = v_x
# v_x' = -GM*x/r**3
# y' = v_y
# v_y' = -GM*y/r**3
