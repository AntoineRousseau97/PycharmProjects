from astropy.constants import G, M_sun  # (m3 / (kg s2) and kg)
from numpy import array, sqrt, arange
import matplotlib.pyplot as plt

#Fonction qui permet d'effectuer la racine d'une somme de carrés
def r_func(x, y):
    return sqrt(x ** 2 + y ** 2)

#Fonction représentant les 4 équations différentiels de notre problème
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

    return array([fx, fy, fv_x, fv_y])

GM = G.value * M_sun.value
x_0 = 4e12  # m
y_0 = 0  # m
vx_0 = 0  # m/s
vy_0 = 500  # m/s
r = array([x_0, y_0, vx_0, vy_0], float)

# Fonction qui effectue un itération de la m;thode Runge-Kutta 4em ordre
def RK_step(r, h):
    k1 = h * f(r)
    k2 = h * f(r + 0.5*k1)
    k3 = h * f(r + 0.5*k2)
    k4 = h * f(r + k3)
    return (k1 + 2*k2 + 2*k3 + k4)/6

# Fonction qui permet d'évaluer l'erreur, de trouver le pas optimal, de trouver la position de la comete, ainsi que le temps associé
def time_step(r, t, h):

    # Opère 2 itérations de longueur h de la fonction Runge-Kutta afin d'obtenir un estimé de la valeur à t+2h
    r_1_1 = RK_step(r, h)
    r_1_2 = RK_step(r + r_1_1, h)
    r_1 = r_1_1 + r_1_2

    # Opère 1 itération de longueur 2h de la fonction Runge-Kutta afin d'obtenir un estimé de la valeur à t+2h
    r_2 = RK_step(r, 2 * h)

    x_1 = r_1[0]
    x_2 = r_2[0]
    y_1 = r_1[1]
    y_2 = r_2[1]

    #calcul de rho, le ratio entre l'erreur voulue et calculée
    rho = h*delta / ((r_func((x_1 - x_2), (y_1 - y_2)))/30)

    # permet d'ajuster h (le pas) et de mettre à jour t (le temps)
    if rho >= 1:
        t = t + 2*h
        if rho**(1/4) > 2:
            h *= 2
        else:
            h *= rho**(1/4)

        return r_1, h, t

    else:
        return time_step(r, t, h*(rho**(1/4)))

a = 0 # Temps initial [s]
b = 100*31536000  # Temps final [s] (100 ans)
delta = 1000/(31536000) #Valeur de la target en m/s
h = b/150000  # step size initial

#Création des lsites contenant la position de la comete ainsi que le temps assosié à chaque itération
tpoints = []
xpoints = []
ypoints = []

t = a # On défini que le t initial est a

# Calcul de la position et de la vitesse de la comete à chaque itération jusqu'à atteindre le temps désiré
while t < b:
    tpoints.append(t)
    xpoints.append(r[0])
    ypoints.append(r[1])
    delta_r, h, t = time_step(r, t, h)
    r += delta_r

# Création du graphique
plt.plot(array(xpoints), array(ypoints), 'b')
plt.plot(array(xpoints)[::100], array(ypoints)[::100], 'ro')
plt.xlabel('Position en x [m]')
plt.ylabel('Position en y [m]')
plt.show()