import numpy as np
import matplotlib.pyplot as plt 

# Definition of constants
M = 9.109e-31  # mass of electron (g)
L = 1e-8  #length of box (m)
h_bar = 6.626e-34 #Planck constant (J⋅s)

N = 1000  # number of spacial steps
a = L / N  # spacial step size

h = 1e-18 # temporal step (s)
tf = 1e-16 #final time (s)
t_step = int(tf/h) #number of temporal steps

#Intialisation of coefficients
const = h * (1j *  h_bar) / (2 * M * a ** 2) 
a1 = 1 + const
a2 = -const/2
b1 = 1 - const
b2 = const/2

# a1 = 1 + h * (1j *  h_bar) / (2 * M * a ** 2)
# a2 = - h * (1j * h_bar) / (4 * M * a ** 2)
# b1 = 1 - 1j * h * h_bar / (2 * M * a ** 2)
# b2 = h * (1j * h_bar) / (4 * M * a ** 2)

#Definition of wave function of electron
def psi0(x):
    #Defining constants
    x_0 = L/2 #inital position
    s = 1e-10  # sigma (m)
    k = 5e10  # kappa (m**-1)
    
    return np.exp(-(x - x_0) ** 2 / (2 * s ** 2)) * np.exp(1j * k * x)

def banded(Aa,va,up,down):

    # Copy the inputs and determine the size of the system
    A = np.copy(Aa)
    v = np.copy(va)
    N = len(v)

    # Gaussian elimination
    for m in range(N):

        # Normalization factor
        div = A[up,m]

        # Update the vector first
        v[m] /= div
        for k in range(1,down+1):
            if m+k<N:
                v[m+k] -= A[up+k,m]*v[m]

        # Now normalize the pivot row of A and subtract from lower ones
        for i in range(up):
            j = m + up - i
            if j<N:
                A[i,j] /= div
                for k in range(1,down+1):
                    A[i+k,j] -= A[up+k,m]*A[i,j]

    # Backsubstitution
    for m in range(N-2,-1,-1):
        for i in range(up):
            j = m + up - i
            if j<N:
                v[m] -= A[i,j]*v[j]

    return v


# initial conditions
x_points = np.linspace(0, L, N + 1)

#initailisation of psi(t) vector
psi = np.zeros(N+1, complex)
psi[:] = psi0(x_points) #setting all elements to psi(x,0)
psi[0], psi[N] = 0, 0 #limit conditions at x = 0 and x = L

#Create the matrix A
# A = np.zeros([N + 1, N + 1], complex)
# A[0, 0] = a1
# A[0, 1] = a2
# A[N, N - 1] = a2
# A[N, N] = a1
# for i in range(N):
#     A[i, i - 1] = a2
#     A[i, i] = a1
#     A[i, i + 1] = a2

A = np.empty((3,N),complex)
A[0,:] = a2
A[1,:] = a1
A[2:,] = a2

for i in range(t_step):
    v = b1*psi[1:N] + b2*(psi[2:N+1] + psi[0:N-1])
    psi[1:N] = banded(A, v, 1, 1)
    
plt.plot(psi)
# plt.show()

#print(len(psi))
# psi = array(list(map(psi0, x_points)), complex)
# psi[0], psi[N - 1] = 0, 0 #x = 0 and x = L


#
# # Create the matrix B
# B = zeros([N + 1, N + 1], complex)
# B[0, 0] = b1
# B[0, 1] = b2
# B[N, N - 1] = b2
# B[N, N] = b1
# for i in range(N):
#     B[i, i - 1] = b2
#     B[i, i] = b1
#     B[i, i + 1] = b2

# Create the matrix A in the form appropriate for the function solve_banded
# A2 = empty([3, N + 1], complex)
# A2[0, 0] = 0
# A2[0, 1:] = a2
# A2[1, :] = a1
# A2[2, 0: N] = a2
# A2[2, N] = 0

# # Main loop
# # store the wavefunction at each time step in a list
# solution = [psi]
# for i in range(300):
#     psi[1: N] = b1 * psi[1: N] + b2 * (psi[2:] + psi[0: N - 1])
#     psi = solve_banded((1, 1), A2, psi)
#     solution.append(psi)

# plot(x_points, abs(solution[0]) ** 2)
# plot(x_points, abs(solution[49]) ** 2)
# plot(x_points, abs(solution[250]) ** 2)
# xlabel("x (m)")
# ylabel("$\psi(x)$")
# show()