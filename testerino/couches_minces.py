from matplotlib import pyplot as plt

p_1235 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
T_p_1235 = [4.52, 4.50, 4.47, 4.43, 4.39, 4.34, 4.28, 4.22, 4.17]

p_1401 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
T_p_1401 = [4.62, 4.61, 4.62, 4.59, 4.57, 4.54, 4.50, 4.47, 4.42, 4.37, 4.32]

#wl_473 = [432, 502, 612, 798]
#n_473 = [2.74, 2.65, 2.59, 2.53]

#wl_1235 = [406, 426, 448, 474, 506, 542, 586, 638, 704, 786]
#n_1235 = [2.63, 2.59, 2.54, 2.49, 2.46, 2.41, 2.37, 2.32, 2.28, 2.23]

#wl_1401 = [419, 437, 459, 483, 513, 547, 587, 635, 693, 765]
#n_1401 = [2.69, 2.65, 2.62, 2.59, 2.56, 2.54, 2.51, 2.49, 2.47, 2.46]


plt.plot(p_1235, T_p_1235, ".k", label= "1235")
plt.plot(p_1401, T_p_1401, ".r", label= "1401")

plt.xlabel("ordre p d'interférence constructive ")
plt.ylabel("épaisseur de la couche [nm]")

#plt.plot(wl_473, n_473, ".k", label="Couche PbCl2 473 nm")
#plt.plot(wl_1235, n_1235, ".r", label="Couche PbCl2 1235 nm")
#plt.plot(wl_1401, n_1401, ".b", label="Couche PbCl2 1401 nm")

#plt.xlabel("indice de réfraction n")
#plt.ylabel("longueur d'onde [nm]")
plt.legend()
plt.show()