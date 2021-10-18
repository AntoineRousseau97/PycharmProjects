import numpy as np
import matplotlib.pyplot as plt

def get_data(file):
    data = np.loadtxt(file, delimiter=', ', skiprows=0)
    WN = data[:, 0]
    count = data[:, 1]
    return WN, count

WN, count = get_data("Methanol.txt")

print(count)

fig, axs = plt.subplots(4, sharex=True, sharey=True)
plt.xlabel("Wave Number [cm-1]", fontsize=15)
fig.suptitle('Raman spectrum of isopropanol, methanol and Ethanol (Vodka)')
axs[0].plot(get_data("Isoprop.txt")[0], get_data("Isoprop.txt")[1])
axs[1].plot(get_data("Methanol.txt")[0], get_data("Methanol.txt")[1])
axs[2].plot(get_data("Vodka.txt")[0], get_data("Vodka.txt")[1])
axs[3].plot(get_data("Olive.txt")[0], get_data("Olive.txt")[1])
plt.show()
#Ovile
# plt.annotate('775nm / 868nm', (500, 140000), size=16)
# plt.annotate('988 / 1082', (850, 133000), size=16)
# plt.annotate('1215 / 1300', (1050, 140000), size=16)
# plt.annotate('1355nm / 1438nm', (1370, 170000), size=16)
# plt.annotate('1572nm / 1655nm', (1400, 97000), size=16)
# plt.annotate('1666nm / 1750nm', (1600, 40000), size=16)
# plt.annotate('2800nm / 2900nm', (2700, 110000), size=16)


#Vodka
# plt.annotate('785nm / 884nm', (800, 75000), size=16)
# plt.annotate('955nm / 1053nm', (820, 28000), size=16)
# plt.annotate('995 / 1097', (1020, 22000), size=16)
# plt.annotate('1370nm / 1455nm', (1200, 32000), size=16)
# plt.annotate('2860nm / 2950nm', (2700, 22000), size=16)

#Methanol
# plt.annotate('925nm / 1035nm', (950, 65000), size=16)
# plt.annotate('1370nm / 1453nm', (1300, 20000), size=16)
# plt.annotate('2775nm / 2835nm', (2300, 12000), size=16)
# plt.annotate('2885nm / 2945nm', (2700, 20000), size=16)

#isoprop
# plt.annotate('723nm / 820nm', (745, 140000), size=16)
# plt.annotate('860nm / 955nm', (860, 35000), size=16)
# plt.annotate('1040nm / 1132nm', (880, 20000), size=16)
# plt.annotate('1369nm / 1454nm', (1369, 40000), size=16)
# plt.annotate('2866nm / 2940nm', (2500, 27000), size=16)
