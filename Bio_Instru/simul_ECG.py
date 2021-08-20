import neurokit2 as nk
import matplotlib.pyplot as plt
import csv


dur = 8
samp_rate = 5000

x = list(range(0,dur*samp_rate))

simul_ECG = nk.ecg_simulate(duration=dur, sampling_rate=samp_rate, heart_rate=210)

#with open("data_simul/001-002.csv", "w", newline="") as fng:
#    a = csv.writer(fng)
#    a.writerows([simul_ECG])

plt.plot(x, simul_ECG)
plt.show()