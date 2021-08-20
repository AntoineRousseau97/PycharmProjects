import numpy as np
import matplotlib.pyplot as plt
import csv
#matplotlib inline

# simulate EMG signal
max = np.random.normal(2, 1, size=25000)     #000-002
norm = np.random.normal(2, 0.7, size=25000)   #000-001
abs = np.random.normal(2, 0.03, size=25000)  #000-000
#emg = np.concatenate([quiet, burst1, quiet, burst2, quiet])
#time = np.array([i/1000 for i in range(0, len(emg), 1)]) # sampling rate 1000 Hz

with open("data_simul/000-000.csv", "w", newline="") as fng:
    a = csv.writer(fng)
    a.writerows([abs])

#zo = 0
#for i in burst1:
#    zo += i
#    a = (zo/len(burst1))
#print(emg/len(emg))


# plot EMG signal
fig = plt.figure()
plt.plot(range(len(norm)), norm)
plt.xlabel('data')
plt.ylabel('EMG (a.u.)')
fig_name = 'fig2.png'
fig.set_size_inches(w=11,h=7)
#fig.savefig(fig_name)
plt.show()