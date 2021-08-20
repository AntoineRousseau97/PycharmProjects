import numpy as np
import pandas as pd
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt
import csv

samp_rate = 5000
th = 0.7

ECG = np.loadtxt("data_simul/001-000.csv", delimiter=',')

samp = list(range(len(ECG)))

plt.plot(samp, ECG)
plt.plot(samp, [th]*len(samp))
plt.show()

th = 0.7

print(ECG)

def peak_detect(x, threshold=th):
    x = x - th
    x2 = np.pad(x, 1)[:-2]
    cross = x*x2
    res = np.sign(cross)
    return np.where(res==-1)[0].size/2

freq_card = 60*(peak_detect(ECG, th)/(len(ECG)/samp_rate))  #BPM
print(freq_card)

def getPD(x, threshold=0.7, samp_rate=5000):  #peak detection
    x = x - threshold
    x2 = np.pad(x, 1)[:-2]
    cross = x*x2
    res = np.sign(cross)
    nb_cross = np.where(res==-1)[0].size/2
    freq_card = 60*(nb_cross/(len(x)/samp_rate))  #BPM
    #print(freq_card)
    return freq_card

duderin = getPD(ECG)
print(duderin)


def plotfft(signal, fs, axis=[0, 10, 0, 5e6]):
  ps = np.abs(np.fft.fft(signal))**2
  time_step = 1/fs
  freqs = np.fft.fftfreq(signal.size, time_step)
  idx = np.argsort(freqs)
  plt.plot(freqs[idx], ps[idx])
  plt.axis(axis)


#plotfft(ECG, fs=5000)

#plt.show()

y= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
x=np.array(y)
def getAVG(x):
    y = 0
    for j in x:
        y = y + j
        AVG = y/len(x)
    return AVG

dude = getAVG(x)
print(dude)

# Python code to find average of each consecutive segment

# List initialisation

# Defining Splits
splits = 20

# Finding average of each consecutive segment
out = [sum(ECG[i:i + splits]) / splits
          for i in range(len(ECG))]

# printing output
print(len(ECG), len(out))


"""with open("data_real/READMEEEE.csv") as file:
    reader = csv.reader(file)
    data = []
    count = 0

    for i in reader:
        if i >= 0:
            count = count + 1
            print(i[0])
            data.append(i[0])
            if count > 10:
                break
#for j in range(len(data)):
#    if int(j) == False:
#        pass
#    data[j] = int(data[j])
#print(data)"""

import os
import sys
def _test(fileName):
    file = open(fileName, 'r')
    lines = file.readlines()
    data = []
    for l in lines:
        splitted = l.split(';')
        valECG = splitted[0]
        data.append(valECG)
    for i in range(0, len(data)):
        data[i] = int(data[i])
    data = np.array(data)
    return data

#data = customToCSV("data_real/repos.txt")


#print(data, len(data))


def customToCSV(fileName):
    file = open(fileName, 'r')
    lines = file.readlines()
    data = []
    poil = []
    for l in lines:
        splitted = l.split(';')
        valECG = splitted[0]
        data.append(valECG)
        valEMG = splitted[1]
        poil.append(valEMG)
    for i in range(0, len(data)):
        data[i] = int(data[i])
    data = np.array(data)

    for i in range(0, len(poil)):
        poil[i] = int(poil[i])
    poil = np.array(poil)
    return data, poil

data, poil = customToCSV("data_real/001-002.txt")

poil = list(poil)

def segment_dataset(filepath, window_length_0=200, classes=None):
    files = os.listdir(filepath)
    fileNames = []
    data = []
    labels = []
    for f in files:
        if f.endswith('.csv'):
            fileNames.append(f)

    for i in fileNames:
        fileName, fileType = i.split('.')
        metaData = fileName.split('-')      # [0]: EMG/ECG, [1]: intensité de l'effort

        if np.in1d(int(metaData[1]), classes) and np.in1d(int(metaData[0]), 0):
          data_read_ch0 = np.loadtxt(filepath+i, delimiter=',')  # Choosing channel 6 as first channel for this exercise

          len_data = len(data_read_ch0)
          n_window = int(len_data / window_length_0)

          data_windows_ch0 = [data_read_ch0[w * window_length_0:w * window_length_0 + window_length_0] for w in range(n_window)]

          data += [(a) for a in zip(data_windows_ch0)]
          labels += [int(metaData[1])]*n_window
        else:
          pass
    #print(data)
    return data, labels


#import pandas as pd
#df = pd.DataFrame(poil)
#df.to_csv("data_real/000-002.csv", index=False, header=False)

path = './data_real/'
classes = [0, 1, 2]
#data, labels = segment_dataset(path, window_length_0=200, classes=classes)

#print(data)

import csv

#a = izip(*csv.reader(open("data_real/interdem.csv", "rb")))
#csv.writer(open("data_real/000-000.csv", "wb")).writerows(a)

