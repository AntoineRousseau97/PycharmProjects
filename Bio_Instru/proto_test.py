import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
import csv

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


def getMAV(x):
    '''
    Computes the Mean Absolute Value (MAV)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Mean Absolute Value as [float]
    '''
    MAV = np.mean(np.abs(x))
    return MAV

def getRMS(x):
    '''
    Computes the Root Mean Square value (RMS)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Root Mean Square value as [float]
    '''
    RMS = np.sqrt(np.mean(x**2))
    return RMS

def getVar(x):
    '''
    Computes the Variance of EMG (Var)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Variance of EMG as [float]
    '''
    N = np.size(x)
    Var = (1/(N-1))*np.sum(x**2)
    return Var

def getSD(x):
    '''
    Computes the Standard Deviation (SD)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Standard Deviation as [float]
    '''
    N = np.size(x)
    xx = np.mean(x)
    SD = np.sqrt(1/(N-1)*np.sum((x-xx)**2))
    return SD

def getZC(x, threshold=0):
    '''
    Computes the Zero Crossing value (ZC)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Zero Crossing value as [float]
    '''
    N = np.size(x)
    ZC=0
    for i in range(N-1):
        if (x[i]*x[i+1] < 0) and (np.abs(x[i]-x[i+1]) >= threshold):
            ZC += 1
    return ZC

def getSSC(x, threshold=0):
    '''
    Computes the Slope Sign Change value (SSC)
    :param x: EMG signal vector as [1-D numpy array]
    :return: Slope Sign Change value as [float]
    '''
    N = np.size(x)
    SSC = 0
    for i in range(1, N-1):
        if (((x[i] > x[i-1]) and (x[i] > x[i+1])) or ((x[i] < x[i-1]) and (x[i] < x[i+1]))) \
                and ((np.abs(x[i]-x[i+1]) >= threshold) or (np.abs(x[i]-x[i-1]) >= threshold)):
            SSC += 1
    return SSC


def getPD(x, threshold=0.7, samp_rate=5000):  #peak detection
    x = x - threshold
    x2 = np.pad(x, 1)[:-2]
    cross = x*x2
    res = np.sign(cross)
    nb_cross = np.where(res==-1)[0].size/2
    freq_card = 60*(nb_cross/(len(x)/(samp_rate/20)))  #BPM
    #print(freq_card)
    return freq_card


def getAVG(x):
    y = 0
    for j in x:
        y = y + j
        AVG = y/len(x)
    return AVG


def seg_AVG(filepath, window_length=20, classes=None):
    files = os.listdir(filepath)
    fileNames = []
    data_AVG = []
    labels_AVG = []
    for f in files:
        if f.endswith('.csv'):
            fileNames.append(f)

    for i in fileNames:
        fileName, fileType = i.split('.')
        metaData = fileName.split('-')      # [0]: EMG/ECG, [1]: intensité de l'effort

        if np.in1d(int(metaData[1]), classes) and np.in1d(int(metaData[0]), 1):
          data_read = np.loadtxt(filepath+i, delimiter=',')

          len_data = len(data_read)
          n_window = int(len_data / window_length)

          data_AVG += [data_read[w * window_length:w * window_length + window_length] for w in range(n_window)]
          #print(data_windows)
          #data_AVG += [(a) for a in zip(data_windows)]
          labels_AVG += [int(metaData[1])]*n_window
        else:
          pass
    #print(data_AVG)
    return data_AVG, labels_AVG, data_read


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
          #data_read_ch0 = data_read_ch0 - getAVG()

          len_data = len(data_read_ch0)
          n_window = int(len_data / window_length_0)

          data_windows_ch0 = [data_read_ch0[w * window_length_0:w * window_length_0 + window_length_0] for w in range(n_window)]

          data += [(a) for a in zip(data_windows_ch0)]
          labels += [int(metaData[1])]*n_window
        else:
          pass
    #print(data)
    return data, labels



def features_dataset(data, MAV=True, RMS=True, Var=True, SD=True, ZC=True, SSC=True):
    dataset = []
    for d in data:
        feature_vector = []
        if MAV==True:
            feature_vector += [getMAV(d[0])]
        if RMS==True:
            feature_vector += [getRMS(d[0])]
        if Var==True:
            feature_vector += [getVar(d[0])]
        if SD==True:
            feature_vector += [getSD(d[0])]
        if ZC==True:
            feature_vector += [getZC(d[0])]
        if SSC==True:
            feature_vector += [getSSC(d[0])]
        dataset += [feature_vector]
        #print(feature_vector)
    return dataset


# Loading recorded EMG signals into numpy arrays
#eff_0_0 = np.loadtxt('./data_simul/000-000.csv', delimiter=',')
eff_0_1 = np.loadtxt('./data_simul/000-001.csv', delimiter=',')
eff_0_2 = np.loadtxt('./data_simul/000-002.csv', delimiter=',')




# Define a function to plot the FFT of a signal
def plotfft(signal, fs, axis=[0, 200, 0, 3e8]):
  ps = np.abs(np.fft.fft(signal))**2
  time_step = 1/fs
  freqs = np.fft.fftfreq(signal.size, time_step)
  idx = np.argsort(freqs)
  plt.plot(freqs[idx], ps[idx])
  plt.axis(axis)



# Exploratory data analysis (EDA)
#data = eff_0_1   # Use index 6 and 17 for this exercise
                        # (channel 6 and 17 of original data)
dog, data = customToCSV("data_real/001-002.txt")
plt.figure(figsize=(15, 3))
plt.subplot(1, 2, 1)
plt.title('EMG Signal')
plt.plot(data)
plt.axis([0, None, 0, 10])

plt.subplot(1, 2, 2)
plt.title('FFT')
plotfft(data, fs=5000)

plt.show()



# Loading recorded EMG signals into numpy arrays
path = './data_real/'

classes = [0, 1, 2]

data, labels = segment_dataset(path, window_length_0=200, classes=classes)

features_set = features_dataset(data, MAV=True, RMS=True, Var=True, SD=True, ZC=True, SSC=True)

features_set = preprocessing.scale(features_set) # preprocessing module imported from sklearn

print(features_set)

feat_x = 1 # 0: MAV, 1: RMS, 2: Var, 3: SD, 4: ZC, 5: SSC
feat_y = 3

for c in classes:
  ind = np.where(np.array(labels)==c)
  plt.scatter([f[feat_x] for f in features_set[ind]], [f[feat_y] for f in features_set[ind]], label='Class '+str(c))
plt.legend()
plt.xlabel("Valeur RMS")
plt.ylabel("Écart type")
plt.show()

#traning algorithm
avgScoreTemp = []

kFold_rep = 3
kFold_splits = 3
kFold = RepeatedKFold(n_splits=kFold_splits, n_repeats=kFold_rep)

for i in range(kFold_rep):
  for i_Train, i_Test in kFold.split(features_set):
    clf = LinearDiscriminantAnalysis()
    X_train, X_test = [features_set[j] for j in i_Train], [features_set[k] for k in i_Test]
    y_train, y_test = [labels[l] for l in i_Train], [labels[m] for m in i_Test]

    clf.fit(X_train, y_train)
    currentScore = clf.score(X_test, y_test)

    avgScoreTemp += [currentScore]

print(X_test, y_test)

avgScore = sum(avgScoreTemp)/len(avgScoreTemp)
print('Mean score with k-fold validation: {}'.format(avgScore))






# ECG !

eff_1_0 = np.loadtxt('./data_simul/001-000.csv', delimiter=',')
eff_1_1 = np.loadtxt('./data_simul/001-001.csv', delimiter=',')
eff_1_2 = np.loadtxt('./data_simul/001-002.csv', delimiter=',')


# No eff

eff_1_0, dog = customToCSV("data_real/001-000.txt")

time_1_0 = list(np.array(list(range(len(eff_1_0))))/(5000))

plt.plot(time_1_0, eff_1_0)
#plt.show()

window_L = 20

SEG_1_0 = []
n_window_1_0 = int(len(eff_1_0) / window_L)
SEG_1_0 += [eff_1_0[w * window_L:w * window_L + window_L] for w in range(n_window_1_0)]

AVG_1_0_li = []
for i in SEG_1_0:
    AVG_1_0_li.append(getAVG(i))
AVG_1_0 = np.array(AVG_1_0_li)

freq_1_0 = getPD(AVG_1_0, threshold=300, samp_rate=5000)
x_freq_1_0 = 0

# effort normal

eff_1_1, dog = customToCSV("data_real/001-001.txt")
time_1_1 = list(np.array(list(range(len(eff_1_1))))/(5000))

plt.plot(time_1_1, eff_1_1)
#plt.show()

SEG_1_1 = []
n_window_1_1 = int(len(eff_1_1) / window_L)
SEG_1_1 += [eff_1_1[w * window_L:w * window_L + window_L] for w in range(n_window_1_1)]

AVG_1_1_li = []
for i in SEG_1_1:
    AVG_1_1_li.append(getAVG(i))
AVG_1_1 = np.array(AVG_1_1_li)

freq_1_1 = getPD(AVG_1_1, threshold=100, samp_rate=5000)
x_freq_1_1 = 1

# effort apres

eff_1_2, dog = customToCSV("data_real/001-002.txt")
time_1_2 = list(np.array(list(range(len(eff_1_2))))/(5000))

plt.plot(time_1_2, eff_1_2)
#plt.show()

SEG_1_2 = []
n_window_1_2 = int(len(eff_1_2) / window_L)
SEG_1_2 += [eff_1_2[w * window_L:w * window_L + window_L] for w in range(n_window_1_2)]

AVG_1_2_li = []
for i in SEG_1_2:
    AVG_1_2_li.append(getAVG(i))
AVG_1_2 = np.array(AVG_1_2_li)

freq_1_2 = getPD(AVG_1_2, threshold=50, samp_rate=5000)
x_freq_1_2 = 1


print('fréquence au repos: ', freq_1_0)
print('fréquence pendant effort: ', freq_1_1)
print("fréquence apres effort: ", freq_1_2)

