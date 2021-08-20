import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing

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

def getFC(filepath, threshold=None, samp_rate=None, classes=None):  #get fréquence cardiaque
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

        if np.in1d(int(metaData[0]), classes) and np.in1d(int(metaData[0]), 1):
          data_read_ch1 = np.loadtxt(filepath+i, delimiter=',')

          len_data = len(data_read_ch1)

          data_read_ch1 = data_read_ch1 - threshold
          x2 = np.pad(data_read_ch1, 1)[:-2]
          cross = data_read_ch1 * x2
          res = np.sign(cross)
          nb_cross = np.where(res == -1)[0].size / 2
          freq_card = 60 * (nb_cross / (len_data / samp_rate))  # BPM

          data += [(a) for a in zip([freq_card])]
          labels += [int(metaData[1])]

        else:
            pass
    #print(data, labels)
    return data, labels


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
    #print([int(metaData[1])])
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
    return dataset


# Loading recorded EMG signals into numpy arrays
eff_0_0 = np.loadtxt('./data_simul/000-000.csv', delimiter=',')
eff_0_1 = np.loadtxt('./data_simul/000-001.csv', delimiter=',')
eff_0_2 = np.loadtxt('./data_simul/000-002.csv', delimiter=',')

# Define a function to plot the FFT of a signal
def plotfft(signal, fs, axis=[0, 500, 0, 2e5]):
  ps = np.abs(np.fft.fft(signal))**2
  time_step = 1/fs
  freqs = np.fft.fftfreq(signal.size, time_step)
  idx = np.argsort(freqs)
  plt.plot(freqs[idx], ps[idx])
  plt.axis(axis)



# Exploratory data analysis (EDA)
data = eff_0_1   # Use index 6 and 17 for this exercise
                        # (channel 6 and 17 of original data)

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
path = './data_simul/'

classes = [0, 1, 2]

data, labels = segment_dataset(path, window_length_0=150, classes=classes)

features_set = features_dataset(data, MAV=True, RMS=True, Var=True, SD=True, ZC=True, SSC=True)

features_set = preprocessing.scale(features_set) # preprocessing module imported from sklearn


feat_x = 1  # 0: MAV, 1: RMS, 2: Var, 3: SD, 4: ZC, 5: SSC
feat_y = 3

for c in classes:
  ind = np.where(np.array(labels)==c)
  plt.scatter([f[feat_x] for f in features_set[ind]], [f[feat_y] for f in features_set[ind]], label='Class '+str(c))
plt.legend()
plt.xlabel("feat_x")
plt.ylabel("feat_y")
plt.show()

#frequence cardiaque

freq_card = getFC(path, threshold=0.7, samp_rate=5000, classes=classes)

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

avgScore = sum(avgScoreTemp)/len(avgScoreTemp)
print('Mean score with k-fold validation: {}'.format(avgScore))

