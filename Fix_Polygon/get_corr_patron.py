from PIL import Image
import os
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from numpy import genfromtxt

test_image_raw = "C:/Users/antoi/OneDrive/Desktop/DCC_patch/Stage DCC/DCC_lab/CARS/experimental_data/2021.05.18/kelly6.tif"
path = "C:/Users/antoi/OneDrive/Desktop/DCC_patch/Stage DCC/DCC_lab/CARS/experimental_data/2021.05.18"
real_path = "C:/Users/antoi/OneDrive/Desktop/DCC_patch/Stage DCC/DCC_lab/CARS/curvature_fix/18.05.2021.kelly-001/18.05.2021.kelly-001/Bliq VMS-ch-2"
os.chdir(path)


def read_file(file_path):
    with Image.open(file_path) as f:
        file_array = np.array(f)
        #f.show()
        return file_array

test_image = read_file(test_image_raw)
images = []

# iterate through all file
for file in os.listdir(real_path):
    # Check whether file is in tif format or not
    if file.endswith(".tif"):
        file_path = f"{real_path}/{file}"

        # call read text file function
        images.append(read_file(file_path))

AVG = np.zeros((512, 1024))

for i in images:
    AVG += i

#print(AVG)

corr = []
rows = []

#print(np.percentile(AVG, 10), np.percentile(AVG, 5),np.percentile(AVG, 4),np.percentile(AVG, 3),np.percentile(AVG, 2),np.percentile(AVG, 1))

#AVG = (AVG > np.percentile(AVG, 20)) * AVG
#print(AVG)



AVG[AVG==0] = np.amax(AVG)


def corr_mat(m):
    try:
        return np.amax(AVG)/m
    except ZeroDivisionError:
        return 0

corr = corr_mat(AVG)

N_row = len(images[0])

np.savetxt("corr_mat.csv", corr, delimiter=",")

test = test_image * corr

for i in range(20, N_row, 36):
    test[i] = (test[i+1] + test[i-1])/2
    #corr[i] = np.percentile(corr, 0)

for i in range(34, N_row, 36):
    test[i] = (test[i+1] + test[i-1])/2
    #corr[i] = np.percentile(corr, 0)

plt.imshow(AVG)
plt.show()
plt.imshow(test_image)
plt.show()
plt.imshow(corr)
plt.show()
plt.imshow(test)
plt.show()

