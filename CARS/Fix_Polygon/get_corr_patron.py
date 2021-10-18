from PIL import Image
import os
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from numpy import genfromtxt

test_image_raw = "/Users/antoinerousseau/Downloads/background/-001/Bliq VMS-ch-2/-001-Bliq VMS-ch-2-Time_0.015s-Frame_004953.tif"
real_path = "/Users/antoinerousseau/Downloads/background/-001/Bliq VMS-ch-2/"


def read_file(file_path):
    with Image.open(file_path) as f:
        file_array = np.array(f)
        #f.show()
        return file_array

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

#print(np.percentile(AVG, 10), np.percentile(AVG, 5),np.percentile(AVG, 4),np.percentile(AVG, 3),np.percentile(AVG, 2),np.percentile(AVG, 1))

#AVG = (AVG > np.percentile(AVG, 20)) * AVG
#print(AVG)

AVG[AVG==0] = np.amax(AVG)

def corr_mat(m):
    try:
        return np.amax(AVG)/m
    except ZeroDivisionError:
        return 0

N_row = len(images[0])

for i in range(7, N_row, 36):
    if i < 511:
        AVG[i] = AVG[i+1]/2 + AVG[i-1]/2
    #corr[i] = np.percentile(corr, 0)

for i in range(35, N_row, 36):
    if i < 511:
        AVG[i] = AVG[i+1]/2 + AVG[i-1]/2
    #corr[i] = np.percentile(corr, 0)

test_image = read_file(test_image_raw)
for i in range(7, N_row, 36):
    if i < 511:
        test_image[i] = test_image[i+1]/2 + test_image[i-1]/2

for i in range(35, N_row, 36):
    if i < 511:
        test_image[i] = test_image[i+1]/2 + test_image[i-1]/2

corr = corr_mat(AVG)

np.savetxt("corr_mat.csv", corr, delimiter=",")

test = test_image * corr

# plt.imshow(AVG)
# plt.show()
# plt.imshow(test_image)
# plt.show()
# plt.imshow(corr)
# plt.show()
# plt.imshow(test)
# plt.show()