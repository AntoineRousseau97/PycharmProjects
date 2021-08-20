import numpy as np
from sklearn.decomposition import PCA
import os
import csv

path = "C:/Users/antoi/OneDrive/Desktop/DCC_patch/Stage DCC/DCC_lab/Spectroscopy_Raman/data_27_05_2021/spectres"


def read_file(file_path):
    with open(file_path, "r") as f:
        file_array = []
        for i in f:
            file_array.append(i)
    print(file_array)
    return file_array


spectra = []

# iterate through all file
for file in os.listdir(path):
    # Check whether file is in txt format or not
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"

        # call read text file function
        spectra.append(read_file(file_path))

print(spectra)


# pca = PCA(n_components=2)
# pca.fit(spectra)
# PCA(n_components=2)
# print(pca.explained_variance_ratio_)
#
# print(pca.singular_values_)
#
# pca = PCA(n_components=2, svd_solver='full')
# pca.fit(spectra)
# PCA(n_components=2, svd_solver='full')
# print(pca.explained_variance_ratio_)
#
# print(pca.singular_values_)
#
# pca = PCA(n_components=1, svd_solver='arpack')
# pca.fit(spectra)
# PCA(n_components=1, svd_solver='arpack')
# print(pca.explained_variance_ratio_)
#
# print(pca.singular_values_)
