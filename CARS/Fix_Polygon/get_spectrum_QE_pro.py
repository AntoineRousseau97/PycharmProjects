import pandas as pd
from tkinter.filedialog import askopenfile
import csv
import os
import fnmatch
import matplotlib.pyplot as plt

def listNameOfFiles(directory: str, extension="txt") -> list:
    foundFiles = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, f'*.{extension}'):
            foundFiles.append(file)
    return foundFiles

def getFilePaths(directory: str, fileNames: list) -> list:
    filesWithFullPath = []
    for fileName in fileNames:
        filesWithFullPath.append(directory+"/"+fileName)
    return filesWithFullPath

path = "C:/Users/antoi/OneDrive/Desktop/DCC_patch/Stage DCC/DCC_lab/Spectroscopy_Raman/data_27_05_2021/spectres"
donnees_tot_x = []
donnees_tot_y = {}
nb = len(listNameOfFiles(path))

for nom in listNameOfFiles(path):
    # Nom du fichier à importer
    fich = open(path + '/' + nom, "r")
    test_str = list(fich)[14:]
    fich.close()
    x = []
    y = []
    # Nettoyer les informations
    for j in test_str:
        elem_str = j.replace(",", ".").replace("\n", "").replace("\t", ",")
        elem = elem_str.split(",")
        x.append(float(elem[0]))
        y.append(float(elem[1]))

    donnees_tot_x = x
    donnees_tot_y[nom] = y


plt.plot(x, y)
plt.show()