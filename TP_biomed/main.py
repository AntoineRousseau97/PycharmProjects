# -*- coding: utf-8 -*-
"""
%This script will implement the edge-response function's curve fitting
%algorithm given a csv file, plot the fitted curve, and calcualte the
%resulting point spread function and modulation transfer function

This script is based on the MATLAB script Analyze_Edge.m modified by Olivier Fillion at Université Laval in 2015.


Created on Mon Mar 20 15:35:58 2017

@author: Pascal Paradis

provided as is
"""

###############################################################################
# Libraries imports
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.special import erf
from scipy.optimize import curve_fit, fsolve
import glob
import re
import os

###############################################################################
# functions
###############################################################################
def varerf(x, a, b, c, d):
    """This function returns an ERF. It is used by analyze_edge in the ERF
    fitting.
    """
    return a*erf(x*b - c) + d

def analyze_edge(file, imagevoxelsize=0.25, linelength=None):
    """This function imports the data of an edge profile from a .csv file
    created by the DeskCat software. It fits an ideal ERF on the raw data and
    then, computes the PSF and MTF. It plots all those 3 functions and saves
    them in png files whose names are derived from the original data file.

    Inputs:
        file =              file name
        imagevoxelsize =    voxelsize as set up in the parameters of the
                            reconstruction
        linelength =        the length of the line in the projection viewer if
                            the profile is taken on a 2D image

    Outputs:
        All the outputs are comprised of the x and y axis data

        orig =      Original profile data
        erf_fit =   ERF fitted on the original profile data
        psf =       PSF based on the ERF fit
        mtf =       MTF based on the PSF
    """
    #importing data and making sure that the decimal separators are "." instead of ","
    # the next 6 lines is the equivalent of ImprtData
    with open(file) as f:
        text = f.read()
    text = text.replace(",", ".")
    with open(file, "w") as f:
        f.write(text)
    data = np.genfromtxt(file, delimiter=". ", skip_header=1)
    pos = data[:,0]
    erf = data[:,1]
    # transforming the pixel number in a spatial position based on the
    # imagevoxelsize or the linelength if a linelength is provided
    if linelength:
        pos *= linelength/pos[-1]
        voxelsize = pos[1] - pos[0]
    else:
        pos *= imagevoxelsize
        voxelsize = pos[1] - pos[0]
    # normalizing the data
    erf /= np.max(erf)
    # reverting the erf vector so it goes up along the position
    if erf[int(len(erf)/4)] > erf[int(3*len(erf)/4)]:
        erf = np.flipud(erf)
    # eliminates the artefact that is present when the profile is drawn
    # outside of the phantom
    if erf[0] > 0.5:
        erf = erf[np.where(erf < 0.5)[0][0]:-1]
        pos = pos[np.where(erf < 0.5)[0]:-1]
    # arbitraty clipping of the profile at the beginning and at the end so
    # the ERF is centered on the x axis
    newend = int(np.where(erf > 0.6)[0][0] + 2*len(erf)/5)
    newstart = int(np.where(erf < 0.4)[0][-1] - 2*len(erf)/5)
    newerf = erf[newstart:newend]
    newpos = pos[newstart:newend]
    # Fitting the ideal ERF on the raw data
    popt, pcov = curve_fit(varerf, newpos, newerf,
                           [0.5,1,newpos[int(len(newpos)/2)],0.5])
    erf_fit = varerf(newpos, popt[0], popt[1], popt[2], popt[3])
    # Creating a spline on the ERF fit in order to get a numerical derivative
    # on the same x axis as the ERF
    erf_us = UnivariateSpline(newpos, erf_fit, k=4, s=0)
    psf = erf_us.derivative(1)(newpos)
    # Normalizing the PSF
    psf /= np.max(psf)
    # Computing the MTF from the PSF
    mtf = np.abs(np.fft.fftshift(np.fft.fft(psf)))
    mtf /= np.max(mtf)
    freq = np.fft.fftshift(np.fft.fftfreq(len(psf), d=voxelsize))
    return newpos, newerf, erf_fit, psf, freq, mtf, pos, erf


#                                                                         nb de projections
def nb_proj():
    file1 = "partie5_modeCT40_25_centre.csv"
    file2 = "partie5_modeCT40_25_dessous.csv"
    file3 = "partie5_modeCT40_25_dessus.csv"
    file4 = "partie5_modeCT160_25_centre.csv"
    file5 = "partie5_modeCT160_25_dessous.csv"
    file6 = "partie5_modeCT160_25_dessus.csv"
    file7 = "partie5_modeCT320_25_centre.csv"
    file8 = "partie5_modeCT320_25_dessous.csv"
    file9 = "partie5_modeCT320_25_dessus.csv"

    newpos1, newerf1, erf_fit1, psf1, freq1, mtf1, pos1, erf1 = analyze_edge(file1, imagevoxelsize=0.25, linelength=None)
    newpos2, newerf2, erf_fit2, psf2, freq2, mtf2, pos2, erf2 = analyze_edge(file2, imagevoxelsize=0.25, linelength=None)
    newpos3, newerf3, erf_fit3, psf3, freq3, mtf3, pos3, erf3 = analyze_edge(file3, imagevoxelsize=0.25, linelength=None)
    newpos4, newerf4, erf_fit4, psf4, freq4, mtf4, pos4, erf4 = analyze_edge(file4, imagevoxelsize=0.25, linelength=None)
    newpos5, newerf5, erf_fit5, psf5, freq5, mtf5, pos5, erf5 = analyze_edge(file5, imagevoxelsize=0.25, linelength=None)
    newpos6, newerf6, erf_fit6, psf6, freq6, mtf6, pos6, erf6 = analyze_edge(file6, imagevoxelsize=0.25, linelength=None)
    newpos7, newerf7, erf_fit7, psf7, freq7, mtf7, pos7, erf7 = analyze_edge(file7, imagevoxelsize=0.25, linelength=None)
    newpos8, newerf8, erf_fit8, psf8, freq8, mtf8, pos8, erf8 = analyze_edge(file8, imagevoxelsize=0.25, linelength=None)
    newpos9, newerf9, erf_fit9, psf9, freq9, mtf9, pos9, erf9 = analyze_edge(file9, imagevoxelsize=0.25, linelength=None)

    # MTF plot
    #plt.figure(" mtf")
    plt.plot(freq1, mtf1, "b-.", label="40 projections")
    plt.plot(freq2, mtf2, "b-.")
    plt.plot(freq3, mtf3, "b-.")
    plt.plot(0.913, 0.1, 'bo', label="Fréquence au point MTF10 = 0.913 lp/mm")

    plt.plot(freq4, mtf4, "k-", label="160 projections")
    plt.plot(freq5, mtf5, "k-")
    plt.plot(freq6, mtf6, "k-")
    plt.plot(0.553, 0.1, 'ks', label="Fréquence au point MTF10 = 0.553 lp/mm")

    plt.plot(freq7, mtf7, "r--", label="320 projections")
    plt.plot(freq8, mtf8, "r--")
    plt.plot(freq9, mtf9, "r--")
    plt.plot(0.553, 0.1, 'rs', label="Fréquence au point MTF10 = 0.553 lp/mm")

    plt.legend(loc="best")
    plt.annotate('a)', (0.1, 0.1), size=16)
    plt.xlim(0, 1.2)
    plt.xlabel("Fréquence spatiale [lp/mm]", fontsize=15)
    plt.ylabel("MTF[-]", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    # plt.savefig("nb_proj_mtf.png", dpi=200)
"""
    # ERF plot
    plt.figure("erf")
    plt.plot(pos1, erf1, label="40 proj", color="blue")
    plt.plot(newpos1, erf_fit1, color="blue", ls="--")
    plt.plot(pos2, erf2, color="blue")
    plt.plot(newpos2, erf_fit2, color="blue", ls="--")
    plt.plot(pos3, erf3, color="blue")
    plt.plot(newpos3, erf_fit3, color="blue", ls="--")


    plt.plot(pos4, erf4, label="160 proj", color="red")
    plt.plot(newpos4, erf_fit4, color="red", ls="--")
    plt.plot(pos5, erf5, color="red")
    plt.plot(newpos5, erf_fit5, color="red", ls="--")
    plt.plot(pos6, erf6, color="red")
    plt.plot(newpos6, erf_fit6,color="red", ls="--")


    plt.plot(pos7, erf7, label="320 proj", color="black")
    plt.plot(newpos7, erf_fit7, color="black", ls="--")
    plt.plot(pos8, erf8, color="black")
    plt.plot(newpos8, erf_fit8, color="black", ls="--")
    plt.plot(pos9, erf9, color="black")
    plt.plot(newpos9, erf_fit9,color="black", ls="--")


    plt.legend(loc="best")
    plt.xlabel("Position $x$ [mm]", fontsize=24)
    plt.ylabel("ERF", fontsize=24)
    plt.tick_params(labelsize=20)
    plt.margins(0.05)
    #plt.tight_layout()
    plt.show()
    # plt.savefig("erf_" + fname[:-4] + ".png", dpi=200)

    # PSF plot
    plt.figure(" psf")
    plt.plot(newpos1, psf1, label="40 proj", color="blue")
    plt.plot(newpos2, psf2, color="blue")
    plt.plot(newpos3, psf3, color="blue")


    plt.plot(newpos4, psf4, label="160 proj", color="red")
    plt.plot(newpos5, psf5, color="red")
    plt.plot(newpos6, psf6, color="red")


    plt.plot(newpos7, psf7, label="320 proj", color="black")
    plt.plot(newpos8, psf8, color="black")
    plt.plot(newpos9, psf9, color="black")


    plt.legend(loc="best")
    plt.xlabel("Position $x$ [mm]", fontsize=24)
    plt.ylabel("PSF", fontsize=24)
    plt.tick_params(labelsize=20)
    plt.margins(0.05)
    #plt.tight_layout()
    plt.show()
    # plt.savefig(fname[:-4] + "_psf.png", dpi=200)
    """


#                                                                                   taille voxel
def t_vox():
    file1 = "partie5_modeCT320_200_centre.csv"
    file2 = "partie5_modeCT320_200_dessous.csv"
    file3 = "partie5_modeCT320_200_dessus.csv"
    file4 = "partie5_modeCT320_50_centre.csv"
    file5 = "partie5_modeCT320_50_dessous.csv"
    file6 = "partie5_modeCT320_50_dessus.csv"
    file7 = "partie5_modeCT320_25_centre.csv"
    file8 = "partie5_modeCT320_25_dessous.csv"
    file9 = "partie5_modeCT320_25_dessus.csv"


    newpos1, newerf1, erf_fit1, psf1, freq1, mtf1, pos1, erf1 = analyze_edge(file1, imagevoxelsize=2, linelength=None)
    newpos2, newerf2, erf_fit2, psf2, freq2, mtf2, pos2, erf2 = analyze_edge(file2, imagevoxelsize=2, linelength=None)
    newpos3, newerf3, erf_fit3, psf3, freq3, mtf3, pos3, erf3 = analyze_edge(file3, imagevoxelsize=2, linelength=None)
    newpos4, newerf4, erf_fit4, psf4, freq4, mtf4, pos4, erf4 = analyze_edge(file4, imagevoxelsize=0.5, linelength=None)
    newpos5, newerf5, erf_fit5, psf5, freq5, mtf5, pos5, erf5 = analyze_edge(file5, imagevoxelsize=0.5, linelength=None)
    newpos6, newerf6, erf_fit6, psf6, freq6, mtf6, pos6, erf6 = analyze_edge(file6, imagevoxelsize=0.5, linelength=None)
    newpos7, newerf7, erf_fit7, psf7, freq7, mtf7, pos7, erf7 = analyze_edge(file7, imagevoxelsize=0.25, linelength=None)
    newpos8, newerf8, erf_fit8, psf8, freq8, mtf8, pos8, erf8 = analyze_edge(file8, imagevoxelsize=0.25, linelength=None)
    newpos9, newerf9, erf_fit9, psf9, freq9, mtf9, pos9, erf9 = analyze_edge(file9, imagevoxelsize=0.25, linelength=None)

    # MTF plot
    #plt.figure(" mtf")
    plt.plot(freq1, mtf1, 'b-.', label="taille de voxel = 2mm")
    plt.plot(freq2, mtf2, 'b-.')
    plt.plot(freq3, mtf3, 'b-.')
    plt.plot(0.206, 0.1, 'bo', label="Fréquence au point MTF10 = 0.206 lp/mm")

    plt.plot(freq4, mtf4, "k-", label="taille de voxel = 0.5mm")
    plt.plot(freq5, mtf5, "k-")
    plt.plot(freq6, mtf6, "k-")
    plt.plot(0.461, 0.1, 'k^', label="Fréquence au point MTF10 = 0.461 lp/mm")

    plt.plot(freq7, mtf7, "r--", label="taille de voxel = 0.25mm")
    plt.plot(freq8, mtf8, "r--")
    plt.plot(freq9, mtf9, "r--")
    plt.plot(0.553, 0.1, 'rs', label="Fréquence au point MTF10 = 0.553 lp/mm")

    plt.legend(loc="best")
    plt.annotate('b)', (0.1, 0.1), size=16)
    plt.xlim(0, 1.2)
    plt.xlabel("Fréquence spatiale [lp/mm]", fontsize=15)
    plt.ylabel("MTF[-]", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    #plt.savefig("taille_vox_mtf.png", dpi=200)

"""    
    # ERF plot
    plt.figure("erf")
    plt.plot(pos1, erf1, label="taille = 2", color="blue")
    plt.plot(newpos1, erf_fit1, color="blue", ls="--")
    plt.plot(pos2, erf2, color="blue")
    plt.plot(newpos2, erf_fit2, color="blue", ls="--")
    plt.plot(pos3, erf3, color="blue")
    plt.plot(newpos3, erf_fit3, color="blue", ls="--")


    plt.plot(pos4, erf4, label="taille = 0.50", color="red")
    plt.plot(newpos4, erf_fit4, color="red", ls="--")
    plt.plot(pos5, erf5, color="red")
    plt.plot(newpos5, erf_fit5, color="red", ls="--")
    plt.plot(pos6, erf6, color="red")
    plt.plot(newpos6, erf_fit6,color="red", ls="--")


    plt.plot(pos7, erf7, label="taille = 0.25", color="black")
    plt.plot(newpos7, erf_fit7, color="black", ls="--")
    plt.plot(pos8, erf8, color="black")
    plt.plot(newpos8, erf_fit8, color="black", ls="--")
    plt.plot(pos9, erf9, color="black")
    plt.plot(newpos9, erf_fit9,color="black", ls="--")


    plt.legend(loc="best")
    plt.xlabel("Position $x$ [mm]", fontsize=24)
    plt.ylabel("ERF", fontsize=24)
    plt.tick_params(labelsize=20)
    plt.margins(0.05)
    #plt.tight_layout()
    #plt.show()
    # plt.savefig("erf_" + fname[:-4] + ".png", dpi=200)

    # PSF plot
    plt.figure(" psf")
    plt.plot(newpos1, psf1, label="taille = 2", color="blue")
    plt.plot(newpos2, psf2, color="blue")
    plt.plot(newpos3, psf3, color="blue")


    plt.plot(newpos4, psf4, label="taille = 0.5", color="red")
    plt.plot(newpos5, psf5, color="red")
    plt.plot(newpos6, psf6, color="red")


    plt.plot(newpos7, psf7, label="taille = 0.25", color="black")
    plt.plot(newpos8, psf8, color="black")
    plt.plot(newpos9, psf9, color="black")


    plt.legend(loc="best")
    plt.xlabel("Position $x$ [mm]", fontsize=24)
    plt.ylabel("PSF", fontsize=24)
    plt.tick_params(labelsize=20)
    plt.margins(0.05)
    #plt.tight_layout()
    #plt.show()
    # plt.savefig(fname[:-4] + "_psf.png", dpi=200)
    """





#                                                                                   type éclairage
def t_ecl():
    file1 = "partie5_modeCT320_25_centre.csv"
    file2 = "partie5_modeCT320_25_eventail15.csv"
    file3 = "partie5_modeCT320_25_eventail05.csv"

    newpos1, newerf1, erf_fit1, psf1, freq1, mtf1, pos1, erf1 = analyze_edge(file1, imagevoxelsize=0.25, linelength=None)
    newpos2, newerf2, erf_fit2, psf2, freq2, mtf2, pos2, erf2 = analyze_edge(file2, imagevoxelsize=0.25, linelength=None)
    newpos3, newerf3, erf_fit3, psf3, freq3, mtf3, pos3, erf3 = analyze_edge(file3, imagevoxelsize=0.25, linelength=None)

    # MTF plot
    #plt.figure(" mtf")

    plt.plot(freq1, mtf1, "k-")
    plt.plot(0.553, 0.1, 'ko', label="Fréquence au point MTF10 = 0.553 lp/mm")

    plt.plot(freq2, mtf2, "b--", label="éclairage avec éventail 1.5cm")
    plt.plot(0.818, 0.1, 'b^', label="Fréquence au point MTF10 = 0.818 lp/mm")

    plt.plot(freq3, mtf3, "r-.", label="éclairage avec éventail 0.5cm")
    plt.plot(0.632, 0.1, 'rs', label="Fréquence au point MTF10 = 0.632 lp/mm")

    plt.legend(loc="best")
    plt.annotate('c)', (0.1, 0.1), size=16)
    plt.xlim(0, 1.2)
    plt.xlabel("Fréquence spatiale [lp/mm]", fontsize=15)
    plt.ylabel("MTF[-]", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    #plt.savefig("type_ecl_mtf.png", dpi=200)


    # ERF plot
    plt.figure("erf")
    plt.plot(pos1, erf1, label="éclairage cônique", color="black")
    plt.plot(newpos1, erf_fit1, color="black", ls="--")

    #plt.plot(pos2, erf2, label="éclairage avec éventail 1.5cm", color="red")
    #plt.plot(newpos2, erf_fit2, color="red", ls="--")

    #plt.plot(pos3, erf3, label="éclairage avec éventail 0.5cm", color="black")
    #plt.plot(newpos3, erf_fit3, color="black", ls="--")

    #plt.legend(loc="best")
    plt.xlabel("Position $x$ [mm]", fontsize=15)
    plt.ylabel("ERF", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.margins(0.05)
    plt.tight_layout()
    plt.show()
    # plt.savefig("erf_" + fname[:-4] + ".png", dpi=200)

    # PSF plot
    plt.figure(" psf")
    plt.plot(newpos1, psf1, label="éclairage conique", color="black")

    #plt.plot(newpos2, psf2, label="éventail 1.5cm", color="red")

    #plt.plot(newpos3, psf3, label="éventail 0.5cm", color="black")

    #plt.legend(loc="best")
    plt.xlabel("Position $x$ [mm]", fontsize=15)
    plt.ylabel("PSF", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.margins(0.05)
    plt.tight_layout()
    plt.show()
    # plt.savefig(fname[:-4] + "_psf.png", dpi=200)


print(nb_proj(), t_vox(), t_ecl())

