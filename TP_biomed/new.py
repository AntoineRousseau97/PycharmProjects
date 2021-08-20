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
def pos_samp():
    file1 = "step_1_320_25_180.csv"
    file2 = "step_1_320_25_181.csv"
    file3 = "step_1_320_25_182.csv"
    file4 = "step_1_320_25_183.csv"
    file5 = "step_1_320_25_184.csv"
    file6 = "step_1_320_25_185.csv"
    file7 = "step_1_320_25_186.csv"
    file8 = "step_1_320_25_187.csv"
    file9 = "step_1_320_25_188.csv"
    file10 = "step_1_320_25_189.csv"
    file11 = "step_1_320_25_190.csv"
    file12 = "step_1_320_25_191.csv"
    file13 = "step_1_320_25_192.csv"
    file14 = "step_1_320_25_193.csv"
    file15 = "step_1_320_25_194.csv"
    file16 = "step_1_320_25_195.csv"
    file17 = "step_1_320_25_196.csv"
    file18 = "step_1_320_25_197.csv"
    file19 = "step_1_320_25_198.csv"
    file20 = "step_1_320_25_199.csv"
    file21 = "step_1_320_25_200.csv"

    newpos1, newerf1, erf_fit1, psf1, freq1, mtf1, pos1, erf1 = analyze_edge(file1, imagevoxelsize=0.25, linelength=None)
    newpos2, newerf2, erf_fit2, psf2, freq2, mtf2, pos2, erf2 = analyze_edge(file2, imagevoxelsize=0.25, linelength=None)
    newpos3, newerf3, erf_fit3, psf3, freq3, mtf3, pos3, erf3 = analyze_edge(file3, imagevoxelsize=0.25, linelength=None)
    newpos4, newerf4, erf_fit4, psf4, freq4, mtf4, pos4, erf4 = analyze_edge(file4, imagevoxelsize=0.25, linelength=None)
    newpos5, newerf5, erf_fit5, psf5, freq5, mtf5, pos5, erf5 = analyze_edge(file5, imagevoxelsize=0.25, linelength=None)
    newpos6, newerf6, erf_fit6, psf6, freq6, mtf6, pos6, erf6 = analyze_edge(file6, imagevoxelsize=0.25, linelength=None)
    newpos7, newerf7, erf_fit7, psf7, freq7, mtf7, pos7, erf7 = analyze_edge(file7, imagevoxelsize=0.25, linelength=None)
    newpos8, newerf8, erf_fit8, psf8, freq8, mtf8, pos8, erf8 = analyze_edge(file8, imagevoxelsize=0.25, linelength=None)
    newpos9, newerf9, erf_fit9, psf9, freq9, mtf9, pos9, erf9 = analyze_edge(file9, imagevoxelsize=0.25, linelength=None)
    newpos10, newerf10, erf_fit10, psf10, freq10, mtf10, pos10, erf10 = analyze_edge(file10, imagevoxelsize=0.25, linelength=None)
    newpos11, newer11, erf_fit11, psf11, freq11, mtf11, pos11, erf11 = analyze_edge(file11, imagevoxelsize=0.25, linelength=None)
    newpos12, newerf12, erf_fit12, psf12, freq12, mtf12, pos12, erf12 = analyze_edge(file12, imagevoxelsize=0.25, linelength=None)
    newpos13, newerf13, erf_fit13, psf13, freq13, mtf13, pos13, erf13 = analyze_edge(file13, imagevoxelsize=0.25, linelength=None)
    newpos14, newerf14, erf_fit14, psf14, freq14, mtf14, pos14, erf14 = analyze_edge(file14, imagevoxelsize=0.25, linelength=None)
    newpos15, newerf15, erf_fit15, psf15, freq15, mtf15, pos15, erf15 = analyze_edge(file15, imagevoxelsize=0.25, linelength=None)
    newpos16, newerf16, erf_fit16, psf16, freq16, mtf16, pos16, erf16 = analyze_edge(file16, imagevoxelsize=0.25, linelength=None)
    newpos17, newerf17, erf_fit17, psf17, freq17, mtf17, pos17, erf17 = analyze_edge(file17, imagevoxelsize=0.25, linelength=None)
    newpos18, newerf18, erf_fit18, psf18, freq18, mtf18, pos18, erf18 = analyze_edge(file18, imagevoxelsize=0.25, linelength=None)
    newpos19, newerf19, erf_fit19, psf19, freq19, mtf19, pos19, erf19 = analyze_edge(file19, imagevoxelsize=0.25, linelength=None)
    newpos20, newerf20, erf_fit20, psf20, freq20, mtf20, pos20, erf20 = analyze_edge(file20, imagevoxelsize=0.25, linelength=None)
    newpos21, newerf21, erf_fit21, psf21, freq21, mtf21, pos21, erf21 = analyze_edge(file21, imagevoxelsize=0.25, linelength=None)


    # MTF plot
    #plt.figure(" mtf")
    plt.plot(freq1, mtf1, "k-")
    plt.plot(freq2, mtf2, "k-")
    plt.plot(freq3, mtf3, "k-")
    plt.plot(freq4, mtf4, "k-")
    plt.plot(freq5, mtf5, "k-")
    plt.plot(freq6, mtf6, "k-")
    plt.plot(freq7, mtf7, "k-")
    plt.plot(freq8, mtf8, "k-")
    plt.plot(freq9, mtf9, "k-")
    plt.plot(freq10, mtf10, "k-")
    plt.plot(freq11, mtf11, "k-")
    plt.plot(freq12, mtf12, "k-")
    plt.plot(freq13, mtf13, "k-")
    plt.plot(freq14, mtf14, "k-")
    plt.plot(freq15, mtf15, "k-")
    plt.plot(freq16, mtf16, "k-")
    plt.plot(freq17, mtf17, "k-")
    plt.plot(freq18, mtf18, "k-")
    plt.plot(freq19, mtf19, "k-")
    plt.plot(freq20, mtf20, "k-")
    plt.plot(0.722, 0.1, 'bo', label="MTF10 min = 0.722 lp/mm")
    plt.plot(0.93, 0.1, 'ro', label="MTF10 max = 0.930 lp/mm")


    plt.legend(loc="best")
    plt.annotate('a)', (0.1, 0.1), size=16)
    plt.xlim(0, 1.2)
    plt.xlabel("Fréquence spatiale [lp/mm]", fontsize=15)
    plt.ylabel("MTF[-]", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    # plt.savefig("nb_proj_mtf.png", dpi=200)



#                                                                                   taille voxel
def same():
    file1 = "step_2_320_25_200.csv"
    file2 = "step_2.1_320_25_200.csv"
    file3 = "step_2.2_320_25_200.csv"
    file4 = "step_2.3_320_25_200.csv"
    file5 = "step_2_320_25_160.csv"
    file6 = "step_2_320_25_180.csv"


    newpos1, newerf1, erf_fit1, psf1, freq1, mtf1, pos1, erf1 = analyze_edge(file1, imagevoxelsize=0.25, linelength=None)
    newpos2, newerf2, erf_fit2, psf2, freq2, mtf2, pos2, erf2 = analyze_edge(file2, imagevoxelsize=0.25, linelength=None)
    newpos3, newerf3, erf_fit3, psf3, freq3, mtf3, pos3, erf3 = analyze_edge(file3, imagevoxelsize=0.25, linelength=None)
    newpos4, newerf4, erf_fit4, psf4, freq4, mtf4, pos4, erf4 = analyze_edge(file4, imagevoxelsize=0.25, linelength=None)
    newpos5, newerf5, erf_fit5, psf5, freq5, mtf5, pos5, erf5 = analyze_edge(file5, imagevoxelsize=0.25, linelength=None)
    newpos6, newerf6, erf_fit6, psf6, freq6, mtf6, pos6, erf6 = analyze_edge(file6, imagevoxelsize=0.25, linelength=None)

    # MTF plot
    #plt.figure(" mtf")
    plt.plot(freq1, mtf1, 'k-')
    plt.plot(freq2, mtf2, 'k-')
    plt.plot(freq3, mtf3, 'k-')
    plt.plot(freq4, mtf4, "k-")
    plt.plot(0.4, 0.1, 'bo', label="MTF10 min = 0.400 lp/mm")
    plt.plot(0.453, 0.1, 'ro', label="MTF10 max = 0.453 lp/mm")

    #plt.plot(freq5, mtf5, 'b-', label='depth=160')
    #plt.plot(freq6, mtf6, "r-", label='depth=180')


    plt.legend(loc="best")
    plt.xlim(0, 0.8)
    plt.xlabel("Fréquence spatiale [lp/mm]", fontsize=15)
    plt.ylabel("MTF[-]", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    #plt.savefig("taille_vox_mtf.png", dpi=200)



#                                                                                   type éclairage
def angle():
    file1 = "step_3_320_25_160.csv"
    file2 = "step_3_320_25_180.csv"
    file3 = "step_3_320_25_200.csv"
    file4 = "step_3_45deg_320_25_180.csv"
    file5 = "step_3.1_45deg_320_25_180.csv"

    newpos1, newerf1, erf_fit1, psf1, freq1, mtf1, pos1, erf1 = analyze_edge(file1, imagevoxelsize=0.25, linelength=None)
    newpos2, newerf2, erf_fit2, psf2, freq2, mtf2, pos2, erf2 = analyze_edge(file2, imagevoxelsize=0.25, linelength=None)
    newpos3, newerf3, erf_fit3, psf3, freq3, mtf3, pos3, erf3 = analyze_edge(file3, imagevoxelsize=0.25, linelength=None)
    newpos4, newerf4, erf_fit4, psf4, freq4, mtf4, pos4, erf4 = analyze_edge(file4, imagevoxelsize=0.25, linelength=None)
    newpos5, newerf5, erf_fit5, psf5, freq5, mtf5, pos5, erf5 = analyze_edge(file5, imagevoxelsize=0.25, linelength=None)

    # MTF plot
    #plt.figure(" mtf")

    plt.plot(freq1, mtf1, "k-")
    plt.plot(freq2, mtf2, "k-")
    plt.plot(freq3, mtf3, "k-")

    plt.plot(freq4, mtf4, "b-", label="45deg")
    plt.plot(freq5, mtf5, 'r-', label='45deg')

    plt.legend(loc="best")
    plt.annotate('a)', (0.1, 0.1), size=16)
    plt.xlim(0, 1.2)
    plt.xlabel("Fréquence spatiale [lp/mm]", fontsize=15)
    plt.ylabel("MTF[-]", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    #plt.savefig("type_ecl_mtf.png", dpi=200)

def diff_edge():

    file1 = "step_1_320_25_180.csv"
    file2 = "step_2_320_25_180.csv"
    file3 = "step_3_320_25_180.csv"
    file4 = "step_2.1_320_25_200.csv"
    file5 = "step_2.2_320_25_200.csv"
    file6 = "step_2.3_320_25_200.csv"


    newpos1, newerf1, erf_fit1, psf1, freq1, mtf1, pos1, erf1 = analyze_edge(file1, imagevoxelsize=0.25, linelength=None)
    newpos2, newerf2, erf_fit2, psf2, freq2, mtf2, pos2, erf2 = analyze_edge(file2, imagevoxelsize=0.25, linelength=None)
    newpos3, newerf3, erf_fit3, psf3, freq3, mtf3, pos3, erf3 = analyze_edge(file3, imagevoxelsize=0.25, linelength=None)
    newpos4, newerf4, erf_fit4, psf4, freq4, mtf4, pos4, erf4 = analyze_edge(file4, imagevoxelsize=0.25, linelength=None)
    newpos5, newerf5, erf_fit5, psf5, freq5, mtf5, pos5, erf5 = analyze_edge(file5, imagevoxelsize=0.25, linelength=None)
    newpos6, newerf6, erf_fit6, psf6, freq6, mtf6, pos6, erf6 = analyze_edge(file6, imagevoxelsize=0.25, linelength=None)


    # MTF plot
    # plt.figure(" mtf")

    plt.plot(freq1, mtf1, "k-")
    plt.plot(freq2, mtf2, "b-.")
    plt.plot(freq4, mtf4, "b-.")
    plt.plot(freq5, mtf5, "b-.")
    plt.plot(freq6, mtf6, "b-.")
    plt.plot(freq3, mtf3, "r--")
    plt.plot(0.803, 0.1, 'ko', label="MTF10 edge 1 = 0.803 lp/mm")
    plt.plot(0.461, 0.1, 'bo', label="MTF10 edge 2 max = 0.461 lp/mm")
    plt.plot(0.4, 0.1, 'bs', label="MTF10 edge 2 min = 0.400 lp/mm")
    plt.plot(0.759, 0.1, 'ro', label="MTF10 edge 3 = 0.759 lp/mm")

    plt.legend(loc="best")
    plt.annotate('b)', (0.1, 0.1), size=16)
    plt.xlim(0, 1.2)
    plt.xlabel("Fréquence spatiale [lp/mm]", fontsize=15)
    plt.ylabel("MTF[-]", fontsize=15)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
    # plt.savefig("type_ecl_mtf.png", dpi=200)

print(pos_samp(), diff_edge())


