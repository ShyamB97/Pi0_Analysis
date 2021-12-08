#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:16:29 2021

@author: sb16165
"""

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt


def Save(name="plot", subDirectory="", reference_filename=""):
    """
    Saves the last created plot to file. Run after one the functions below
    ----- Parameters -----
    out                 : file path to save to
    reference_filename  : global variable, is the file name prefix of the plots
    ----------------------
    """
    out = subDirectory + name + reference_filename + ".png"
    plt.savefig(out)
    plt.close()


def Plot(x, y, xlabel="", ylabel="", title="", label="", marker=""):
    """
    Plot line graph.
    """
    plt.plot(x, y, marker=marker, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if label != "": plt.legend()
    plt.tight_layout()


def PlotHist(data, bins=100, xlabel="", title="", label="", alpha=1, sf=2, density=False, newFigure=True):
    """
    Plot histogram of data and axes including bin width.
    ----- Parameters -----
    height      : bin height
    edges       : right edge of the bins
    ----------------------
    """
    if newFigure is True: plt.figure()
    height, edges, _ = plt.hist(data, bins, label=label, alpha=alpha, density=density)
    binWidth = round((edges[-1] - edges[0]) / len(edges), sf)
    plt.ylabel("Number of events (bin width=" + str(binWidth) + ")")
    plt.xlabel(xlabel)
    plt.title(title)
    if label != "": plt.legend()
    plt.tight_layout()
    return height, edges


def PlotHist2D(data_x, data_y, bins=100, x_range=[], y_range=[], xlabel="", ylabel="", title="", label="", newFigure=True):
    """
    Plots two datasets in a 2D histogram.
    """
    if newFigure is True: plt.figure()
    # clamp data_x and data_y given the x range
    if len(x_range) == 2:
        data_y = data_y[data_x >= x_range[0]] # clamp y before x
        data_x = data_x[data_x >= x_range[0]]
        
        data_y = data_y[data_x < x_range[1]]
        data_x = data_x[data_x < x_range[1]]
    
    # clamp data_x and data_y given the y range
    if len(y_range) == 2:
        data_x = data_x[data_y >= y_range[0]] # clamp x before y
        data_y = data_y[data_y >= y_range[0]]
        
        data_x = data_x[data_y < y_range[1]]
        data_y = data_y[data_y < y_range[1]]

    # plot data with a logarithmic color scale
    plt.hist2d(data_x, data_y, 100, norm=matplotlib.colors.LogNorm(), label=label)
    plt.colorbar()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if label != "": plt.legend()
    plt.tight_layout()


def PlotHistComparison(data_1, data_2, bins=100, xlabel="", title="", label_1="", label_2="", alpha=1, sf=2, density=False, newFigure=True):
    """
    Plot two histograms on the same axes, plots larger set first based on the provided bin numbers.
    ----- Parameters -----
    height_1    : bin height
    height_2    : bin height
    edges       : right edge of the bins
    ----------------------
    """

    if newFigure is True: plt.figure()

    c_1 = "C0"
    c_2 = "C1"

    range_1 = max(data_1) - min(data_1)
    range_2 = max(data_2) - min(data_2)

    # data_1 should be bigger than data_2 so histograms aren't cut off
    if range_1 < range_2:
       tmp = data_1
       data_1 = data_2
       data_2 = tmp
       
       tmp = label_1
       label_1 = label_2
       label_2 = tmp
       
       tmp = c_1
       c_1 = c_2
       c_2 = tmp

    height_1, edges, _ = plt.hist(data_1, bins, label=label_1, alpha=alpha, density=density, color=c_1)
    height_2, _, _ = plt.hist(data_2, edges, label=label_2, alpha=alpha, density=density, color=c_2)

    binWidth = round((edges[-1] - edges[0]) / len(edges), sf)
    plt.ylabel("Number of events (bin width=" + str(binWidth) + ")")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return height_1, height_2, edges


def BW(x, A, M, T):
    """
    Breit Wigner distribution.
    ----- Parameters -----
    x   : COM energy (data)
    M   : particle mass
    T   : decay width
    A   : amplitude to scale PDF to data
    ----------------------
    """
    # see https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution for its definition
    gamma = np.sqrt(M**2 * (M**2 + T**2))  # formula is complex, so split it into multiple terms
    k = A * ((2 * 2**0.5)/np.pi) * (M * T * gamma)/((M**2 + gamma)**0.5)
    return k/((x**2 - M**2)**2 + (M*T)**2)


def Gaussian(x, A, mu, sigma):
    """
    Gaussain distribution (not normalised)
    ----- Parameters -----
    x       : sample data
    A       : amplitude to scale
    mu      : mean value
    sigma   : standard deviation
    ----------------------
    """
    return A * np.exp( -0.5 * ((x-mu) / sigma)**2 )


def ChiSqrPDF(x, ndf):
    """
    Chi Squared PDF
    ----- Parameters -----
    x           : sample data
    ndf         : degrees of freedom
    scale       : pdf normalisation?
    poly        : power term
    exponent    : exponential term
    ----------------------
    """
    scale = 1 /( np.power(2, ndf/2) * gamma(ndf/2) )
    poly = np.power(x, ndf - 2)
    exponent = np.exp(- (x**2) / 2)
    return scale * poly * exponent


def LeastSqrFit(data, nbins=25, function=Gaussian, pinit=None, xlabel="", sf=3, interpolation=500, capsize=1):
    """
    fit a function to binned data using the least squares method, implemented in Scipy.
    Plots the fitted function and histogram with y error bars.
    ----- Parameters -----
    hist        : height of each histogram bin
    bins        : data range of each bin
    x           : ceneterd value of each bin
    binWidth    : width of the bins
    uncertainty : poisson uncertainty of each bin
    scale       : normalisation of data for curve fitting
    popt        : paramters of the fitting function which minimises the chi-qsr
    cov         : covarianc matrix of least sqares fit
    ndf         : number of degrees of freedom
    chi_sqr     : chi squared
    x_inter     : interplolated x values of the best fit curve to show the fit in a plot
    y_inter     : interpolated y values
    ----------------------
    """
    data = data[data != -999] # reject null data
    hist, bins = np.histogram(data, nbins) # bin data
    x = (bins[:-1] + bins[1:])/2  # get center of bins
    x = np.array(x, dtype=float) # convert from object to float
    binWidth = bins[1] - bins[0] # calculate bin width

    uncertainty = np.sqrt(hist) # calculate poisson uncertainty if each bin

    # normalise data
    scale = 1 / max(hist)
    uncertainty = uncertainty * scale
    hist = hist * scale
    
    popt, cov = curve_fit(function, x, hist, pinit, uncertainty) # perform least squares curve fit, get the optimal function parameters and covariance matrix

    ndf = nbins - len(popt) # degrees of freedom
    chi_sqr = np.sum( (hist - Gaussian(x, *popt) )**2 / Gaussian(x, *popt) ) # calculate chi squared

    p = quad(ChiSqrPDF, np.sqrt(chi_sqr), np.Infinity, args=(ndf)) # calculate the p value, integrate the chi-qsr function from the chi-qsr to infinity to get p(x > chi-sqr)

    print( "chi_sqaured / ndf: " + str(chi_sqr/ ndf))
    print("p value and compuational error: " + str(p))

    popt[0] = popt[0] / scale
    print("optimised parameters: " + str(popt))

    cov = np.sqrt(cov)  # get standard deviations
    print("uncertainty in optimised parameters: " + str([cov[0, 0], cov[1, 1], cov[2, 2]]))
    
    # calculate plot points for optimised curve
    x_inter = np.linspace(x[0], x[-1], interpolation)  # create x values to draw the best fit curve
    y_inter = function(x_inter, *popt)

    # plot data / fitted curve
    plt.bar(x, hist/scale, binWidth, yerr=uncertainty/scale, capsize=capsize, color="C0")
    Plot(x_inter, y_inter)
    binWidth = round(binWidth, sf)
    plt.ylabel("Number of events (bin width=" + str(binWidth) + ")")
    plt.xlabel(xlabel)
    plt.tight_layout()