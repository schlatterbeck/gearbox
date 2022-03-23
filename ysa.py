#!/usr/bin/python

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

yfa_table = np.array \
   (( (17,  3.08)
    , (18,  3.025)
    , (19,  2.97)
    , (20,  2.92)
    , (21,  2.87)
    , (22,  2.82)
    , (23,  2.77)
    , (24,  2.745)
    , (25,  2.72)
    , (26,  2.69)
    , (27,  2.67)
    , (28,  2.64)
    , (29,  2.62)
    , (30,  2.59)
    , (35,  2.51)
    , (40,  2.44)
    , (45,  2.4)
    , (50,  2.37)
    , (60,  2.31)
    , (70,  2.275)
    , (80,  2.24)
    , (90,  2.225)
    , (100, 2.21)
    , (200, 2.155)
   ))
yfa_x = yfa_table.T [0]
yfa_y = yfa_table.T [1]


ysa_table = np.array \
   (( (17,  1.57)
    , (18,  1.58)
    , (19,  1.59)
    , (20,  1.6)
    , (25,  1.65)
    , (30,  1.69)
    , (35,  1.725)
    , (40,  1.755)
    , (45,  1.78)
    , (50,  1.8)
    , (60,  1.84)
    , (70,  1.872)
    , (80,  1.895)
    , (90,  1.92)
    , (100, 1.94)
    , (200, 2.06)
   ))
ysa_x = ysa_table.T [0]
ysa_y = ysa_table.T [1]

def yfa (x, a, b, c, d, e) :
    #return a * x + b * x ** 2 / (c * x + d * x ** 2) + e
    #return a ** (x * b) + c * np.log (x * d) + e
    #return a * x ** 3 + b * x ** 2 + c * x + d
    #return a ** (x * b) + c * np.log (x) -d
    #return a * x ** 2 + b * x + c
    return a * x ** 2 + b * (1/x) + c

def ysa (x, a, b, c) :
    #return a * x ** 2 + b * x + c
    return a * np.log (x) + b * x + c

if (1) :
    p, dummy = curve_fit (yfa, yfa_x, yfa_y)

    yy = []
    for x in yfa_x :
        yy.append (yfa (x, *p))
    plt.plot (yfa_x, yfa_y)
    plt.plot (yfa_x, yy)
    plt.show ()
    yy = np.array (yy)
    yy -= yfa_y
    plt.plot (yfa_x, yy)
    plt.show ()

if (1) :
    p, dummy = curve_fit (ysa, ysa_x, ysa_y)

    yy = []
    for x in ysa_x :
        yy.append (ysa (x, *p))
    plt.plot (ysa_x, yy)
    plt.plot (ysa_x, ysa_y)
    plt.show ()
    yy = np.array (yy)
    yy -= ysa_y
    plt.plot (ysa_x, yy)
    plt.show ()
