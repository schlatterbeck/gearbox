#!/usr/bin/python

from rsclib.autosuper import autosuper
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

class Fit_Curve (autosuper) :
    def __init__ (self) :
        self._fit = staticmethod (self.__class__._fit).__get__ (autosuper)
    # end def __init__

    def _fitted_function (self, x, *params) :
        if not params :
            params = self.fit_params
        return self._fit (x, *params)
    # end def _fitted_function
# end class Fit_Curve

class Callable_Fit_Curve (Fit_Curve) :
    def __call__ (self, x, *params) :
        return self._fitted_function (x, *params)
    # end def __call__
# end class Callable_Fit_Curve

class YFA (Callable_Fit_Curve) :
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

    def _fit (x, a, b, c) :
        return a * x ** 2 + b * (1/x) + c
    # end def _fit

    fit_params, dummy = curve_fit (_fit, yfa_x, yfa_y)

# end class YFA

class YSA (Callable_Fit_Curve) :
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

    def _fit (x, a, b, c) :
        return a * np.log (x) + b * x + c
    # end def _fit

    fit_params, dummy = curve_fit (_fit, ysa_x, ysa_y)

# end class YSA

yfa = YFA ()
ysa = YSA ()

if __name__ == '__main__' :

    yy = []
    for x in yfa.yfa_x :
        yy.append (yfa (x))
    plt.plot (yfa.yfa_x, yfa.yfa_y)
    plt.plot (yfa.yfa_x, yy)
    plt.show ()
    yy = np.array (yy)
    yy -= yfa.yfa_y
    plt.plot (yfa.yfa_x, yy)
    plt.show ()

    yy = []
    for x in ysa.ysa_x :
        yy.append (ysa (x))
    plt.plot (ysa.ysa_x, yy)
    plt.plot (ysa.ysa_x, ysa.ysa_y)
    plt.show ()
    yy = np.array (yy)
    yy -= ysa.ysa_y
    plt.plot (ysa.ysa_x, yy)
    plt.show ()
