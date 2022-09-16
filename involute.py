#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from rsclib.autosuper import autosuper
from argparse import ArgumentParser
from math import gcd, atan, acos
from bisect import bisect_right
import pga
import sys

# Various Inverse Involute implementations using the following papers:
# [1] Alberto López Rosado, Federico Prieto Muñoz, and Roberto Alvarez
#     Fernández. An analytic expression for the inverse involute.
#     Mathematical Problems in Engineering, 2019(3586012), September 2019.
# [2] Harry H. Cheng. Derivation of the explicit solution of the inverse
#     involute function and its applications in gear tooth geometry
#     calculations. Journal of Applied Mechanisms & Robotics,
#     3(2):13–23, April 1996.

class Inverse_Involute (autosuper) :
    """ Base class: We use a callable object that computes the inverse
        involtion. This allows us to store pre-computed tables etc.
        We also store the maximum degree for which the inverse involute
        has reasonable error.
    """

    max_degree = 45.0

    def __init__ (self) :
        self.err_range = np.arange (0, self.max_degree, .021)
    # end def __init__

    def plot_error (self, reference = None, absolute = False) :
        y = []
        for a in self.err_range :
            b = a / 180 * np.pi
            invol = np.tan (b) - b
            ref = b
            refname = 'Involute'
            if reference:
                ref = reference (invol)
                refname = reference.name
            dif = self (invol) - ref
            if absolute or ref == 0 :
                y.append (dif)
            else :
                y.append (dif / ref)
        fig = plt.figure ()
        ax  = fig.add_subplot (1, 1, 1)
        ax.plot (self.err_range, y)
        ax.set_title (self.name + ' ref: ' + refname)
        plt.show ()
    # end def plot_error

# end class Inverse_Involute

class Inverse_Involute_Liu (Inverse_Involute) :
    """ Corrected version by Liu cited by Rosado et. al.
        The version in Rosado et.al. has an error where they use
        x ** (5/3 instead of x ** (8 / 3) in the denominator
    """
    name = 'Inverse Involute Liu'
    def __call__ (self, x) :
        if x == 0 :
            return 0
        x3 = x  ** (1/3)
        x8 = x  ** (8/5)
        k3 = 3  ** (1/3)
        a  = atan (k3 * x3 + 3/5 * x + 1/11 * x8)
        return acos (np.sin (a) / (x + a))
    # end def __call__
# end class Inverse_Involute_Liu

class Inverse_Involute_Cheng (Inverse_Involute) :
    name = 'Inverse Involute Cheng'
    def __call__ (self, x) :
        k3 = 3 ** (1/3)
        return \
            ( k3 * x ** (1/3)
            - 2/5 * x
            + 9/175 * k3 * k3 * x ** (5/3) 
            - 2/175 * k3 * x ** (7/3)
            - 144/67375 * x ** 3
            + 3258/3128125 * k3 * k3 * x ** (11/3)
            - 49711/153278125 * k3 * x ** (13/3)
            - 1130112/9306171875 * x ** 5
            + 5169659643/95304506171875 * k3 * k3 * x ** (17/3)
            )
    # end def __call__
# end class Inverse_Involute_Cheng

class Inverse_Involute_Lookup (Inverse_Involute) :
    """ Perform linear interpolation of involute_table to
        reverse tan(x) - x to x
    """
    name = 'Inverse Involute with lookup table'

    # Pre-computations for involute lookup version
    poly           = 12
    istep          = 1000
    maxangle       = 46 / 180 * np.pi
    scaled_angle   = maxangle ** (1 / poly)
    x_table        = (np.arange (0, scaled_angle, scaled_angle / istep))
    x_table        = x_table ** poly
    involute_table = np.tan (x_table) - x_table

    def __call__ (self, x) :
        idx = bisect_right (self.involute_table, x)
        assert 0 < idx <= len (self.involute_table)
        idx -= 1
        if self.involute_table [idx] < x :
            assert idx + 1 < len (self.involute_table)
            inv_l = self.involute_table [idx]
            inv_r = self.involute_table [idx + 1]
            x1 = self.x_table [idx]
            x2 = self.x_table [idx + 1]
            factor = (x - inv_l) / (inv_r - inv_l)
            return x1 + (x2 - x1) * factor
        else :
            assert self.involute_table [idx] == x
            return self.x_table [idx]
    # def __call__
# end class Inverse_Involute_Lookup

class Inverse_Involute_Apsol4 (Inverse_Involute) :
    """ This is "Apsol4" from [1]
        Note that this is accurate only to about 37°.
    """
    max_degree = 37.6
    name = 'Inverse Involute Apsol4'
    def __call__ (self, x) :
        x3 = x ** (1/3)
        return x3 / (0.69336473 + (-0.0000654976 + 0.1926063 * x3) * x3)
    # end def __call__
# end class Inverse_Involute_Apsol4

class Inverse_Involute_Apsol5 (Inverse_Involute) :
    """ This is "Apsol5" from [1]
        Note that this is accurate only to about 37°.
        It is slightly less accurate than Apsol4.
    """
    max_degree = 37.6
    name = 'Inverse Involute Apsol5'
    def __call__ (self, x) :
        x3 = x ** (1/3)
        return x3 / (0.693357 + 0.192848 * x3 * x3)
    # end def __call__
# end class Inverse_Involute

inv_involute_liu    = Inverse_Involute_Liu ()
inv_involute_cheng  = Inverse_Involute_Cheng ()
inv_involute_lookup = Inverse_Involute_Lookup ()
inv_involute_apsol4 = Inverse_Involute_Apsol4 ()
inv_involute_apsol5 = Inverse_Involute_Apsol5 ()

if __name__ == '__main__' :
    impl = ('cheng', 'liu', 'lookup', 'apsol4', 'apsol5')
    cmd  = ArgumentParser ()
    cmd.add_argument \
        ( '--absolute-error'
        , help    = "When plotting errors, use absolute error (default "
                    "is relative error)"
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '--reference', '--error-reference'
        , help    = "Error reference can be empty or 'liu' or 'cheng'"
        )
    cmd.add_argument \
        ( '--test-implementation'
        , help    = "Implementation for which we want to plot errors, "
                    "One of %s default=%%(default)s" % ', '.join (impl)
        , default = "lookup"
        )
    args = cmd.parse_args ()
    under_test = globals () ['inv_involute_%s' % args.test_implementation]
    reference = None
    if args.reference:
        reference  = globals () ['inv_involute_%s' % args.reference]
    under_test.plot_error (reference, absolute = args.absolute_error)
