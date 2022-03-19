#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from rsclib.autosuper import autosuper
from argparse import ArgumentParser
from math import gcd, atan, acos
from bisect import bisect_right
import pga
import sys

def inv_involute_slow (x) :
    x3 = x  ** (1/3)
    x8 = x3 ** 8
    k3 = 3  ** (1/3)
    return acos \
        ( np.sin (atan (k3 * x3 + 3/5 * x + (1/11 * x8)))
        / (x + atan (k3 * x3 + 3/5 * x + 1/11 * x8))
        )
# end def inv_involute_slow

# Pre-computations for involute lookup version
istep = 1000
maxangle = 46 / 180 * np.pi
involute_table = [np.tan (x) - x for x in np.arange (0, maxangle, 1/istep)]

def inv_involute_lookup (x) :
    """ Perform linear interpolation of involute_table to
        reverse tan(x) - x to x
    """
    idx = bisect_right (involute_table, x)
    assert 0 < idx <= len (involute_table)
    idx -= 1
    if involute_table [idx] < x :
        assert idx + 1 < len (involute_table)
        inv_l = involute_table [idx]
        inv_r = involute_table [idx + 1]
        x1 = idx / istep
        x2 = (idx + 1) / istep
        factor = (x - inv_l) / (inv_r - inv_l)
        return x1 + (x2 - x1) * factor
    else :
        assert involute_table [idx] == x
        return idx / istep
# def inv_involute_lookup

def inv_involute_apsol4 (x) :
    """ This is "Apsol4" from Alberto López Rosado, Federico Prieto
        Muñoz, and Roberto Alvarez Fernández. An analytic expression for
        the inverse involute. Mathematical Problems in Engineering,
        2019(3586012), September 2019.
        Note that this is accurate only to about 37°.
    """
    x3 = x ** (1/3)
    return x3 / (0.69336473 + (-0.0000654976 + 0.1926063 * x3) * x3)
# end def inv_involute_apsol4

def inv_involute_apsol5 (x) :
    """ This is "Apsol5" from Alberto López Rosado, Federico Prieto
        Muñoz, and Roberto Alvarez Fernández. An analytic expression for
        the inverse involute. Mathematical Problems in Engineering,
        2019(3586012), September 2019.
        Note that this is accurate only to about 37°.
    """
    x3 = x ** (1/3)
    return x3 / (0.693357 + 0.192848 * x3 * x3)
# end def inv_involute_apsol5

inv_involute = inv_involute_apsol4

def plot_error (inv_version, absolute = False) :
    x = np.arange (0, 37.6, 0.03)
    y = []
    for a in x :
        b = a / 180 * np.pi
        invol = np.tan (b) - b
        sl = inv_involute_slow (invol)
        if absolute :
            y.append (inv_version (invol) - sl)
        else :
            y.append ((inv_version (invol) - sl) / sl)
    fig = plt.figure ()
    ax  = fig.add_subplot (1, 1, 1)
    ax.plot (x, y)
    plt.show ()
# end def plot_error

def plot_error_apsol4 (absolute) :
    plot_error (inv_involute_apsol4, absolute)
# end def plot_error_apsol4

def plot_error_apsol5 (absolute) :
    plot_error (inv_involute_apsol5, absolute)
# end def plot_error_apsol5

def plot_error_lookup (absolute) :
    plot_error (inv_involute_lookup, absolute)
# end def plot_error_lookup

class Material :
    # Einheit Härtegrad
    # FIXME: HRCN should be removed, this is now done with the
    # material_type (see below)
    HB   = 0
    HRC  = 1
    HRCN = 2

    # Resulting psi_dlim from shaft bearing (Aufhängung / Lager)
    # symetrical   O--X--O
    psi_dlim_symmetrical = dict \
        ( normal_annealed = 1.6
        , tempered        = 1.4
        , case_hardened   = 1.1
        , nitrified       = 0.8
        )
    # asymetrical  O-X---O
    psi_dlim_asymetrical = dict \
        ( normal_annealed = 1.3
        , tempered        = 1.1
        , case_hardened   = 0.9
        , nitrified       = 0.6
        )
    # flying       O---O-X
    psi_dlim_flying = dict \
        ( normal_annealed = 0.8
        , tempered        = 0.7
        , case_hardened   = 0.6
        , nitrified       = 0.4
        )
    material_types = set (k for k in psi_dlim_flying)

    dlim_by_bearing = dict \
        ( symmetrical   = psi_dlim_symmetrical
        , asymmetrical  = psi_dlim_asymetrical
        , flying        = psi_dlim_flying
        )

    def __init__ \
        ( self
        , name
        , unit
        , material_type
        , hardness_min,    hardness_max
        , delta_f_lim_min, delta_f_lim_max
        , delta_h_lim_min, delta_h_lim_max
        , cost_factor
        ) :
        self.name               = name
        self.material_type      = material_type
        self.unit               = unit
        self.hardness_min       = hardness_min
        self.hardness_max       = hardness_max
        self.delta_f_lim_min    = delta_f_lim_min
        self.delta_f_lim_max    = delta_f_lim_max
        self.delta_h_lim_min    = delta_h_lim_min
        self.delta_h_lim_max    = delta_h_lim_max
        self.cost_factor        = cost_factor
        # Material type must be known
        assert self.material_type in self.material_types
    # end def __init__

    @property
    def hardness (self) :
        return (self.hardness_min + self.hardness_max) / 2
    # end def hardness

    @property
    def delta_f_lim (self) :
        return (self.delta_f_lim_min + self.delta_f_lim_max) / 2
    # end def delta_f_lim

    @property
    def delta_h_lim (self) :
        return (self.delta_h_lim_min + self.delta_h_lim_max) / 2
    # end def delta_h_lim

    @property
    def psi_d (self) :
        if self.unit == self.HB :
            if self.hardness < 180 :
                return 1.2
            return 1.0
        elif self.unit == self.HRC :
            return 0.8
        return 0.5
    # end def psi_d

    def psi_dlim (self, shaft_bearing) :
        """ Compute *maximum* psi_d from given shaft bearing
            and the parameter of the material
        """
        return self.dlim_by_bearing [shaft_bearing][self.material_type]
    # end psi_dlim

# end class Material

class Gear :

    # Constants (for this project at least):
    alpha = np.pi * 20 / 180
    # Anwendungsfaktor K_A
    K_A = 1.5
    # Profilverschiebung (profile shift)
    profile_shift = [0, 0]

    # FIXME
    # A note on modul: This is a european way to describe this, the
    # american version uses Diametral Pitch which is the reciprocal.
    # See https://de.wikipedia.org/wiki/Modul_(Zahnrad)
    modul_table_DIN_780_I = \
        [ 0.0, 1.1, 2.2
        ]
    modul_table_DIN_780_II = \
        [ 0.0, 1.1, 2.2
        ]

    def __init__ (self, materials, z, beta, n_ein) :
        self.materials = materials
        self.z         = z
        self.beta      = beta
        assert len (z) == 2
        assert len (materials) == 2
        # Leistung P
        self.P   = 50e3
        self.Z_E = 189.8
        n_r      = n_ein
        # Stirnmodul Rad 1, Rad 2
        self.stirnmodul = []
        for m in materials :
            delta_Hlim = m.delta_h_lim
            ### FIXME: psi_dlim ???
            ###psi_d      = m.psi_d
            psi_d = 1 # FIXME
            u_tat      = z [1] / z [0]
            # Betriebsmoment Welle
            T_ges = self.P * self.K_A / (2 * n_r * np.pi)
            Z_H2  = 1 # FIXME
            sm = np.sqrt \
                ( (2 * T_ges * 1.2) / ((delta_Hlim / 1.4) ** 2)
                * (u_tat + 1) / u_tat
                * (self.Z_E ** 2) * (Z_H2 ** 2)
                * 1 / (psi_d * z [0] ** 3)
                )
            self.stirnmodul.append (sm)
            n_r = n_r * z [1] / z [0]
        self.stirnmodul  = np.array (self.stirnmodul)
        self.normalmodul = self.stirnmodul * np.cos (beta)
    # end def __init__

    @property
    def profile_shift_normalized (self) :
        """ This is the term (x1 + x2) / (z1 + z2) in computation of Z_H
        """
        return sum (self.profile_shift) / sum (self.z)
    # end def profile_shift_normalized

    def zone_factor (self, beta = None, shift = None) :
        """ Zone factor Z_H. This typically gets the transmission index
            and computes all the other values from it.
            Note that we use a efficient method for computing the
            inverse of the involute. The largest angle we use is 37.43°
        >>> m  = Material ('S235JR', Material.HB, 'normal_annealed'
        ...               , 120, 120, 125, 190, 315, 430, 1)
        >>> g  = Gear ((m, m), (30, 30), 15, 1)
        >>> zf = g.zone_factor
        >>> print ("%.5f" % zf (beta = 0, shift = 0))
        2.49457
        >>> print ("%.5f" % zf (beta = 5 * np.pi / 180, shift = 0))
        2.48675
        >>> print ("%.5f" % zf (beta = 10 * np.pi / 180, shift = 0))
        2.46337
        >>> print ("%.5f" % zf (beta = 15 * np.pi / 180, shift = 0))
        2.42473
        >>> print ("%.5f" % zf (beta = 25 * np.pi / 180, shift = 0))
        2.30385
        >>> print ("%.5f" % zf (beta = 35 * np.pi / 180, shift = 0))
        2.13072
        >>> print ("%.5f" % zf (beta = 40 * np.pi / 180, shift = 0))
        2.02782
        >>> print ("%.5f" % zf (beta = 30 * np.pi / 180, shift = -0.02))
        2.66981
        >>> print ("%.5f" % zf (beta = 30 * np.pi / 180, shift = 0.1))
        1.70126
        """
        if beta is None :
            beta = self.beta
        if shift is None :
            shift = self.profile_shift_normalized
        alpha_t = atan (np.tan (self.alpha) / np.cos (beta))
        beta_b  = atan (np.tan (beta) * np.cos (alpha_t))
        if shift == 0 :
            alpha_tw = alpha_t
        else :
            inv_alpha_tw = \
                np.tan (alpha_t) - alpha_t + 2 * np.tan (self.alpha) * shift
            alpha_tw = inv_involute (inv_alpha_tw)
        return \
            ( (1 / np.cos (alpha_t))
            * np.sqrt (2 * np.cos (beta_b) / np.tan (alpha_tw))
            )
    # end def zone_factor

    def plot_zone_factor (self) :
        shifts = ( .1, .09, .08, .07, .06, .05, .04, .03, .025, .02, .015
                 , .01, .005, 0.0, -.005, -.01, -.015, -.02
                 )
        x   = np.arange (0, 45, .01)
        fig = plt.figure ()
        ax  = fig.add_subplot (1, 1, 1)
        ax.set_ylim ((1.5, 3.0))
        ax.grid (True)
        x_ticks = np.arange (0, 46, 5)
        y_ticks = np.arange (1.5, 3.05, 0.1)
        ax.set_xticks (x_ticks)
        ax.set_yticks (y_ticks)
        for shift in shifts :
            y = []
            for b in x :
                br = b / 180 * np.pi
                y.append (self.zone_factor (beta = br, shift = shift))
            ax.plot (x, y)
        plt.show ()
    # end def plot_zone_factor

# end class Gear

class Gearbox :
    def __init__ (self, materials, z, beta, n_ein) :
        assert len (z) == 4
        assert len (materials) == 4
        # FIXME: the two gears get different parameters (beta, n_ein)
        self.gears = \
            [ Gear (materials [:2], z [:2], beta, n_ein)
            , Gear (materials [2:], z [2:], beta, n_ein)
            ]
        self.fac = (z [0] * z [2]) / (z [1] * z [3])
        # FIXME: The following probably should be moved to the Gear class
        #        And we probably want the different computations in
        #        different methods of that class -- with separate tests
        # FIXME: We probably want to rename variables for english terms
        #        e.g. n_ein -> n_in (?) etc. or even longer names, e.g.
        #        instead of n_in we might use rotary_speed
        # Eingangsdrehzahl n_ein
        self.n_ein = n_ein
        # Durchmesser/Breitenverhältnis
        # Modul/Breitenverhältnis TB 21-13b small-> easier; FIXABLE TB
        self.phi_m = 20
        # FIXME: see n_ein same value?
        ### self.n_r          = self.args.numerator
        n_r = n_ein # FIXME ?????

        ### max:

        ### normalmodul = m_n


        # Zahnbreite
        ### b = phi_m * m_n
        # check breite
        ### psi_d = b / D_R # = psi_dlim

        # check umfangs v
        ### v = D_Ritzel * n_ein * np.pi
        ### v <= 10
        ###     if beta == 0
        ###     else 15

        # m_n Normalmodul

        # d_RW gibt maximalen durchmesser Ritzelwelle
        ### d_RW = m_n * (z_R - 2.5) / (1.1 * np.cos (beta))
        # d_W gibt maximalen Durchmesser Welle mit aufgestecktem Ritzel
        ### d_W  = m_n * (z_R - 2.5) / (1.8 * np.cos (beta))

        # Kopfkreisdurchmesser
        ### D_KRad = D_Rad + 2 * m_n
        # Profilüberdeckung
        ### epsilon_alpha = (  0.5
        ###                 * np.sqrt (D_KRitzel ** 2 - D_bRitzel ** 2)
        ###                 + np.sqrt (D_KRad ** 2 - D_bRad ** 2)
        ###                 - a * np.sin (alpha_t)
        ###                 / np.pi * m_t * np.cos (alpha_t)
        ###                 )
        # Normal-Profilüberdeckung zwischen [1.1,1.25]
        ### epsilon_alphan = epsilon_alpha / np.cos (beta) ** 2
        # Sprungüberdeckung (nur für Schrägvz) >1
        ### epsilon_beta = b_Rad * np.sin (beta) / np.pi * m_n

    ### Öltauchschmierung hier tatsächlich wichtig beide Großräder zu
    ### beschreiben
        # Eintauchtiefe
        ### t_1 = x * m_n
        # Ölniveau
        ### t_oil = D_KRad / 2 - t
        # Eintauchtiefe Rad2
        ### t2 = D_KRad2 FIXME / 2 - t
        # Eintauchfaktor Rad2
        ### x2 = t2 / m_n2 FIXME #zw 2-10 * m_n2
    # end def __init__

# end class Gearbox

class Gear_Optimizer (pga.PGA, autosuper) :
    """
    """
    # Name, Härtegrad-Einheit Flankenhärte, Zahnfußdauerfestigkeit (min/max)
    # Zahnflankendauerfestigkeit (min/max), rel Material cost
    # FIXME: The HRCN should probably be replaced with HRC and the
    # material_type name above. The material_type is a simple string.
    HB   = Material.HB
    HRC  = Material.HRC
    HRCN = Material.HRCN
    # FIXME: This is an example, nitrified below is almost certainly wrong
    # All the other materials have to be fixed.
    materials = \
        ( Material ( 'S235JR', HB, 'nitrified'
                   , 120, 120, 125, 190,  315,  430, 1
                   )
#        , Material ( 'E295', HB,
#                   , 160, 160, 140, 210,  350,  485, 1.1
#                   )
#        , Material ( 'E335', HB,
#                   , 190, 190, 160, 225,  375,  540, 1.7
#                   )
#        , Material ( 'C45E_N', HB,
#                   , 190, 190, 160, 260,  470,  590, 1.7
#                   )
#        , Material ( 'QT34CrMo4', HB,
#                   , 270, 270, 220, 335,  540,  800, 2.4
#                   )
#        , Material ( 'QT42CrMo4', HB,
#                   , 300, 300, 230, 335,  540,  800, 2.4
#                   )
#        , Material ( 'QT34CrNiMo6', HB,
#                   , 310, 310, 235, 345,  580,  840, 2.4
#                   )
#        , Material ( 'QT30CrNiMo8', HB,
#                   , 320, 320, 240, 355,  610,  870, 2.7
#                   )
#        , Material ( 'QT36CrNiMo16', HB,
#                   , 350, 350, 250, 365,  640,  915, 3
#                   )
#        , Material ( 'UH +FH CrMo TB20-1(Nr.19-22)', HRC
#                   , 50,  50, 230, 380,  980, 1275, 5
#                   )
#        , Material ( 'UH -FH CrMo TB20-1(Nr.19-22)', HRC
#                   , 50,  50, 150, 230,  980, 1275, 4
#                   )
#        , Material ( 'EH +FH CrMo TB20-1(Nr.19-22)', HRC
#                   , 56,  56, 270, 410, 1060, 1330, 7
#                   )
#        , Material ( 'EH -FH CrMo TB20-1(Nr.19-22)', HRC
#                   , 56,  56, 150, 230, 1060, 1330, 6
#                   )
#        , Material ( 'QT42CrMo4N', HRC
#                   , 48,  57, 260, 430,  780, 1215, 4
#                   )
#        , Material ( 'QT16MnCr5N', HRC
#                   , 48,  57, 260, 430,  780, 1215, 4
#                   )
#        , Material ( 'C45E_NN', HRCN
#                   , 30,  45, 225, 290,  650,  780, 3
#                   )
#        , Material ( '16MnCr5N', HRCN
#                   , 45,  57, 225, 385,  650,  950, 3.5
#                   )
#        , Material ( 'QT34Cr4CN', HRCN
#                   , 55,  60, 300, 450, 1100, 1350, 5.5
#                   )
#        , Material ( '16MnCr5EGm20', HRCN
#                   , 58,  62, 310, 525, 1300, 1650, 9
#                   )
#        , Material ( '15CrNi6EGm16', HRCN
#                   , 58,  62, 310, 525, 1300, 1650, 10
#                   )
#        , Material ( '18CrNiMo7-6EGm16', HRCN
#                   , 58,  62, 310, 525, 1300, 1650, 11
#                   )
        )

    def __init__ (self, args) :
        self.args   = args
        self.factor = self.args.numerator / self.args.denominator
        # Teeth (4 parameters)
        minmax = [(self.args.min_tooth, self.args.max_tooth)] * 4
        # Stahl: Index into table "materials" above
        minmax.extend ([(0, len (self.materials))] * 4)
        # Schrägungswinkel
        minmax.append ((8, 20))
        # Eintauchfaktor Rad1 x_1
        minmax.append ((2, 5))
        d = dict \
            ( maximize             = False
            , num_eval             = 2
            , num_constraint       = 1
            , pop_size             = 60
            , num_replace          = 60
            , select_type          = pga.PGA_SELECT_LINEAR
            , pop_replace_type     = pga.PGA_POPREPL_PAIRWISE_BEST
            , mutation_only        = True
            , mutation_type        = pga.PGA_MUTATION_DE
            , DE_crossover_prob    = 0.8
            , DE_crossover_type    = pga.PGA_DE_CROSSOVER_BIN
            , DE_variant           = pga.PGA_DE_VARIANT_RAND
            , DE_scale_factor      = 0.85
            , init                 = minmax
            , max_GA_iter          = 1000
            , print_options        = [pga.PGA_REPORT_STRING]
            , mutation_bounce_back = True
            )
        if args.random_seed :
            d ['random_seed'] = args.random_seed
        self.__super.__init__ (float, len (minmax), **d)
    # end def __init__

    def err (self, x1, x2, x3, x4) :
        return abs (self.factor - (x3 * x4) / (x1 * x2)) / self.factor * 100
    # end def err

    def phenotype (self, p, pop) :
        z = []
        for i in range (4) :
            z.append (round (self.get_allele (p, pop, i)))
        mat = []
        for i in range (4) :
            stahl = int (self.get_allele (p, pop, i + 4))
            if stahl >= len (self.materials) :
                stahl = len (self.materials) - 1
            stahl = self.materials [stahl]
            mat.append (stahl)
        beta  = self.get_allele (p, pop, 8)
        x_1   = self.get_allele (p, pop, 9)
        g = Gearbox (mat, z, beta, x_1, n_ein = self.args.numerator)
        return g
    # end def phenotype

    def evaluate (self, p, pop) :
        # FIXME: Use phenotype above,
        #        decide what checks to put into Gearbox class
        gc1 = gcd (z [0], z [2]) + gcd (z [1], z [3])
        f   = (1 / self.factor - fac) ** 2
        return f, min (gc1, gc2) - 2
    # end def evaluate

    def print_string (self, file, p, pop) :
        z = []
        for i in range (4) :
            z.append (round (self.get_allele (p, pop, i)))
        print (z, file = file)
        print ("Gear Error: %12.9f%%" % self.err (*z))
        print ("Random seed: %d" % self.random_seed)
        self.__super.print_string (file, p, pop)
    # end def print_string

# end class Gear_Optimizer

if __name__ == '__main__' :
    cmd = ArgumentParser ()
    cmd.add_argument \
        ( '--absolute-error'
        , help    = "When plotting errors, use absolute error"
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '-c', '--check'
        , help    = "A comma-separated list of 4 integers for gears to check"
        )
    cmd.add_argument \
        ( '-l', '--min-tooth'
        , type    = int
        , default = 12
        )
    cmd.add_argument \
        ( '-u', '--max-tooth'
        , type    = int
        , default = 60
        )
    cmd.add_argument \
        ( '-d', '--denominator'
        , type    = float
        , default = 207
        )
    cmd.add_argument \
        ( '-n', '--numerator'
        , type    = float
        , default = 3510
        )
    cmd.add_argument \
        ( '-p', '--plot-zone-factor'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '--plot-error'
        , type    = int
        , default = 1
        )
    cmd.add_argument \
        ( '-r', '--random-seed'
        , type    = int
        )
    args = cmd.parse_args ()
    if args.plot_error :
        if args.plot_error == 1 :
            plot_error_lookup (absolute = args.absolute_error)
        elif args.plot_error == 4 :
            plot_error_apsol4 (absolute = args.absolute_error)
        elif args.plot_error == 5 :
            plot_error_apsol5 (absolute = args.absolute_error)
    elif args.plot_zone_factor :
        m  = Material ('S235JR', Material.HB, 'normal_annealed'
                      , 120, 120, 125, 190, 315, 430, 1)
        g = Gear ([m, m], [30, 30], 15, 1)
        g.plot_zone_factor ()
    elif args.check :
        go = Gear_Optimizer (args)
        z  = [int (i) for i in args.check.split (',')]
        print ("Factor: %f" % go.factor)
        print ("Gear Error: %12.9f%%" % go.err (*z))
    else :
        go = Gear_Optimizer (args)
        go.run ()
