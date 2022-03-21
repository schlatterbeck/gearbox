#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from rsclib.autosuper import autosuper
from argparse import ArgumentParser
from math import gcd, atan, acos
from bisect import bisect_right
from involute import inv_involute_apsol4 as inv_involute
import pga
import sys

class Material :
    # Einheit Härtegrad
    HB   = 0
    HRC  = 1

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
        , sigma_f_lim_min, sigma_f_lim_max
        , sigma_h_lim_min, sigma_h_lim_max
        , cost_factor
        ) :
        self.name               = name
        self.material_type      = material_type
        self.unit               = unit
        self.hardness_min       = hardness_min
        self.hardness_max       = hardness_max
        self.sigma_f_lim_min    = sigma_f_lim_min
        self.sigma_f_lim_max    = sigma_f_lim_max
        self.sigma_h_lim_min    = sigma_h_lim_min
        self.sigma_h_lim_max    = sigma_h_lim_max
        self.cost_factor        = cost_factor
        # Material type must be known
        assert self.material_type in self.material_types
    # end def __init__

    @property
    def hardness (self) :
        return (self.hardness_min + self.hardness_max) / 2
    # end def hardness

    @property
    def sigma_f_lim (self) :
        return (self.sigma_f_lim_min + self.sigma_f_lim_max) / 2
    # end def sigma_f_lim

    @property
    def sigma_h_lim (self) :
        return (self.sigma_h_lim_min + self.sigma_h_lim_max) / 2
    # end def sigma_h_lim

    def psi_dlim (self, shaft_bearing) :
        """ Compute *maximum* psi_d from given shaft bearing
            and the parameter of the material
        """
        return self.dlim_by_bearing [shaft_bearing][self.material_type]
    # end psi_dlim

# end class Material

class Shaft :
    def __init__ (self, gearbox, T_ges) :
        self.gearbox = gearbox
        self.T_ges   = T_ges
    # end def __init__

    def add_gears (input_gear = None, output_gear = None) :
        self.input_gear  = input_gear
        self.output_gear = output_gear

        g = self.gear = input_gear
        if g is None :
            g = output_gear
        # Tangentialkraft
        self.F_t = 2 * self.T_ges / g.D_R [0]
        # Radialkraft
        self.F_r = self.F_t * np.tan (g.alpha) / np.cos (g.beta)
        # Axialkraft
        self.F_a = self.F_t * np.tan (g.beta)
        # Bearings
        # FIXME
        gb = self.gearbox
        # FIXME: This is different for each Shaft depending on
        # input/output gear(s)
        # Tangentiale Lagerkraft A, B
        self.F_Bt = F_t * gb.l_A / gb.l_AB
        self.F_At = F_t - F_Bt
        # Normale Lagerkraft A, B
        self.F_Bn = F_r * gb.l_A / gb.l_AB
        self.F_An = F_r - F_Bn
        # Radiale Lagerkraft A, B
        self.F_Br = np.sqrt (F_Bt ** 2 + F_Bn ** 2)
        self.F_Ar = np.sqrt (F_At ** 2 + F_An ** 2)
        # Axiale Lagerkraft A, B
        #self.F_Ba = FIXME
        #self.F_Aa = FIXME

        #F_Bt = F_t * gb.l_A + F_t2 * (gb.l_AB - l_B2)
        #M_bmax = F_Br * gb.l_B
        #M_v = np.sqrt (M_bmax ** 2 + 0.75 * (0.7 * T_ges) ** 2)

    # end def add_input_gear
# end class Shaft

class Gear :

    # Constants (for this project at least):
    alpha = np.pi * 20 / 180
    # Profilverschiebung (profile shift)
    profile_shift = [0, 0]
    # Elastizitätsfaktor Stahl-Stahl TB21-21b
    Z_E = 189.8
    # Modul/Breitenverhältnis TB 21-13b
    psi_m = 25
    # Submersion depth default if unspecified in constructor
    submersion_factor = 2

    # FIXME
    # A note on modul: This is a european way to describe this, the
    # american version uses Diametral Pitch which is the reciprocal.
    # See https://de.wikipedia.org/wiki/Modul_(Zahnrad)
    # TB 21-1
    modul_DIN_780_I = \
        [ 0.1, 0.12, 0.16, 0.20, 0.25, 0.3, 0.4, 0.5, 0.6,
          0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6,
          7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 60
        ]
    modul_DIN_780_II = \
        [ 0.11, 0.14, 0.18, 0.22, 0.28, 0.35, 0.45, 0.55,0.65, 0.75,
          0.85, 0.95, 1.125, 1.375, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5,
          7, 9, 11, 14, 18, 22, 28, 36, 45, 55, 70
        ]

    # submersion depth factor 2..5 (sub_fac)
    def __init__ (self, materials, z, beta, n_ein, shaft, sub_fac = None) :
        self.materials = materials
        self.z         = np.array (z)
        self.beta      = beta
        self.n_ein     = n_ein
        self.shaft     = shaft
        self.alpha_t   = atan (np.tan (self.alpha) / np.cos (beta))
        if s_depth is not None :
            self.submersion_factor = sub_fac
        assert len (z) == 2
        assert len (materials) == 2
        # Zone factor Z_H
        Z_H      = self.zone_factor ()
        # Ritzel
        sigma_Hlim = m [0].sigma_h_lim
        # tatsächliches Übersetzungsverhältnis
        u_tat = z [1] / z [0]
        # Betriebsmoment Eingangs-Welle
        self.T_ges = self.shaft.T_ges
        T_ges = self.P * self.K_A / (2 * n_ein * np.pi)
        # Stirnmodul Rad 1, Rad 2
        psi_d_max = max (m.psi_dlim for m in materials)
        self.stirnmodul_calc = np.sqrt \
            ( (2 * T_ges * 1.2) / ((sigma_Hlim / 1.4) ** 2)
            * (u_tat + 1) / u_tat
            * (self.Z_E ** 2) * (Z_H ** 2)
            * 1 / (psi_d_max * z [0] ** 3)
            )
        self.normalmodul_calc = self.stirnmodul_calc * np.cos (beta)
        # lookup moduln
        idx = bisect_right (self.modul_DIN_780_I, self.normalmodul_calc)
        nm_I = modul_DIN_780_I [idx]
        if nm_I < self.normalmodul_calc :
            nm_I = modul_DIN_780_I [idx + 1]
        nm_II = modul_DIN_780_II [idx]
        if nm_II < self.normalmodul_calc :
            nm_II = modul_DIN_780_II [idx + 1]     
        self.stirnmodul = self.normalmodul / np.cos (beta)
        # Only keep nm_II if it matches better than nm_I
        if  ( abs (nm_I  - self.normalmodul_calc)
            < abs (nm_II - self.normalmodul_calc)
            ) :
            nm_II = None
        self.D_R = z * self.stirnmodul
        self.D_K = self.D_R + 2 * self.normalmodul
        # Durchmesser/Breitenverhältnis
        b   = self.psi_m * self.normalmodul
        # check breite
        self.psi_d = b / self.D_R # <= psi_dlim

        # v out
        self.v = self.D_R [1] * n_ein * np.pi

        # d_RW gibt maximalen durchmesser Ritzelwelle
        m_n = self.normalmodul
        self.d_RW = m_n * (z [0] - 2.5) / (1.1 * np.cos (self.beta))
        # d_W gibt maximalen Durchmesser Welle mit aufgestecktem Ritzel
        self.d_W  = m_n * (z [0] - 2.5) / (1.8 * np.cos (self.beta))

    # end def __init__

    @property
    def profile_shift_normalized (self) :
        """ This is the term (x1 + x2) / (z1 + z2) in computation of Z_H
        """
        return sum (self.profile_shift) / sum (self.z)
    # end def profile_shift_normalized

    @property
    def out_T_ges (self) :
        return self.T_ges * self.u_tat
    # end def out_T_ges

    def profile_overlap (self, gearbox)
        """ Profilüberdeckung
        """
        m_t = self.stirnmodul
        D_b = self.D_R * np.cos (self.alpha_t)
        self.epsilon_alpha =
            ( ( 0.5
              * ( np.sqrt (D_K [0] ** 2 - D_b [0] ** 2)
                + np.sqrt (D_K [1] ** 2 - D_b [1] ** 2)
                )
              )
            - gearbox.a * np.sin (self.alpha_t)
            ) / (np.pi * m_t * np.cos (self.alpha_t))

        # FIXME
        # Normal-Profilüberdeckung zwischen [1.1,1.25]
        ### epsilon_alphan = epsilon_alpha / np.cos (beta) ** 2
        # Sprungüberdeckung (nur für Schrägvz) >1
        ### epsilon_beta = b_Rad * np.sin (beta) / np.pi * m_n
    # end def profile_overlap (self)

    def zone_factor (self, beta = None, shift = None) :
        """ Zone factor Z_H. This typically gets the transmission index
            and computes all the other values from it.
            Note that we use a efficient method for computing the
            inverse of the involute. The largest angle we use is 37.43°
        >>> m  = Material ('S235JR', Material.HB, 'normal_annealed'
        ...               , 120, 120, 125, 190, 315, 430, 1)
        >>> s  = Shaft (1)
        >>> g  = Gear ((m, m), (30, 30), 15, 1, s)
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
        beta_b  = atan (np.tan (beta) * np.cos (self.alpha_t))
        if shift == 0 :
            alpha_tw = self.alpha_t
        else :
            inv_alpha_tw = \
                ( np.tan (self.alpha_t)
                - self.alpha_t
                + 2 * np.tan (self.alpha) * shift
                )
            alpha_tw = inv_involute (inv_alpha_tw)
        return \
            ( (1 / np.cos (self.alpha_t))
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

    def constrain_v (self) :
        """ v must be less than a given number.
            Constraints for the GA must be <= 0
        """
        if self.beta > 0 :
            return self.v - 15
        return self.v - 10
    # end def constrain_v

# end class Gear

class Gearbox :
    # Anwendungsfaktor K_A
    K_A = 1.5
    # Leistung P
    P   = 50e3

    def __init__ (self, materials, z, beta, n, delta_b) :
        assert len (z) == 4
        assert len (materials) == 4
        n2 = n * z [0] / z [1]
        T_ges = self.P * self.K_A / (2 * n * np.pi)
        s = self.shaft = []
        g = self.gears = []
        self.shaft.append (Shaft (self, T_ges))
        self.gears.append (Gear (materials [:2], z [:2], beta, n, s [-1]))
        self.shaft.append (Shaft (self, g [-1].out_T_ges))
        self.gears.append (Gear (materials [2:], z [2:], 0, n2, s [-1]))
        self.shaft.append (Shaft (self, g [-1].out_T_ges))

        self.fac = (z [0] * z [2]) / (z [1] * z [3])
        self.delta_b = delta_b

        g = self.gears
        # Distance of box to first wheel
        s_z = [3 * x.normalmodul for x in g]
        # Wandstärke Gehäuse (s_12)
        s_Wg = 8 # FIXME from table GJS RM S.785/20.6
        # Flanschbreite
        s_b = 3 * s_Wg + 10
        # Flanschdicke
        s_d = 1.5 * s_Wg
        # Gehäuse Innenlänge
        a = [sum (x.D_K) / 2 for x in g]
        l_Gi = ( sum (a)
               + (g [0].D_K [0] + g [1].D_K [1]) / 2
               + sum (s_z)
               )
        # Gehäuse Innenbreite
        l_12 = ((g [0].b - 2) + (g [1].b)) / 2 + s_z [1]
        l_0G = g [0].b / 2 + s_z [0]
        l_1G = g [1].b / 2 + s_z [1]
        b_Gi = l_12 + l_1G + l_2G

        l_GL = delta_b + s_Wg + g [0].b / 2
        l_A  = l_GL + l_0G
        l_B  = l_GL + l_1G
        l_AB = l_A  + l_B + l_12

        for g in self.gears :
            g.profile_overlap (self)

    # end def __init__

    def oil (self) :
        """ Öltauchschmierung hier tatsächlich wichtig beide Großräder
            zu beschreiben
        """
        g0 = self.gears [0]
        g1 = self.gears [1]
        t.append (g0.submersion_factor * g0.normalmodul)
        # Ölniveau
        t_oil = g0.D_K [1] / 2 - t [0]
        # Eintauchtiefe Rad2
        t.append (g1.D_K [1] / 2 - t_oil)
        # Eintauchfaktor Rad2
        submersion_factor_2 = t [1] / g1.normalmodul 
        g1.submersion_factor = submersion_factor_2
    # end def oil

    def constrain_submersion_upper_bound (self) :
        """ Submersion <= 10
            Checked for <= 0
        """
        return self.gears [1].submersion_factor - 10
    # end def constrain_submersion_upper_bound

    def constrain_submersion_lower_bound (self) :
        """ Submersion >= 2
            Checked for <= 0
        """
        return 2 - self.gears [1].submersion_factor
    # end def constrain_submersion_lower_bound

    def constrain_v_0 (self) :
        return self.gears [0].constrain_v ()
    # end def constrain_v_0

    def constrain_v_1 (self) :
        return self.gears [1].constrain_v ()
    # end def constrain_v_1

# end class Gearbox

class Gear_Optimizer (pga.PGA, autosuper) :
    """
    """
    # Name, Härtegrad-Einheit Flankenhärte, Zahnfußdauerfestigkeit (min/max)
    # Zahnflankendauerfestigkeit (min/max), rel Material cost

    HB   = Material.HB
    HRC  = Material.HRC

    materials = \
        ( Material ( 'S235JR', HB, 'normal_annealed'
                   , 120, 120, 125, 190,  315,  430, 1
                   )
        , Material ( 'E295', HB, 'normal_annealed'
                   , 160, 160, 140, 210,  350,  485, 1.1
                   )
        , Material ( 'E335', HB, 'normal_annealed'
                   , 190, 190, 160, 225,  375,  540, 1.7
                   )
        , Material ( 'C45E_N', HB, 'normal_annealed'
                   , 190, 190, 160, 260,  470,  590, 1.7
                   )
        , Material ( 'QT34CrMo4', HB, 'tempered'
                   , 270, 270, 220, 335,  540,  800, 2.4
                   )
        , Material ( 'QT42CrMo4', HB, 'tempered'
                   , 300, 300, 230, 335,  540,  800, 2.4
                   )
        , Material ( 'QT34CrNiMo6', HB, 'tempered'
                   , 310, 310, 235, 345,  580,  840, 2.4
                   )
        , Material ( 'QT30CrNiMo8', HB, 'tempered'
                   , 320, 320, 240, 355,  610,  870, 2.7
                   )
        , Material ( 'QT36CrNiMo16', HB, 'tempered'
                   , 350, 350, 250, 365,  640,  915, 3
                   )
        , Material ( 'UH +FH CrMo TB20-1(Nr.19-22)', HRC, 'case_hardened'
                   , 50,  50, 230, 380,  980, 1275, 5
                   )
        , Material ( 'UH -FH CrMo TB20-1(Nr.19-22)', HRC, 'case_hardened'
                   , 50,  50, 150, 230,  980, 1275, 4
                   )
        , Material ( 'EH +FH CrMo TB20-1(Nr.19-22)', HRC, 'case_hardened'
                   , 56,  56, 270, 410, 1060, 1330, 7
                   )
        , Material ( 'EH -FH CrMo TB20-1(Nr.19-22)', HRC, 'case_hardened'
                   , 56,  56, 150, 230, 1060, 1330, 6
                   )
        , Material ( 'QT42CrMo4N', HRC, 'nitrified'
                   , 48,  57, 260, 430,  780, 1215, 4
                   )
        , Material ( 'QT16MnCr5N', HRC, 'nitrified'
                   , 48,  57, 260, 430,  780, 1215, 4
                   )
        , Material ( 'C45E_NN', HRC, 'nitrified'
                   , 30,  45, 225, 290,  650,  780, 3
                   )
        , Material ( '16MnCr5N', HRC, 'nitrified'
                   , 45,  57, 225, 385,  650,  950, 3.5
                   )
        , Material ( 'QT34Cr4CN', HRC, 'nitrified'
                   , 55,  60, 300, 450, 1100, 1350, 5.5
                   )
        , Material ( '16MnCr5EGm20', HRC, 'nitrified'
                   , 58,  62, 310, 525, 1300, 1650, 9
                   )
        , Material ( '15CrNi6EGm16', HRC, 'nitrified'
                   , 58,  62, 310, 525, 1300, 1650, 10
                   )
        , Material ( '18CrNiMo7-6EGm16', HRC, 'nitrified'
                   , 58,  62, 310, 525, 1300, 1650, 11
                   )
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
        # delta_b (should be minimized)
        minmax.append ((0, 1000))
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
        delta_b = self.get_allele (p, pop, 10)
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
        ( '-r', '--random-seed'
        , type    = int
        )
    args = cmd.parse_args ()
    if args.plot_zone_factor :
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
