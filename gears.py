#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from rsclib.autosuper import autosuper
from argparse import ArgumentParser
from math import gcd, atan, acos
from bisect import bisect_right, bisect_left
from involute import inv_involute_apsol4 as inv_involute
from corr import yfa, ysa, Fit_Curve, curve_fit
import pga
import sys

class Material (Fit_Curve) :
    """ Note on hardness: Brinell hardness can either be measured with a
        standardized ball or with a tungsten carbide ball. The latter
        starts to differ from the measurements with the standard ball at
        about 442 brinell hardness (where the tungsten ball measures 443)
        Note that measurements with the standard ball are defined only
        up to 451 or so.
        So we are using Brinell hardness measured with the tungsten ball
        especially when converting Rockwell Scale C hardness to Brinell.
        All values directly given in Brinell hardness are below 200
        anyway.
    """
    # Einheit Härtegrad
    HB   = 0
    HRC  = 1

    # Resulting psi_dlim from shaft bearing (Aufhängung / Lager)
    # From TB-21-13
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
        self.__super.__init__ ()
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
    def is_hardened (self) :
        return self.material_type in ('case_hardened', 'nitrified')
    # end def is_hardened

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

    hrc_table = \
        ( (30, ( 5.97,   104.7))
        , (40, ( 8.57,    27.6))
        , (50, (11.158,  -79.6))
        , (60, (17.515, -401.0))
        , (65, (17.515, -401.0))
        )
    hrc_tbl2 = \
        ( (20, 224)
        , (21, 231)
        , (22, 237)
        , (23, 243)
        , (24, 247)
        , (25, 253)
        , (26, 258)
        , (27, 264)
        , (28, 271)
        , (29, 279)
        , (30, 286)
        , (31, 294)
        , (32, 301)
        , (33, 311)
        , (34, 319)
        , (35, 327)
        , (36, 336)
        , (37, 344)
        , (38, 353)
        , (39, 362)
        , (40, 371)
        , (41, 381)
        , (42, 390)
        , (43, 400)
        , (44, 409)
        , (45, 421)
        , (46, 432)
        , (47, 443)
        , (48, 455)
        , (49, 469)
        , (50, 481)
        , (51, 496)
        , (52, 512)
        , (53, 525)
        , (54, 543)
        , (55, 560)
        , (56, 577)
        , (57, 595)
        , (58, 615)
        , (59, 634)
        , (60, 654)
        , (61, 670)
        , (62, 688)
        , (63, 705)
        , (64, 722)
        , (65, 739)
        )

    def hardness_hb (self, h = None) :
        """ Convert hardness in whatever unit to HB (Brinell)
            Currently we support only HRC (Rockwell Scale C)
        >>> HRC = Material.HRC
        >>> ni = 'nitrified'
        >>> for hrc in range (20, 66) :
        ...     m = Material ('', HRC, ni, hrc, hrc, 0, 0, 0, 0, 1)
        ...     hb = m.hardness_hb ()
        ...     d  = dict ((m.hrc_tbl2))
        ...     tb = d [hrc]
        ...     print ("%.0f %.0f %+.1f" % (m.hardness, hb, hb - tb))
        20 224 +0.1
	21 230 -0.9
	22 236 -1.0
	23 242 -1.0
	24 248 +1.0
	25 254 +0.9
	26 260 +1.9
	27 266 +1.9
	28 272 +0.9
	29 278 -1.2
	30 284 -2.2
	31 293 -0.7
	32 302 +0.8
	33 310 -0.6
	34 319 -0.0
	35 328 +0.6
	36 336 +0.1
	37 345 +0.7
	38 353 +0.3
	39 362 -0.2
	40 370 -0.6
	41 378 -3.1
	42 389 -1.0
	43 400 +0.2
	44 411 +2.4
	45 423 +1.5
	46 434 +1.7
	47 445 +1.8
	48 456 +1.0
	49 467 -1.9
	50 478 -2.7
	51 492 -3.7
	52 510 -2.2
	53 527 +2.3
	54 545 +1.8
	55 562 +2.3
	56 580 +2.8
	57 597 +2.4
	58 615 -0.1
	59 632 -1.6
	60 650 -4.1
	61 667 -2.6
	62 685 -3.1
	63 702 -2.6
	64 720 -2.0
	65 737 -1.5
        """
        if h is None :
            h = self.hardness
        if self.unit == self.HB :
            return h
        else :
            assert self.unit == self.HRC
            tbl = self.hrc_table
            k   = (h, (0, 0))
            idx = bisect_left (tbl, k)
            assert 0 <= idx <= len (tbl) - 1
            assert h <= tbl [idx][0]
            a, c = tbl [idx][1]
            return h * a + c
    # end def hardness_hb

    def material_combination_factor (self, other) :
        """ Z_W
            Material combination factor
            Werkstoffpaarungsfaktor
        >>> tt  = 'tempered'
        >>> ni  = 'nitrified'
        >>> HB  = Material.HB
        >>> HRC = Material.HRC
        >>> om  = Material ('', HRC, ni, 65, 65, 0, 0, 0, 0, 1)
        >>> print ("%.3f" % om.material_combination_factor (om))
        1.000
        >>> for hb in 120, 130, 150, 200, 250, 300, 350, 400, 450, 470, 500 :
        ...     m = Material ('', HB, tt, hb, hb, 0, 0, 0, 0, 1)
        ...     print ("%.3f" % m.material_combination_factor (om))
        1.200
        1.200
        1.188
        1.159
        1.129
        1.100
        1.071
        1.041
        1.012
        1.000
        1.000
        """
        #is_h = [self.is_hardened, other.is_hardened]
        #if is_h [0] == is_h [1] :
        #    return 1.0
        if self.hardness_hb () == other.hardness_hb () :
            return 1.0
        if self.hardness_hb () > other.hardness_hb () :
            return other.material_combination_factor (self)
        hb = self.hardness_hb ()
        if hb < 130 :
            return 1.2
        if hb > 470 :
            return 1.0
        return 1.2 - (hb - 130) / 1700
    # end def material_combination_factor

    def plot_hardness (self) :
        x, y = np.array (self.hrc_tbl2).T
        y_comp = []
        for xx in x :
            y_comp.append (self.hardness_hb (xx))
        plt.plot (x, y)
        plt.plot (x, y_comp)
        plt.show ()
        z = np.array (y_comp) - y
        plt.plot (x, z)
        plt.show ()
        yy = []
        for xx in x :
            yy.append (self._fitted_function (xx))
        yy = np.array (yy)
        plt.plot (x, y)
        plt.plot (x, yy)
        plt.show ()
        plt.plot (x, yy - y)
        plt.show ()
    # end def plot_hardness

    def psi_dlim (self, shaft_bearing = 'asymmetrical') :
        """ Compute *maximum* psi_d from given shaft bearing
            and the parameter of the material
        """
        return self.dlim_by_bearing [shaft_bearing][self.material_type]
    # end psi_dlim

    def size_factor_flank (self, m_n) :
        """ Z_X TB 21-20
            Size factor (flank) (Flank pressure)
            Größenfaktor (Flanke) (Flankenpressung)
            Note that this is always 1 for "Vergütungsstahl".
        >>> ch = 'case_hardened'
        >>> tt = 'tempered'
        >>> ni = 'nitrified'
        >>> HB = Material.HB
        >>> m1 = Material ('', HB, ch, 0, 0, 0, 0,  740,  740, 1)
        >>> m2 = Material ('', HB, ni, 0, 0, 0, 0,  740,  740, 1)
        >>> m3 = Material ('', HB, tt, 0, 0, 0, 0,  740,  740, 1)
        >>> print ("%.3f" % m1.size_factor_flank (10))
        1.000
        >>> print ("%.3f" % m1.size_factor_flank (25))
        0.925
        >>> print ("%.3f" % m1.size_factor_flank (30))
        0.900
        >>> print ("%.3f" % m1.size_factor_flank (45))
        0.900
        >>> print ("%.3f" % m2.size_factor_flank (10))
        0.970
        >>> print ("%.3f" % m2.size_factor_flank (25))
        0.805
        >>> print ("%.3f" % m2.size_factor_flank (30))
        0.750
        >>> print ("%.3f" % m2.size_factor_flank (45))
        0.750
        >>> print ("%.3f" % m3.size_factor_flank (10))
        1.000
        >>> print ("%.3f" % m3.size_factor_flank (45))
        1.000
        """
        if self.material_type in ('normal_annealed', 'tempered') :
            v = 1.0
        elif self.material_type == 'nitrified' :
            v = 1.08 - 0.011 * m_n
            if v < 0.75 :
                v = 0.75
        else :
            v = 1.05 - 0.005 * m_n
            if v < 0.9 :
                v = 0.9
        if v > 1 :
            v = 1.0
        return v
    # end def size_factor_flank

    def size_factor_tooth (self, m_n) :
        """ Y_X TB 21-20
            Size factor (tooth foot)
            Größenfaktor (Zahnfuß)
            Note that we do not have "Gusswerkstoffe"
            which would go down to 0.7
        >>> na = 'normal_annealed'
        >>> ni = 'nitrified'
        >>> HB = Material.HB
        >>> m1 = Material ('', HB, na, 0, 0, 0, 0,  740,  740, 1)
        >>> m2 = Material ('', HB, ni, 0, 0, 0, 0,  740,  740, 1)
        >>> print ("%.3f" % m1.size_factor_tooth (2))
        1.000
        >>> print ("%.3f" % m1.size_factor_tooth (5))
        1.000
        >>> print ("%.3f" % m1.size_factor_tooth (10))
        0.970
        >>> print ("%.3f" % m1.size_factor_tooth (15))
        0.940
        >>> print ("%.3f" % m1.size_factor_tooth (25))
        0.880
        >>> print ("%.3f" % m1.size_factor_tooth (30))
        0.850
        >>> print ("%.3f" % m1.size_factor_tooth (45))
        0.850
        >>> print ("%.3f" % m2.size_factor_tooth (2))
        1.000
        >>> print ("%.3f" % m2.size_factor_tooth (5))
        1.000
        >>> print ("%.3f" % m2.size_factor_tooth (10))
        0.950
        >>> print ("%.3f" % m2.size_factor_tooth (15))
        0.900
        >>> print ("%.3f" % m2.size_factor_tooth (25))
        0.800
        >>> print ("%.3f" % m2.size_factor_tooth (30))
        0.800
        >>> print ("%.3f" % m2.size_factor_tooth (45))
        0.800
        """
        if self.material_type in ('nitrified', 'case_hardened') :
            v = 1.05 - 0.01 * m_n
            if v < 0.8 :
                v = 0.8
        else :
            v = 1.03 - 0.006 * m_n
            if v < 0.85 :
                v = 0.85
        if v > 1 :
            v = 1.0
        return v
    # end def size_factor_tooth

    def y_beta (self, F_betax, v = 0) :
        """ From TB 21-17
        >>> tt = 'tempered'
        >>> ni = 'nitrified'
        >>> HB = Material.HB
        >>> m1 = Material ('', HB, tt, 0, 0, 0, 0,  740,  740, 1)
        >>> m2 = Material ('', HB, ni, 0, 0, 0, 0,  740,  740, 1)
        >>> m3 = Material ('', HB, tt, 0, 0, 0, 0,  800,  800, 1)
        >>> m4 = Material ('', HB, tt, 0, 0, 0, 0, 1000, 1000, 1)
        >>> print ("%.2f" % m2.y_beta (20))
        3.00
        >>> print ("%.2f" % m3.y_beta (30))
        12.00
        >>> print ("%.2f" % m4.y_beta (40))
        12.80
        >>> print ("%.2f" % m4.y_beta (90, 15))
        12.80
        >>> print ("%.2f" % m4.y_beta (90, 9))
        25.60
        >>> print ("%.2f" % m4.y_beta (90, 4))
        28.80
        """
        max_F_betax = 40
        if v < 10 :
            max_F_betax = 80
        if v < 5 :
            max_F_betax = 1e6
        if self.material_type in ('nitrified', 'case_hardened') :
            k = 0.15
            m = 6
        else :
            k = 320 / self.sigma_h_lim
            m = max_F_betax * k
        r = F_betax * k
        if r > m :
            r = m
        return r
    # end def y_beta

    def _fit (x, a, b, c, d, e, f, g, h, i, j, k) :
        """ Currently severely overfitted.
            Looks like the given values are simply too noisy to fit
            better than with error 2-4 or so.
        """
        #return (a*x**2 + b*x + c/x + d)
        return ( (a*x + b*x**2 + c*x**3 + d*x**4 + e*x**5)
               / (f + g*x + h*x**2 + i*x**3 + j*x**4 + k*x**5)
               )
    # end def _fit

    hrc_tbl_x, hrc_tbl_y = np.array (hrc_tbl2).T
    fit_params, dummy = curve_fit (_fit, hrc_tbl_x, hrc_tbl_y)

# end class Material

class Shaft :
    def __init__ (self, gearbox, T_ges) :
        self.gearbox = gearbox
        self.T_ges   = T_ges
    # end def __init__

    def add_gears (self, input_gear = None, output_gear = None) :
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
        gb = self.gearbox

        if output_gear is None :
            # Shaft 0
            assert input_gear is not None
            # Tangentiale Lagerkraft A, B
            self.F_Bt = self.F_t * gb.l_A / gb.l_AB
            self.F_At = self.F_t - self.F_Bt
            # Normale Lagerkraft A, B
            self.F_Bn = self.F_r * gb.l_A / gb.l_AB
            self.F_An = self.F_r - self.F_Bn
            # Axiale Lagerkraft A, B
            #self.F_Ba = FIXME
            #self.F_Aa = FIXME
        elif input_gear is None :
            # Shaft 2
            assert output_gear is not None
            F_t1 = output_gear.shaft.F_t
            F_r1 = output_gear.shaft.F_r
            # Tangentiale Lagerkraft A, B
            self.F_Bt = F_t1 * (gb.l_AB - gb.l_B) / (gb.l_AB)
            self.F_At = F_t1 - self.F_Bt
            # Normale Lagerkraft A, B
            self.F_Bn = F_r1 * (gb.l_AB - gb.l_B) / (gb.l_AB)
            self.F_An = F_r1 - self.F_Bn
            # Axiale Lagerkraft A, B
            self.F_Ba = 0
            self.F_Aa = 0
        else :
            # Shaft 1
            F_t0 = output_gear.shaft.F_t
            F_r0 = output_gear.shaft.F_r
            # Tangentiale Lagerkraft A, B
            self.F_Bt = \
                (F_t0 * gb.l_A + self.F_t * (gb.l_AB - gb.l_B)) / (gb.l_AB)
            self.F_At = F_t0 + self.F_t - self.F_Bt
            # Normale Lagerkraft A, B
            self.F_Bn = \
                (F_r0 * gb.l_A + self.F_r * (gb.l_AB - gb.l_B)) / (gb.l_AB)
            self.F_An = F_r0 + self.F_r - self.F_Bn
            # Axiale Lagerkraft A, B
            #self.F_Ba = FIXME
            #self.F_Aa = FIXME
        # Radiale Lagerkraft A, B
        self.F_Br = np.sqrt (self.F_Bt ** 2 + self.F_Bn ** 2)
        self.F_Ar = np.sqrt (self.F_At ** 2 + self.F_An ** 2)

        if output_gear is None :
            # Maximales Biegemoment
            self.M_bmax = self.F_Ar * gb.l_A / 1e3
            # Vergleichsmoment
        elif input_gear is None :
            self.M_bmax = self.F_Ar * (gb.l_AB - gb.l_B) / 1e3
        else :
            M_bz3 = self.F_Ar * gb.l_A / 1e3
            M_bz2 = self.F_Br * gb.l_B / 1e3
            # Maximales Biegemoment
            self.M_bmax = max (M_bz2, M_bz3)
        self.M_v = np.sqrt \
            (self.M_bmax ** 2 + .75 * (.7 * self.T_ges / 1e3) ** 2)
    # end def add_input_gear
# end class Shaft

class Zone_Factor (autosuper) :
    # Constants (for this project at least):
    alpha = np.pi * 20 / 180
    # Profilverschiebung (profile shift)
    profile_shift = [0, 0]

    def __init__ (self, z, beta = 0, alpha = None, **kw) :
        self.z    = np.array (z)
        if alpha is not None :
            self.alpha = alpha
        self.set_beta (beta)
    # end def __init__

    @property
    def profile_shift_normalized (self) :
        """ This is the term (x1 + x2) / (z1 + z2) in computation of Z_H
        """
        return sum (self.profile_shift) / sum (self.z)
    # end def profile_shift_normalized

    f_sh_200 = \
        ( ( 20.0,  5.0)
        , ( 40.0,  6.5)
        , (100.0,  7.0)
        , (200.0,  8.0)
        , (315.0, 10.0)
        , (560.0, 12.0)
        , (1e30,  16.0)
        )
    f_sh_200_1000 = \
        ( ( 20.0,  6.0)
        , ( 40.0,  7.0)
        , (100.0,  8.0)
        , (200.0, 11.0)
        , (315.0, 14.0)
        , (560.0, 18.0)
        , (1e30,  24.0)
        )
    f_sh_1000 = \
        ( ( 20.0, 10.0)
        , ( 40.0, 13.0)
        , (100.0, 18.0)
        , (200.0, 25.0)
        , (315.0, 30.0)
        , (560.0, 38.0)
        , (1e30,  50.0)
        )

    def flank_line_deformation (self, b, F_t) :
        """ f_sh in µm (TB 21-16)
            flank line deviation due to deformation (torsion and bending)
            Flankenlinienabweichung durch Verformung
            Zahnbreite b
            Note: Bei ungleichen b ist die kleinere Breite einzusetzen
        >>> z = Zone_Factor ([30, 30], 0)
        >>> print ("%.1f" % z.flank_line_deformation (20, 199 * 20))
        5.0
        >>> print ("%.1f" % z.flank_line_deformation (20, 200 * 20))
        6.0
        >>> print ("%.1f" % z.flank_line_deformation (20, 1000 * 20))
        6.0
        >>> print ("%.1f" % z.flank_line_deformation (20, 1001 * 20))
        10.0
        >>> print ("%.1f" % z.flank_line_deformation (560, 199 * 560))
        12.0
        >>> print ("%.1f" % z.flank_line_deformation (560, 200 * 560))
        18.0
        >>> print ("%.1f" % z.flank_line_deformation (560, 1000 * 560))
        18.0
        >>> print ("%.1f" % z.flank_line_deformation (560, 1001 * 560))
        38.0
        >>> print ("%.1f" % z.flank_line_deformation (1000, 199 * 1000))
        16.0
        >>> print ("%.1f" % z.flank_line_deformation (1000, 200 * 1000))
        24.0
        >>> print ("%.1f" % z.flank_line_deformation (1000, 1000 * 1000))
        24.0
        >>> print ("%.1f" % z.flank_line_deformation (1000, 1001 * 1000))
        50.0
        """
        ftb = F_t / b
        tbl = None
        if ftb < 200 :
            tbl = self.f_sh_200
        elif 200 <= ftb <= 1000 :
            tbl = self.f_sh_200_1000
        else :
            tbl = self.f_sh_1000
        assert tbl is not None
        k   = (b, 0)
        idx = bisect_left (tbl, k)
        assert 0 <= idx <= len (tbl) - 1
        assert b <= tbl [idx][0]
        return tbl [idx][1]
    # end def flank_line_deformation

    def set_beta (self, beta) :
        self.beta    = beta
        self.alpha_t = atan (np.tan (self.alpha) / np.cos (beta))
    # end def set_beta

    tt_by_din = dict \
        ((  ( 6, ((20,  8), (40,   9), (100,  10), (1e4,  11)))
         ,  ( 7, ((20, 11), (40,  13), (100,  14), (1e4,  16)))
         ,  ( 8, ((20, 16), (40,  18), (100,  20), (1e4,  22)))
         ,  ( 9, ((20, 25), (40,  28), (100,  28), (1e4,  32)))
         ,  (10, ((20, 36), (40,  40), (100,  45), (1e4,  50)))
         ,  (11, ((20, 56), (40,  63), (100,  71), (1e4,  80)))
         ,  (12, ((20, 90), (40, 100), (100, 110), (1e4, 125)))
        ))

    def tt_angle_deviation (self, b = None, din_quality = 6) :
        """ f_Hß in µm TB 21-16c
            Tooth trace angle deviation
            Flankenlinien-Winkel-Abweichung
        >>> z = Zone_Factor ([30, 30], 0)
        >>> z.tt_angle_deviation (20)
        8
        >>> z.tt_angle_deviation (21)
        9
        >>> z.tt_angle_deviation (40)
        9
        >>> z.tt_angle_deviation (41)
        10
        >>> z.tt_angle_deviation (100)
        10
        >>> z.tt_angle_deviation (101)
        11
        >>> z.tt_angle_deviation (20, 12)
        90
        >>> z.tt_angle_deviation (21, 12)
        100
        >>> z.tt_angle_deviation (40, 12)
        100
        >>> z.tt_angle_deviation (41, 12)
        110
        >>> z.tt_angle_deviation (100, 12)
        110
        >>> z.tt_angle_deviation (101, 12)
        125
        """
        if b is None :
            b = self.b
        tbl = self.tt_by_din [din_quality]
        k   = (b, 0)
        idx = bisect_left (tbl, k)
        assert 0 <= idx <= len (tbl) - 1
        assert b <= tbl [idx][0]
        return tbl [idx][1]
    # end def tt_angle_deviation

    def zone_factor (self, shift = None) :
        """ Zone factor Z_H. This typically gets the transmission index
            and computes all the other values from it.
            Note that we use a efficient method for computing the
            inverse of the involute. The largest angle we use is 37.43°
        >>> for beta_deg in 0, 5, 10, 15, 25, 35, 40 :
        ...     z  = Zone_Factor ([30, 30], beta_deg * np.pi / 180)
        ...     zf = z.zone_factor
        ...     print ("%.5f" % zf (shift = 0))
        2.49457
        2.48675
        2.46337
        2.42473
        2.30385
        2.13072
        2.02782
        >>> for shift in -0.02, 0.1 :
        ...     z  = Zone_Factor ([30, 30], 30 * np.pi / 180)
        ...     zf = z.zone_factor
        ...     print ("%.5f" % zf (shift = shift))
        2.66981
        1.70126
        """
        if shift is None :
            shift = self.profile_shift_normalized
        beta_b  = atan (np.tan (self.beta) * np.cos (self.alpha_t))
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
                self.set_beta (br)
                y.append (self.zone_factor (shift = shift))
            ax.plot (x, y)
        plt.show ()
    # end def plot_zone_factor
# end class Zone_Factor

class Gear (Zone_Factor) :

    # Elastizitätsfaktor Stahl-Stahl TB21-21b
    Z_E = 189.8
    # Submersion depth default if unspecified in constructor
    # submersion depth factor 2..5
    submersion_factor = 2

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
    din_quality = 6

    def __init__ (self, gb, materials, z, n_ein, shaft, psi_m, **kw) :
        self.__super.__init__ (z, **kw)
        self.gb        = gb
        self.materials = materials
        self.n_ein     = n_ein
        self.shaft     = shaft
        self.psi_m     = psi_m
        self.mod_ovl   = 0
        if 'din_quality' in kw :
            self.din_quality = kw ['din_quality']
        if 'submersion_factor' in kw :
            self.submersion_factor = kw ['submersion_factor']
        assert len (z) == 2
        assert len (materials) == 2
        # Zone factor Z_H
        self.Z_H = Z_H = self.zone_factor ()
        # Ritzel
        self.sigma_Hlim = sigma_Hlim = materials [0].sigma_h_lim
        # tatsächliches Übersetzungsverhältnis
        u_tat = self.u_tat = self.z [1] / self.z [0]
        # Betriebsmoment Eingangs-Welle
        self.T_ges = T_ges = self.shaft.T_ges
        # Use psi_dlim of Ritzel
        self.psi_dlim = psi_dlim = materials [0].psi_dlim ()
        # T_ges is Nm -> convert to mm
        # Stirnmodul Rad 1, Rad 2
        self.stirnmodul_calc = \
            ( (2 * T_ges * 1.2) / ((sigma_Hlim / 1.4) ** 2)
            * (u_tat + 1) / u_tat
            * (self.Z_E ** 2) * (Z_H ** 2)
            * 1 / (psi_dlim * self.z [0] ** 3)
            ) ** (1/3)
        self.normalmodul_calc = self.stirnmodul_calc * np.cos (self.beta)
        # lookup moduln
        idx = bisect_right (self.modul_DIN_780_I, self.normalmodul_calc)
        if idx >= len (self.modul_DIN_780_I) :
            self.mod_ovl += 1
            nm_I = self.modul_DIN_780_I [-1]
        else :
            nm_I = self.modul_DIN_780_I [idx]
            if nm_I < self.normalmodul_calc :
                nm_I = self.modul_DIN_780_I [idx + 1]
        idx = bisect_right (self.modul_DIN_780_II, self.normalmodul_calc)
        if idx >= len (self.modul_DIN_780_II) :
            self.mod_ovl += 1
            nm_II = self.modul_DIN_780_II [-1]
        else :
            nm_II = self.modul_DIN_780_II [idx]
            if nm_II < self.normalmodul_calc :
                nm_II = self.modul_DIN_780_II [idx + 1]
            # Overflow for nm_I ?
            if self.mod_ovl :
                assert self.mod_ovl == 1
                self.mod_ovl = 0
                nm_I = None
        self.normalmodul = nm_I
        # This only occurs if last value of table-I is too small but the
        # last item (70) of table-II matches
        if self.normalmodul is None :
            self.normalmodul = nm_II

        self.stirnmodul = self.normalmodul / np.cos (self.beta)
        # Only keep nm_II if it matches better than nm_I
        if  ( abs (nm_I  - self.normalmodul_calc)
            < abs (nm_II - self.normalmodul_calc)
            ) :
            nm_II = None
        self.D_R = self.z * self.stirnmodul
        self.D_K = self.D_R + 2 * self.normalmodul
        # Kopfspiel
        c_K      = 0.25 * self.normalmodul
        h_f      = self.normalmodul + c_K
        self.D_F = self.D_R - 2 * h_f
        # Durchmesser/Breitenverhältnis
        self.b = b = self.psi_m * self.normalmodul
        # z1 must be 2mm smaller than z0
        self.b_rad = b - 2
        # check breite
        self.psi_d = b / self.D_R # <= psi_dlim

        # v out in m/s, note that n_ein is per *minute*
        self.v = self.D_R [0] * n_ein * np.pi / 1000 / 60

        # d_RW gibt maximalen Durchmesser Ritzelwelle
        m_n = self.normalmodul
        self.d_RW = m_n * (z [0] - 2.5) / (1.1 * np.cos (self.beta))
        # d_W gibt maximalen Durchmesser Welle mit aufgestecktem Ritzel
        self.d_W  = m_n * (z [0] - 2.5) / (1.8 * np.cos (self.beta))
        # Achsabstand
        self.a = sum (self.D_R) / 2
    # end def __init__

    gfk1_beta0 = dict \
        (( ( 6,   9.6)
         , ( 7,  15.3)
         , ( 8,  24.5)
         , ( 9,  34.5)
         , (10,  53.5)
         , (11,  76.6)
         , (12, 122.5)
        ))
    gfk1_betanonnull = dict \
        (( ( 6,   8.5)
         , ( 7,  13.6)
         , ( 8,  21.8)
         , ( 9,  30.7)
         , (10,  47.7)
         , (11,  68.2)
         , (12, 109.1)
        ))
    def gear_factor_k1 (self) :
        if self.beta == 0 :
            return self.gfk1_beta0       [self.din_quality]
        else :
            return self.gfk1_betanonnull [self.din_quality]
    # end def gear_factor_k1

    gfqh = dict \
        (( ( 6,  1.32)
         , ( 7,  1.85)
         , ( 8,  2.59)
         , ( 9,  4.01)
         , (10,  6.22)
         , (11,  9.63)
         , (12, 14.9)
        ))
    def gear_factor_qh (self) :
        return self.gfqh [self.din_quality]
    # end def gear_factor_qh

    @property
    def out_T_ges (self) :
        return self.T_ges * self.u_tat
    # end def out_T_ges

    def profile_overlap (self) :
        """ Profilüberdeckung
        """
        m_t = self.stirnmodul
        D_b = self.D_b = self.D_R * np.cos (self.alpha_t)
        self.epsilon_alpha = \
            ( ( 0.5
              * ( np.sqrt (self.D_K [0] ** 2 - D_b [0] ** 2)
                + np.sqrt (self.D_K [1] ** 2 - D_b [1] ** 2)
                )
              )
            - self.a * np.sin (self.alpha_t)
            ) / (np.pi * m_t * np.cos (self.alpha_t))
        self.epsilon_alphan = self.epsilon_alpha / np.cos (self.beta) ** 2
        m_n = self.normalmodul
        self.epsilon_beta = self.b_rad * np.sin (self.beta) / (np.pi * m_n)
    # end def profile_overlap

    def tooth_root_strength (self) :
        """ Zahnfussfestigkeit
        """
        # FIXME: put in regression-test, which Shaft?
        self.F_tbase = F_tbase = self.shaft.F_t / self.gb.K_A
        m_n       = self.normalmodul
        # S. 1317 (323-4) TB 21-20d
        Y_X       = [m.size_factor_tooth (m_n) for m in self.materials]
        self.Y_X  = Y_X = sum (Y_X) / 2
        self.K_Ft = K_Ft = self.shaft.F_t / self.b_rad
        # S. 1310 (316) TB 21.14
        self.K_1  = K_1 = self.gear_factor_k1 ()
        # S. 1310 (316) TB 21.14
        # Geradverzahnung vs. Schrägverzahnung: Different value for K_2
        # depending on self.beta.
        self.K_2  = K_2 = 0.0087 if self.beta != 0 else 0.0193
        self.K_3  = K_3 = \
            ( 0.01 * self.z [0] * self.v
            * np.sqrt (self.u_tat ** 2 / (1 + self.u_tat) ** 2)
            )
        self.K_v  = K_v = 1 + (K_1 / K_Ft + K_2) * K_3
        # S. 1313 (319) TB 21.17
        c         = 1.0
        # µm S. 1313 (319) TB 21:16c
        self.f_Hbeta = f_Hbeta = self.tt_angle_deviation \
            (din_quality = self.din_quality)
        f_ma      = c * f_Hbeta
        # flank line deviation due to deformation (torsion and bending)
        f_sh      = self.flank_line_deformation (self.b_rad, self.shaft.F_t)
        F_betax   = f_ma + 1.33 * f_sh
        y_beta    = [m.y_beta (F_betax, self.v) for m in self.materials]
        # According to TB 21-17 we compute average if materials are different
        self.y_beta = y_beta = sum (y_beta) / 2
        K_Fm      = K_v * K_Ft
        F_betay   = F_betax - y_beta
        K_Hbeta1  = 1 + 10 * F_betay / K_Fm
        K_Hbeta2  = 2 * np.sqrt (10 * F_betay / K_Fm)
        self.K_Hbeta = K_Hbeta = K_Hbeta1 if K_Hbeta1 <= 2 else K_Hbeta2
        h         = (self.D_K [0] - self.D_F [0]) / 2
        self.K_bh = K_bh = self.b_rad / h
        self.N_F  = N_F = K_bh ** 2 / (1 + K_bh + K_bh ** 2)
        self.K_Fbeta = K_Fbeta  = K_Hbeta ** N_F
        if self.beta == 0 :
            self.K_Fges = K_Fges = self.gb.K_A * K_v * K_Fbeta
        else :
            self.K_Fges = K_Fges = \
                self.gb.K_A * K_v * self.epsilon_alphan * K_Fbeta
        b         = np.array ([self.b, self.b_rad])
        # Y_beta holds for beta < 30° (and is set to 30° in the
        # following formula, see DIN 3990 part 41 p. 13
        # we never have beta > 30°
        Y_beta    = (1 - (min (self.epsilon_beta, 1)
                         * self.beta / (120 * np.pi / 180)
                         )
                    )
        # Y_Fa:  S. 1316 (322) TB 21-19a
        self.Y_Fa = Y_Fa = np.array ([yfa (self.z [0]), yfa (self.z [1])])
        # Y_Sa:  S. 1316 (322) TB 21-19b
        self.Y_Sa = Y_Sa = np.array ([ysa (self.z [0]), ysa (self.z [1])])
        Y_eps     = 0.25 + 0.75 / self.epsilon_alphan
        sigma_F0  = F_tbase / (b * m_n) * Y_Fa * Y_Sa * Y_eps * Y_beta
        self.sigma_F0 = sigma_F0
        self.sigma_FE = sigma_FE = np.array \
            ([m.sigma_f_lim for m in self.materials])
        sigma_FP  = sigma_FE * self.gb.Y_NT * Y_X
        self.sigma_F = sigma_F = sigma_F0 * K_Fges
        # When computing for z0: f(z0), for z1: f(z1)
        self.S_F  = S_F = sigma_FP / sigma_F
    # end def tooth_root_strength

    def hertz_pressure (self) :
        """ Hertzsche Pressung
        """
        self.Z_beta = Z_beta = np.sqrt (np.cos (self.beta))
        self.Z_eps1 = Z_eps1 = np.sqrt \
            ( (4 - self.epsilon_alpha) / 3
            * (1 - self.epsilon_beta)
            + self.epsilon_beta / self.epsilon_alpha
            )
        self.Z_eps2 = Z_eps2 = np.sqrt (1 / self.epsilon_alpha)
        self.Z_eps  = Z_eps  = Z_eps1 if Z_eps1 < 1 else Z_eps2
        self.sigma_H0 = sigma_H0 = \
            ( self.Z_E * self.Z_H * Z_eps * Z_beta
            * np.sqrt ( self.F_tbase
                      / (self.b_rad * self.D_R [0])
                      * (self.u_tat + 1) / self.u_tat
                      )
            )
        Z_NT     = 1
        # S. 1317 (323-4) TB 21-20d
        m_n      = self.normalmodul
        Z_X      = [m.size_factor_flank (m_n) for m in self.materials]
        self.Z_X = Z_X = sum (Z_X) / 2
        # This does the correct thing (using the softer material) no
        # matter which is called
        self.Z_W = Z_W = self.materials [0].material_combination_factor \
            (self.materials [1])
        Z_LVR    = 0.92
        self.sigma_HP = sigma_HP = \
            ( np.array ([m.sigma_h_lim for m in self.materials])
            * Z_NT * Z_X * Z_W * Z_LVR
            )
        if self.beta == 0 :
            self.K_H = K_H = np.sqrt (self.gb.K_A * self.K_Hbeta * self.K_v)
        else :
            self.K_H = K_H = np.sqrt \
                (self.gb.K_A * self.K_Hbeta * self.K_v * self.epsilon_alphan)
        self.sigma_H = sigma_H = sigma_H0 * K_H

        self.S_H = sigma_HP / sigma_H
    # end def hertz_pressure

    def gewaltbruchsicherheit (self) :
        self.Y_S = Y_S = self.Y_Sa * (0.6 + 0.4 * self.epsilon_alphan)
        Y_deltarel_Tstat = 0.2 + 0.4 * Y_S
        Y_NTG = 2.5
        self.sigma_FGstat = sigma_FGstat = \
            self.sigma_FE * Y_deltarel_Tstat * Y_NTG
        self.S_G = S_G = sigma_FGstat / self.sigma_F
    # end def gewaltbruchsicherheit

    def constrain_S_G_0 (self) :
        return 2 - self.S_G [0]
    # end def constrain_S_G_0

    def constrain_S_G_1 (self) :
        return 2 - self.S_G [1]
    # end def constrain_S_G_1

    def constrain_S_H_min_0 (self) :
        return 1 - self.S_H [0]
    # end def constrain_S_H_min_0

    def constrain_S_H_min_1 (self) :
        return 1 - self.S_H [1]
    # end def constrain_S_H_min_1

    def constrain_S_H_max_0 (self) :
        return self.S_H [0] - 1.3
    # end def constrain_S_H_max_0

    def constrain_S_H_max_1 (self) :
        return self.S_H [1] - 1.3
    # end def constrain_S_H_max_1

    def constrain_S_F_min_0 (self) :
        return 1.4 - self.S_F [0]
    # end def constrain_S_H_min_0

    def constrain_S_F_min_1 (self) :
        return 1.4 - self.S_F [1]
    # end def constrain_S_H_min_1

    def constrain_S_F_max_0 (self) :
        return self.S_F [0] - 1.7
    # end def constrain_S_H_max_0

    def constrain_S_F_max_1 (self) :
        return self.S_F [1] - 1.7
    # end def constrain_S_H_max_1

    def constrain_v (self) :
        """ v must be less than a given number.
            Constraints for the GA must be <= 0
        """
        if self.beta > 0 :
            return self.v - 15
        return self.v - 10
    # end def constrain_v

    def constrain_epsilon_alphan (self) :
        """ Normal-Profilüberdeckung mindestens [1.1,1.25]
            min 1.1 better 1.25
        """
        return 1.25 - self.epsilon_alphan
    # end def constrain_epsilon_alphan

    def constrain_epsilon_beta (self) :
        """ Sprungüberdeckung (nur für Schrägvz) >1
        """
        if self.beta == 0 :
            return 0
        return 1 - self.epsilon_beta
    # end def constrain_epsilon_beta

# end class Gear

class Gearbox :
    """ The gearbox with all the sub-parts
    >>> HB = Material.HB
    >>> t = 'tempered'
    >>> m1 = Material ('QT30CrNiMo8',  HB, t, 320, 320, 240, 355, 610, 870, 2.4)
    >>> m2 = Material ('QT34CrNiMo6',  HB, t, 310, 310, 235, 345, 580, 840, 2.4)
    >>> m3 = Material ('QT36CrNiMo16', HB, t, 350, 350, 250, 365,  640,  915, 3)
    >>> m4 = Material ('QT30CrNiMo8',  HB, t, 320, 320, 240, 355, 610, 870, 2.4)
    >>> m  = [m1, m2, m3, m4]
    >>> z = [19, 89, 21, 76]
    >>> beta = 18 * np.pi / 180
    >>> n  = 3510
    >>> delta_b = 0
    >>> psi_m = [20, 20]
    >>> gb = Gearbox (m, z, beta, n, delta_b, psi_m)
    >>> g0 = gb.gears [0]
    >>> g1 = gb.gears [1]

    >>> print ("T_ges: %.4f" % g0.T_ges)
    T_ges: 204044.7988
    >>> print ("Z_H: %.4f" % g0.Z_H)
    Z_H: 2.3944
    >>> print ("sigma_Hlim: %.4f" % g0.sigma_Hlim)
    sigma_Hlim: 740.0000
    >>> print ("u_tat: %.4f" % g0.u_tat)
    u_tat: 4.6842
    >>> print ("psi_dlim: %.4f" % g0.psi_dlim)
    psi_dlim: 1.1000
    >>> print ("m_tcalc: %.4f" % g0.stirnmodul_calc)
    m_tcalc: 3.8759
    >>> print ("m_ncalc: %.4f" % g0.normalmodul_calc)
    m_ncalc: 3.6862
    >>> print ("m_t: %.4f" % g0.stirnmodul)
    m_t: 4.2058
    >>> print ("m_n: %.4f" % g0.normalmodul)
    m_n: 4.0000
    >>> print ("D_R: %.4f %.4f" % tuple (g0.D_R))
    D_R: 79.9111 374.3206
    >>> print ("D_K: %.4f %.4f" % tuple (g0.D_K))
    D_K: 87.9111 382.3206
    >>> print ("b: %.4f" % g0.b)
    b: 80.0000
    >>> print ("psi_d: %.4f %.4f" % tuple (g0.psi_d))
    psi_d: 1.0011 0.2137
    >>> print ("v: %.4f" % g0.v)
    v: 14.6863
    >>> print ("d_RW: %.4f" % g0.d_RW)
    d_RW: 63.0877
    >>> print ("d_W: %.4f" % g0.d_W)
    d_W: 38.5536
    >>> print ("epsilon_alpha: %.4f" % g0.epsilon_alpha)
    epsilon_alpha: 1.5751
    >>> print ("epsilon_alphan: %.4f" % g0.epsilon_alphan)
    epsilon_alphan: 1.7414
    >>> print ("epsilon_beta: %.4f" % g0.epsilon_beta)
    epsilon_beta: 1.9181
    >>> print ("N_F: %.4f" % g0.N_F)
    N_F: 0.8860
    >>> print ("K_Fbeta: %.4f" % g0.K_Fbeta)
    K_Fbeta: 2.0394
    >>> print ("K_Fges: %.4f" % g0.K_Fges)
    K_Fges: 7.0238
    >>> print ("Y_Fa: %.3f %.3f" % tuple (g0.Y_Fa))
    Y_Fa: 2.956 2.214
    >>> print ("Y_Sa: %.3f %.3f" % tuple (g0.Y_Sa))
    Y_Sa: 1.591 1.918
    >>> print ("Y_X: %.4f" % g0.Y_X)
    Y_X: 1.0000
    >>> print ("K_Ft: %.4f" % g0.K_Ft)
    K_Ft: 65.4717
    >>> print ("K_1: %.4f" % g0.K_1)
    K_1: 8.5000
    >>> print ("K_2: %.4f" % g0.K_2)
    K_2: 0.0087
    >>> print ("K_3: %.4f" % g0.K_3)
    K_3: 2.2995
    >>> print ("K_v: %.4f" % g0.K_v)
    K_v: 1.3185
    >>> print ("y_beta: %.3f" % g0.y_beta)
    y_beta: 8.527
    >>> print ("f_Hbeta: %d" % g0.f_Hbeta)
    f_Hbeta: 10
    >>> print ("sigma_F0: %.4f %.4f" % tuple (g0.sigma_F0))
    sigma_F0: 28.9547 26.8115
    >>> print ("sigma_F: %.4f %.4f" % tuple (g0.sigma_F))
    sigma_F: 203.3721 188.3186
    >>> print ("S_F: %.4f %.4f" % tuple (g0.S_F))
    S_F: 1.4628 1.5399
    >>> print ("Z_beta: %.4f" % g0.Z_beta)
    Z_beta: 0.9752
    >>> print ("Z_eps2: %.4f" % g0.Z_eps2)
    Z_eps2: 0.7968
    >>> print ("Z_eps: %.4f" % g0.Z_eps)
    Z_eps: 0.6897
    >>> print ("sigma_H0: %.4f" % g0.sigma_H0)
    sigma_H0: 248.8586
    >>> print ("Z_X: %.4f" % g0.Z_X)
    Z_X: 1.0000
    >>> print ("Z_W: %.4f" % g0.Z_W)
    Z_W: 1.0941
    >>> print ("sigma_HP: %.4f %.4f" % tuple (g0.sigma_HP))
    sigma_HP: 744.8753 714.6776
    >>> print ("K_H: %.4f" % g0.K_H)
    K_H: 2.7746
    >>> print ("sigma_H: %.4f" % g0.sigma_H)
    sigma_H: 690.4858
    >>> print ("S_H: %.4f %.4f" % tuple (g0.S_H))
    S_H: 1.0788 1.0350
    >>> print ("S_G: %.4f %.4f" % tuple (g0.S_G))
    S_G: 3.7492 4.5991

    >>> print ("T_ges: %.4f" % g1.T_ges)
    T_ges: 955788.7945
    >>> print ("Z_H: %.4f" % g1.Z_H)
    Z_H: 2.4946
    >>> print ("sigma_Hlim: %.4f" % g1.sigma_Hlim)
    sigma_Hlim: 777.5000
    >>> print ("u_tat: %.4f" % g1.u_tat)
    u_tat: 3.6190
    >>> print ("psi_dlim: %.4f" % g1.psi_dlim)
    psi_dlim: 1.1000
    >>> print ("m_tcalc: %.4f" % g1.stirnmodul_calc)
    m_tcalc: 5.9334
    >>> print ("m_ncalc: %.4f" % g1.normalmodul_calc)
    m_ncalc: 5.9334
    >>> print ("m_t: %.4f" % g1.stirnmodul)
    m_t: 6.0000
    >>> print ("m_n: %.4f" % g1.normalmodul)
    m_n: 6.0000
    >>> print ("D_R: %.4f %.4f" % tuple (g1.D_R))
    D_R: 126.0000 456.0000
    >>> print ("D_K: %.4f %.4f" % tuple (g1.D_K))
    D_K: 138.0000 468.0000
    >>> print ("b: %.4f" % g1.b)
    b: 120.0000
    >>> print ("psi_d: %.4f %.4f" % tuple (g1.psi_d))
    psi_d: 0.9524 0.2632
    >>> print ("v: %.4f" % g1.v)
    v: 4.9436
    >>> print ("d_RW: %.4f" % g1.d_RW)
    d_RW: 100.9091
    >>> print ("d_W: %.4f" % g1.d_W)
    d_W: 61.6667
    >>> print ("epsilon_alpha: %.4f" % g1.epsilon_alpha)
    epsilon_alpha: 1.6941
    >>> print ("epsilon_alphan: %.4f" % g1.epsilon_alphan)
    epsilon_alphan: 1.6941
    >>> print ("epsilon_beta: %.4f" % g1.epsilon_beta)
    epsilon_beta: 0.0000
    >>> print ("N_F: %.4f" % g1.N_F)
    N_F: 0.8869
    >>> print ("K_Fbeta: %.4f" % g1.K_Fbeta)
    K_Fbeta: 1.9107
    >>> print ("K_Fges: %.4f" % g1.K_Fges)
    K_Fges: 3.1884
    >>> print ("Y_Fa: %.3f %.3f" % tuple (g1.Y_Fa))
    Y_Fa: 2.865 2.245
    >>> print ("Y_Sa: %.3f %.3f" % tuple (g1.Y_Sa))
    Y_Sa: 1.613 1.887
    >>> print ("Y_X: %.4f" % g1.Y_X)
    Y_X: 0.9940
    >>> print ("K_Ft: %.4f" % g1.K_Ft)
    K_Ft: 128.5699
    >>> print ("K_1: %.4f" % g1.K_1)
    K_1: 15.3000
    >>> print ("K_2: %.4f" % g1.K_2)
    K_2: 0.0193
    >>> print ("K_3: %.4f" % g1.K_3)
    K_3: 0.8134
    >>> print ("K_v: %.4f" % g1.K_v)
    K_v: 1.1125
    >>> print ("y_beta: %.3f" % g1.y_beta)
    y_beta: 11.242
    >>> print ("f_Hbeta: %d" % g1.f_Hbeta)
    f_Hbeta: 16
    >>> print ("sigma_F0: %.4f %.4f" % tuple (g1.sigma_F0))
    sigma_F0: 44.9736 41.9209
    >>> print ("sigma_F: %.4f %.4f" % tuple (g1.sigma_F))
    sigma_F: 143.3960 133.6627
    >>> print ("Z_X: %.4f" % g1.Z_X)
    Z_X: 1.0000
    >>> print ("Z_W: %.4f" % g1.Z_W)
    Z_W: 1.0882
    >>> print ("S_F: %.4f %.4f" % tuple (g1.S_F))
    S_F: 2.1315 2.2124
    >>> print ("S_H: %.4f %.4f" % tuple (g1.S_H))
    S_H: 1.0815 1.0293
    >>> print ("S_G: %.4f %.4f" % tuple (g1.S_G))
    S_G: 5.4923 6.4781

    >>> print ("s_z: %.4f %.4f"  % tuple (gb.s_z))
    s_z: 12.0000 18.0000
    >>> print ("s_Wg: %.4f" % gb.s_Wg)
    s_Wg: 8.0000
    >>> print ("s_b: %.4f"  % gb.s_b)
    s_b: 34.0000
    >>> print ("s_d: %.4f"  % gb.s_d)
    s_d: 12.0000
    >>> print ("a: %.4f %.4f" % tuple (gb.a))
    a: 227.1158 291.0000
    >>> print ("l_Gi: %.4f" % gb.l_Gi)
    l_Gi: 826.0714
    >>> print ("l_12: %.4f" % gb.l_12)
    l_12: 117.0000
    >>> print ("l_0G: %.4f" % gb.l_0G)
    l_0G: 52.0000
    >>> print ("l_1G: %.4f" % gb.l_1G)
    l_1G: 78.0000
    >>> print ("b_Gi: %.4f" % gb.b_Gi)
    b_Gi: 247.0000
    >>> print ("l_GL: %.4f" % gb.l_GL)
    l_GL: 25.0000
    >>> print ("l_A: %.4f" % gb.l_A)
    l_A: 77.0000
    >>> print ("l_B: %.4f" % gb.l_B)
    l_B: 103.0000
    >>> print ("l_AB: %.4f" % gb.l_AB)
    l_AB: 297.0000

    >>> print ("F_t: %.4f"    % gb.shaft [0].F_t)
    F_t: 5106.7930
    >>> print ("F_r: %.4f"    % gb.shaft [0].F_r)
    F_r: 1954.3746
    >>> print ("F_Br: %.4f"   % gb.shaft [0].F_Br)
    F_Br: 1417.6270
    >>> print ("F_Bt: %.4f"   % gb.shaft [0].F_Bt)
    F_Bt: 1323.9834
    >>> print ("F_Ar: %.4f"   % gb.shaft [0].F_Ar)
    F_Ar: 4050.3630
    >>> print ("F_At: %.4f"   % gb.shaft [0].F_At)
    F_At: 3782.8097

    #>>> print ("F_Ba: %.4f"   % gb.shaft [0].F_Ba)
    #>>> print ("F_Aa: %.4f"   % gb.shaft [0].F_Aa)
    >>> print ("M_bmax: %.4f" % gb.shaft [0].M_bmax)
    M_bmax: 311.8780
    >>> print ("M_v: %.4f"    % gb.shaft [0].M_v)
    M_v: 335.5122

    >>> print ("F_t: %.4f"    % gb.shaft [1].F_t)
    F_t: 15171.2507
    >>> print ("F_r: %.4f"    % gb.shaft [1].F_r)
    F_r: 5521.8837
    >>> print ("F_Br: %.4f"   % gb.shaft [1].F_Br)
    F_Br: 11963.2902
    >>> print ("F_Bt: %.4f"   % gb.shaft [1].F_Bt)
    F_Bt: 11233.8239
    >>> print ("F_Ar: %.4f"   % gb.shaft [1].F_Ar)
    F_Ar: 9649.1212
    >>> print ("F_At: %.4f"   % gb.shaft [1].F_At)
    F_At: 9044.2198

    #>>> print ("F_Ba: %.4f"   % gb.shaft [1].F_Ba)
    #>>> print ("F_Aa: %.4f"   % gb.shaft [1].F_Aa)
    >>> print ("M_bmax: %.4f" % gb.shaft [1].M_bmax)
    M_bmax: 1232.2189
    >>> print ("M_v: %.4f"    % gb.shaft [1].M_v)
    M_v: 1361.6484

    >>> print ("F_t: %.4f"    % gb.shaft [2].F_t)
    F_t: 54905.4787
    >>> print ("F_r: %.4f"    % gb.shaft [2].F_r)
    F_r: 19983.9600
    >>> print ("F_Br: %.4f"   % gb.shaft [2].F_Br)
    F_Br: 10545.8320
    >>> print ("F_Bt: %.4f"   % gb.shaft [2].F_Bt)
    F_Bt: 9909.8405
    >>> print ("F_Ar: %.4f"   % gb.shaft [2].F_Ar)
    F_Ar: 5599.0758
    >>> print ("F_At: %.4f"   % gb.shaft [2].F_At)
    F_At: 5261.4102
    >>> print ("F_Ba: %.4f"   % gb.shaft [2].F_Ba)
    F_Ba: 0.0000
    >>> print ("F_Aa: %.4f"   % gb.shaft [2].F_Aa)
    F_Aa: 0.0000
    >>> print ("M_bmax: %.4f" % gb.shaft [2].M_bmax)
    M_bmax: 1086.2207
    >>> print ("M_v: %.4f"    % gb.shaft [2].M_v)
    M_v: 2361.5695
    >>> print ("T_ges: %.4f" % gb.shaft [2].T_ges)
    T_ges: 3459045.1612
    """
    # Anwendungsfaktor K_A
    K_A  = 1.5
    # Leistung P
    P    = 50e3
    # Lebensdauerfaktor
    Y_NT = 1.0

    def __init__ (self, materials, z, beta, n, delta_b, psi_m, x_1, **kw) :
        """ Note that n is minutes^-1
        """
        if 'P' in kw :
            self.P = kw ['P']
        self.submersion_factor = x_1
        assert len (z) == 4
        assert len (materials) == 4
        n2 = n * z [0] / z [1]
        # Betriebsmoment (minutes not seconds)
        T_ges = self.P * self.K_A / (2 * n * np.pi) * 60 * 1000
        s = self.shaft = []
        g = self.gears = []
        s.append (Shaft (self, T_ges))
        g.append \
            ( Gear
                ( self, materials [:2], z [:2]
                , n, s [-1], beta = beta, psi_m = psi_m [0]
                , submersion_factor = self.submersion_factor
                )
            )
        s.append (Shaft (self, g [-1].out_T_ges))
        g.append \
            ( Gear
                ( self, materials [2:], z [2:]
                , n2, s [-1], psi_m = psi_m [1]
                , din_quality = 7
                )
            )
        s.append (Shaft (self, g [-1].out_T_ges))

        self.factor = (z [1] * z [3]) / (z [0] * z [2])
        self.delta_b = delta_b

        g = self.gears
        # Distance of box to first wheel
        self.s_z = s_z = [3 * x.normalmodul for x in g]
        # Wandstärke Gehäuse (s_12)
        self.s_Wg = s_Wg = 8 # FIXME from table GJS RM S.785/20.6
        # Flanschbreite
        self.s_b = s_b = 3 * s_Wg + 10
        # Flanschdicke
        self.s_d = s_d = 1.5 * s_Wg
        # Gehäuse Innenlänge
        self.a = a = [x.a for x in g]
        self.l_Gi = l_Gi = \
            ( sum (a)
            + (g [0].D_K [0] + g [1].D_K [1]) / 2
            + sum (s_z)
            )
        # Gehäuse Innenbreite
        self.l_12 = l_12 = ((g [0].b - 2) + (g [1].b)) / 2 + s_z [1]
        self.l_0G = l_0G = g [0].b / 2 + s_z [0]
        self.l_1G = l_1G = g [1].b / 2 + s_z [1]
        self.b_Gi = b_Gi = l_12 + l_0G + l_1G

        self.l_GL = l_GL = delta_b + s_Wg + s_b / 2
        self.l_A  = l_A  = l_GL + l_0G
        self.l_B  = l_B  = l_GL + l_1G
        self.l_AB = l_AB = l_A  + l_B + l_12

        s [0].add_gears (input_gear  = g [0])
        s [1].add_gears (output_gear = g [0], input_gear = g [1])
        s [2].add_gears (output_gear = g [1])

        for g in self.gears :
            g.profile_overlap ()

        self.oil ()

        for g in self.gears :
            g.tooth_root_strength ()
            g.hertz_pressure ()
            g.gewaltbruchsicherheit ()

    # end def __init__

    @property
    def cost (self) :
        g = self.gears
        return sum (( sum (m.cost_factor for m in g [0].materials)
                    , sum (m.cost_factor for m in g [1].materials)
                   ))
    # end def cost

    def oil (self) :
        """ Öltauchschmierung hier tatsächlich wichtig beide Großräder
            zu beschreiben
        """
        g0 = self.gears [0]
        g1 = self.gears [1]
        self.t = t = []
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
        self.factor = ( self.args.input_rotation_speed
                      / self.args.output_rotation_speed
                      )
        # Teeth (4 parameters)
        minmax = [(self.args.min_tooth, self.args.max_tooth)] * 4
        # Material: Index into table "materials" above
        minmax.extend ([(0, len (self.materials))] * 4)
        # Schrägungswinkel beta: in degree in gene, converted to rad later
        minmax.append ((8, 20))
        # Eintauchfaktor Rad1 x_1
        minmax.append ((2, 5))
        # delta_b (should be minimized)
        minmax.append ((0, 1000))
        # Modul/Breitenverhältnis TB 21-13b psi_m for each gear
        minmax.append ((20, 35))
        minmax.append ((20, 35))
        # Compute number of constraint-methods in Gearbox
        num_constraint = 2
        self.constraints = []
        self.gear_constr = []
        for n in Gearbox.__dict__ :
            if n.startswith ('constrain_') :
                num_constraint += 1
                self.constraints.append (n)
        for n in Gear.__dict__ :
            if n.startswith ('constrain_') :
                num_constraint += 2
                self.gear_constr.append (n)
        d = dict \
            ( maximize             = False
            , num_eval             = 5 + num_constraint
            , num_constraint       = num_constraint
            , sum_constraints      = True
            , pop_size             = self.args.pop_size
            , num_replace          = self.args.pop_size
            , select_type          = pga.PGA_SELECT_LINEAR
            , pop_replace_type     = pga.PGA_POPREPL_NSGA_II
            , mutation_only        = True
            , mutation_type        = pga.PGA_MUTATION_DE
            , DE_crossover_prob    = 0.8
            , DE_crossover_type    = pga.PGA_DE_CROSSOVER_BIN
            , DE_variant           = pga.PGA_DE_VARIANT_RAND
            , DE_scale_factor      = 0.85
            , init                 = minmax
            , max_GA_iter          = self.args.generations
            , print_options        = [pga.PGA_REPORT_STRING]
            , mutation_bounce_back = True
            )
        if args.random_seed :
            d ['random_seed'] = args.random_seed
        self.__super.__init__ (float, len (minmax), **d)
    # end def __init__

    def err (self, *z) :
        f = self.factor
        return abs (f - (z [1] * z [3]) / (z [0] * z [2])) / f * 100
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
        beta    = self.get_allele (p, pop, 8) * np.pi / 180
        x_1     = self.get_allele (p, pop, 9)
        delta_b = self.get_allele (p, pop, 10)
        psi_m   = [self.get_allele (p, pop, x) for x in (11, 12)]
        n_in    = self.args.input_rotation_speed
        power   = self.args.power
        g = Gearbox (mat, z, beta, n_in, delta_b, psi_m, x_1, P=power)
        return g
    # end def phenotype

    def evaluate (self, p, pop) :
        gb   = self.phenotype (p, pop)
        z    = list (gb.gears [0].z) + list (gb.gears [1].z)
        gc   = gcd (* z [:2]) + gcd (*z [2:])
        ret  = [ self.err (*z), gb.cost, gb.l_Gi
               , gb.gears [0].normalmodul, gb.gears [1].normalmodul
               ]
        ret.extend ([self.err (*z) - 1.5, gc - 2])
        for n in self.constraints :
            m = getattr (gb, n)
            ret.append (m ())
        for n in self.gear_constr :
            for g in gb.gears :
                m = getattr (g, n)
                ret.append (m ())
        return ret
    # end def evaluate

    def print_string (self, file, p, pop) :
        gb = self.phenotype (p, pop)
        z = []
        for g in gb.gears :
            z.extend (g.z)
        print (z, file = file)
        print ("Gear Error: %12.9f%%" % self.err (*z), file = file)
        print ("Cost: %.3f" % gb.cost, file = file)
        print ("Size: %.3f" % gb.l_Gi, file = file)
        print ("Random seed: %d" % self.random_seed, file = file)
        if self.args.verbose :
            for n in self.constraints :
                print ("%s: %s" % (n, getattr (gb, n)()), file = file)
            for i, g in enumerate (gb.gears) :
                for n in self.gear_constr :
                    v = getattr (g, n)()
                    print ("Gear %s %s: %s" % (i, n, v), file = file)
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
        ( '-g', '--generations'
        , type    = int
        , help    = "Maximum number of generations, default=%(default)s"
        , default = 2000
        )
    cmd.add_argument \
        ( '-i', '--input-rotation-speed'
        , help    = "Input rotation speed (Eingangs-Drehzahl) n_in, "
                    "default=%(default)s"
        , type    = float
        , default = 3510
        )
    cmd.add_argument \
        ( '-l', '--min-tooth'
        , type    = int
        , default = 17
        )
    cmd.add_argument \
        ( '-o', '--output-rotation-speed'
        , help    = "Output rotation speed (Ausgangs-Drehzahl) n_out, "
                    "default=%(default)s"
        , type    = float
        , default = 207
        )
    cmd.add_argument \
        ( '--plot-zone-factor'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '--plot-hardness'
        , action  = 'store_true'
        )
    cmd.add_argument \
        ( '-P', '--power'
        , type    = float
        , help    = "Power, default=%(default)s"
        , default = 50e3
        )
    cmd.add_argument \
        ( '-p', '--pop-size'
        , type    = int
        , help    = "Population size, default=%(default)s"
        , default = 100
        )
    cmd.add_argument \
        ( '-r', '--random-seed'
        , type    = int
        , default = 42
        )
    cmd.add_argument \
        ( '-u', '--max-tooth'
        , type    = int
        , default = 200
        )
    cmd.add_argument \
        ( '-v', '--verbose'
        , action  = 'store_true'
        )
    args = cmd.parse_args ()
    if args.plot_zone_factor :
        z = Zone_Factor ([30, 30])
        z.plot_zone_factor ()
    elif args.plot_hardness :
        m = Material ('', Material.HRC, 'tempered', 0, 0, 0, 0, 1000, 1000, 1)
        m.plot_hardness ()
    elif args.check :
        go = Gear_Optimizer (args)
        z  = [int (i) for i in args.check.split (',')]
        print ("Factor: %f" % go.factor)
        print ("Gear Error: %12.9f%%" % go.err (*z))
    else :
        go = Gear_Optimizer (args)
        go.run ()
