#!/usr/bin/python3

import numpy as np
from rsclib.autosuper import autosuper
from argparse import ArgumentParser
from math import gcd, sqrt, pi, sin, cos, tan, atan
import pga
import sys

class Material :
    # Einheit Härtegrad
    # FIXME: Should this become one of NORMAL_ANNEALED, TEMPERED, etc
    #        (see below)?
    HB   = 0
    HRC  = 1
    HRCN = 2

    # Resulting phi_dlim from shaft bearing (Aufhängung / Lager)
    # symetrical   O--X--O
    phi_dlim_symmetrical = dict \
        ( normal_annealed = 1.6
        , tempered        = 1.4
        , case_hardened   = 1.1
        , nitrified       = 0.8
        )
    # asymetrical  O-X---O
    phi_dlim_asymetrical = dict \
        ( normal_annealed = 1.3
        , tempered        = 1.1
        , case_hardened   = 0.9
        , nitrified       = 0.6
        )
    # flying       O---O-X
    phi_dlim_flying = dict \
        ( normal_annealed = 0.8
        , tempered        = 0.7
        , case_hardened   = 0.6
        , nitrified       = 0.4
        )
    material_types = set (k for k in phi_dlim_flying)

    dlim_by_bearing = dict \
        ( symmetrical   = phi_dlim_symmetrical
        , asymmetrical  = phi_dlim_asymetrical
        , flying        = phi_dlim_flying
        )

    def __init__ \
        ( self
        , name
        , material_type
        , hardness_min,    hardness_max
        , delta_f_lim_min, delta_f_lim_max
        , delta_h_lim_min, delta_h_lim_max
        , cost_factor
        ) :
        self.name               = name
        self.material_type      = material_type
        #self.unit               = unit
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

    def phi_dlim (self, shaft_bearing) :
        """ Compute *minimum* phi_d from given shaft bearing
            and the parameter of the material
        """
        return self.dlim_by_bearing [shaft_bearing][self.material_type]
    # end phi_dlim

# end class Material

class Gearbox :

    def __init__ (self, materials, z, beta, n_ein) :
        self.materials = materials
        self.z         = z
        self.beta      = beta
        self.n_ein     = n_ein
        assert len (z) == 4
        assert len (materials) == 4
        self.fac        = (z [0] * z [2]) / (z [1] * z [3])
        # Anwendungsfaktor K_A
        self.K_A = 1.5
        # Eingangsdrehzahl n_ein
        self.n_ein = n_ein
        # Leistung P
        self.P = 50e3
        # Durchmesser/Breitenverhältnis
        # Modul/Breitenverhältnis TB 21-13b small-> easier; FIXABLE TB
        self.phi_m = 20
        # Stirnmodul Rad 1, Rad 2
        self.stirnmodul   = []
        # FIXME: see n_ein same value?
        ### self.n_r          = self.args.numerator
        n_r = n_ein # FIXME ?????
        self.Z_E          = 189.8
        for i in range (2) :
            stahl      = self.materials [i]
            delta_Hlim = stahl.delta_h_lim
            ### FIXME: phi_dlim ???
            ###psi_d      = stahl.psi_d
            psi_d = 1 # FIXME
            u_tat      = z [2*i+1] / z [2*i]
            # Betriebsmoment Welle
            T_ges = self.P * self.K_A / (2 * n_r * pi)
            Z_H2  = 1 # FIXME
            m = sqrt \
                ( (2 * T_ges * 1.2) / ((delta_Hlim / 1.4) ** 2)
                * (u_tat + 1) / u_tat
                * (self.Z_E ** 2) * (Z_H2 ** 2)
                * 1 / (psi_d * z [2*i] ** 3)
                )
            self.stirnmodul.append (m)
            n_r = n_r * z [1] / z [0]
        self.stirnmodul  = np.array (self.stirnmodul)
        self.normalmodul = self.stirnmodul * cos (beta)

        ### max:

        ### normalmodul = m_n


        # Zahnbreite
        ### b = phi_m * m_n
        # check breite
        ### phi_d = b / D_R # = phi_dlim

        # check umfangs v
        ### v = D_Ritzel * n_ein * pi
        ### v <= 10
        ###     if beta == 0
        ###     else 15

        # m_n Normalmodul

        # d_RW gibt maximalen durchmesser Ritzelwelle
        ### d_RW = m_n * (z_R - 2.5) / (1.1 * cos (beta))
        # d_W gibt maximalen Durchmesser Welle mit aufgestecktem Ritzel
        ### d_W  = m_n * (z_R - 2.5) / (1.8 * cos (beta))

        # Kopfkreisdurchmesser
        ### D_KRad = D_Rad + 2 * m_n
        # Profilüberdeckung
        ### epsilon_alpha = (  0.5
        ###                 * sqrt (D_KRitzel ** 2 - D_bRitzel ** 2)
        ###                 + sqrt (D_KRad ** 2 - D_bRad ** 2)
        ###                 - a * sin (alpha_t)
        ###                 / pi * m_t * cos (alpha_t)
        ###                 )
        # Normal-Profilüberdeckung zwischen [1.1,1.25]
        ### epsilon_alphan = epsilon_alpha / cos (beta) ** 2
        # Sprungüberdeckung (nur für Schrägvz) >1
        ### epsilon_beta = b_Rad * sin (beta) / pi * m_n

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

    def zone_factor (self, beta = None, alpha = pi * 20 / 180) :
        """ Zone factor Z_H. This is normally tabulated because it seems
            hard to give a closed formula if the "Profilverschiebungsfaktor" 
            is != 0, i.e. (x1 + x2) != 0. So the currently-implemented
            method can only deal with (x1 + x2) == 0
        >>> m  = Material ('S235JR', 'normal_annealed'
        ...               , 120, 120, 125, 190, 315, 430, 1)
        >>> gb = Gearbox ([m] * 4, [30] * 4, 15, 1)
        >>> print ("%.5f" % gb.zone_factor (0))
        2.49457
        >>> print ("%.5f" % gb.zone_factor (5 * pi / 180))
        2.48675
        >>> print ("%.5f" % gb.zone_factor (10 * pi / 180))
        2.46331
        >>> print ("%.5f" % gb.zone_factor (15 * pi / 180))
        2.42443
        >>> print ("%.5f" % gb.zone_factor (25 * pi / 180))
        2.30154
        >>> print ("%.5f" % gb.zone_factor (35 * pi / 180))
        2.14
        >>> print ("%.5f" % gb.zone_factor (40 * pi / 180))
        2.04
        """
        if beta is None :
            beta = self.beta
        beta_b  = atan (tan (beta) * cos (alpha))
        alpha_t = atan (tan (alpha) / cos (beta))
        return 1 / cos (alpha_t) * sqrt (2 * cos (beta_b) / tan (alpha_t))
    # end def zone_factor

# end class Gearbox

class Gear_Optimizer (pga.PGA, autosuper) :
    """
    """
    # Name, Härtegrad-Einheit Flankenhärte, Zahnfußdauerfestigkeit (min/max)
    # Zahnflankendauerfestigkeit (min/max), rel Material cost
    # FIXME: The HB, HRC etc should probably be replaced with the
    # material_type name above. This is a simple string.
    HB   = Material.HB
    HRC  = Material.HRC
    HRCN = Material.HRCN
#    materials = \
#        ( Material ( 'S235JR'
#                   ,  HB, 120, 120, 125, 190,  315,  430, 1
#                   )
#        , Material ( 'E295'
#                   ,  HB, 160, 160, 140, 210,  350,  485, 1.1
#                   )
#        , Material ( 'E335'
#                   ,  HB, 190, 190, 160, 225,  375,  540, 1.7
#                   )
#        , Material ( 'C45E_N'
#                   ,  HB, 190, 190, 160, 260,  470,  590, 1.7
#                   )
#        , Material ( 'QT34CrMo4'
#                   ,  HB, 270, 270, 220, 335,  540,  800, 2.4
#                   )
#        , Material ( 'QT42CrMo4'
#                   ,  HB, 300, 300, 230, 335,  540,  800, 2.4
#                   )
#        , Material ( 'QT34CrNiMo6'
#                   ,  HB, 310, 310, 235, 345,  580,  840, 2.4
#                   )
#        , Material ( 'QT30CrNiMo8'
#                   ,  HB, 320, 320, 240, 355,  610,  870, 2.7
#                   )
#        , Material ( 'QT36CrNiMo16'
#                   ,  HB, 350, 350, 250, 365,  640,  915, 3
#                   )
#        , Material ( 'UH +FH CrMo TB20-1(Nr.19-22)'
#                   , HRC, 50,  50, 230, 380,  980, 1275, 5
#                   )
#        , Material ( 'UH -FH CrMo TB20-1(Nr.19-22)'
#                   , HRC, 50,  50, 150, 230,  980, 1275, 4
#                   )
#        , Material ( 'EH +FH CrMo TB20-1(Nr.19-22)'
#                   , HRC, 56,  56, 270, 410, 1060, 1330, 7
#                   )
#        , Material ( 'EH -FH CrMo TB20-1(Nr.19-22)'
#                   , HRC, 56,  56, 150, 230, 1060, 1330, 6
#                   )
#        , Material ( 'QT42CrMo4N'
#                   , HRC, 48,  57, 260, 430,  780, 1215, 4
#                   )
#        , Material ( 'QT16MnCr5N'
#                   , HRC, 48,  57, 260, 430,  780, 1215, 4
#                   )
#        , Material ( 'C45E_NN'
#                   , HRCN, 30,  45, 225, 290,  650,  780, 3
#                   )
#        , Material ( '16MnCr5N'
#                   , HRCN, 45,  57, 225, 385,  650,  950, 3.5
#                   )
#        , Material ( 'QT34Cr4CN'
#                   , HRCN, 55,  60, 300, 450, 1100, 1350, 5.5
#                   )
#        , Material ( '16MnCr5EGm20'
#                   , HRCN, 58,  62, 310, 525, 1300, 1650, 9
#                   )
#        , Material ( '15CrNi6EGm16'
#                   , HRCN, 58,  62, 310, 525, 1300, 1650, 10
#                   )
#        , Material ( '18CrNiMo7-6EGm16'
#                   , HRCN, 58,  62, 310, 525, 1300, 1650, 11
#                   )
#        )

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
        ( '-r', '--random-seed'
        , type    = int
        )
    args = cmd.parse_args ()
    go = Gear_Optimizer (args)
    if args.check :
        z = [int (i) for i in args.check.split (',')]
        print ("Factor: %f" % go.factor)
        print ("Gear Error: %12.9f%%" % go.err (*z))
    else :
        go.run ()
