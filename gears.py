#!/usr/bin/python3

from __future__ import print_function, division
from rsclib.autosuper import autosuper
from argparse import ArgumentParser
from math import gcd, sqrt, pi
import pga
import sys

class Werkstoff :
    # Einheit Härtegrad
    HB   = 0
    HRC  = 1
    HRCN = 2

    def __init__ \
        ( self
        , name
        , unit
        , hardness_min,    hardness_max
        , delta_f_lim_min, delta_f_lim_max
        , delta_h_lim_min, delta_h_lim_max
        , cost_factor
        )
        self.name            = name
        self.unit            = unit
        self.hardness_min    = hardness_min
        self.hardness_max    = hardness_max
        self.delta_f_lim_min = delta_f_lim_min
        self.delta_f_lim_max = delta_f_lim_max
        self.delta_h_lim_min = delta_h_lim_min
        self.delta_h_lim_min = delta_h_lim_max
        self.cost_factor     = cost_factor
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
        if self.unit == self.HB
            if self.hardness < 180 :
                return 1.2
            return 1.0
        elif self.unit == self.HRC
            return 0.8
        return 0.5
    # end def psi_d

# end class Werkstoff

class Gears (pga.PGA, autosuper) :
    """ Example from presentation by Deb 2008
    """


    # Name, Härtegrad-Einheit Flankenhärte, Zahnfußdauerfestigkeit (min/max)
    # Zahnflankendauerfestigkeit (min/max), rel Werkstoffkosten
    HB   = Werkstoff.HB
    HRC  = Werkstoff.HRC
    HRCN = Werkstoff.HRCN
    werkstoffe = \
        ( Werkstoff ( 'S235JR'
                    ,  HB, 120, 120, 125, 190,  315,  430, 1
                    )
        , Werkstoff ( 'E295'
                    ,  HB, 160, 160, 140, 210,  350,  485, 1.1
                    )
        , Werkstoff ( 'E335'
                    ,  HB, 190, 190, 160, 225,  375,  540, 1.7
                    )
        , Werkstoff ( 'C45E_N'
                    ,  HB, 190, 190, 160, 260,  470,  590, 1.7
                    )
        , Werkstoff ( 'QT34CrMo4'
                    ,  HB, 270, 270, 220, 335,  540,  800, 2.4
                    )
        , Werkstoff ( 'QT42CrMo4'
                    ,  HB, 300, 300, 230, 335,  540,  800, 2.4
                    )
        , Werkstoff ( 'QT34CrNiMo6'
                    ,  HB, 310, 310, 235, 345,  580,  840, 2.4
                    )
        , Werkstoff ( 'QT30CrNiMo8'
                    ,  HB, 320, 320, 240, 355,  610,  870, 2.7
                    )
        , Werkstoff ( 'QT36CrNiMo16'
                    ,  HB, 350, 350, 250, 365,  640,  915, 3
                    )
        , Werkstoff ( 'UH +FH CrMo TB20-1(Nr.19-22)'
                    , HRC, 50,  50, 230, 380,  980, 1275, 5
                    )
        , Werkstoff ( 'UH -FH CrMo TB20-1(Nr.19-22)'
                    , HRC, 50,  50, 150, 230,  980, 1275, 4
                    )
        , Werkstoff ( 'EH +FH CrMo TB20-1(Nr.19-22)'
                    , HRC, 56,  56, 270, 410, 1060, 1330, 7
                    )
        , Werkstoff ( 'EH -FH CrMo TB20-1(Nr.19-22)'
                    , HRC, 56,  56, 150, 230, 1060, 1330, 6
                    )
        , Werkstoff ( 'QT42CrMo4N'
                    , HRC, 48,  57, 260, 430,  780, 1215, 4
                    )
        , Werkstoff ( 'QT16MnCr5N'
                    , HRC, 48,  57, 260, 430,  780, 1215, 4
                    )
        , Werkstoff ( 'C45E_NN'
                    , HRCN, 30,  45, 225, 290,  650,  780, 3
                    )
        , Werkstoff ( '16MnCr5N'
                    , HRCN, 45,  57, 225, 385,  650,  950, 3.5
                    )
        , Werkstoff ( 'QT34Cr4CN'
                    , HRCN, 55,  60, 300, 450, 1100, 1350, 5.5
                    )
        , Werkstoff ( '16MnCr5EGm20'
                    , HRCN, 58,  62, 310, 525, 1300, 1650, 9
                    )
        , Werkstoff ( '15CrNi6EGm16'
                    , HRCN, 58,  62, 310, 525, 1300, 1650, 10
                    )
        , Werkstoff ( '18CrNiMo7-6EGm16'
                    , HRCN, 58,  62, 310, 525, 1300, 1650, 11
                    )
        )

    def __init__ (self, args) :
        self.args   = args
        self.factor = self.args.numerator / self.args.denominator
        # Teeth (4 parameters)
        minmax = [(self.args.min_tooth, self.args.max_tooth)] * 4
        # Stahl: Index into table "werkstoffe" above
        minmax.extend ([(0, len (self.werkstoffe))] * 4)
        # Schrägungswinkel
        minmax.append ((0, 20))
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

    def genotype (self, p, pop) :
        z = []
        for i in range (4) :
            z.append (round (self.get_allele (p, pop, i)))
        fac = (z [0] * z [2]) / (z [1] * z [3])
        werks = []
        for i in range (4) :
            stahl = int (self.get_allele (p, pop, i + 4))
            if stahl >= len (self.werkstoffe) :
                stahl = len (self.werkstoffe) - 1
            stahl = self.werkstoffe [stahl]
            werks.append (stahl)
        beta  = self.get_allele (p, pop, 8)
        x_1   = self.get_allele (p, pop, 9)
        # Anwendungsfaktor K_A
        K_A = 1.5
        # Eingangsdrehzahl n_ein
        n_ein = self.args.numerator
        # Leistung P
        P = 50e3
        # Durchmesser/Breitenverhältnis
        # Stirnmodul Rad 1, Rad 2
        stirnmodul   = []
        normalmodul  = []
        n_r          = self.args.numerator
        Z_E          = 189.8
        for i in range (2) :
            stahl      = werks [i]
            delta_Hlim = stahl.delta_h_lim
            psi_d      = stahl.psi_d
            u_tat      = z [2*i+1] / z [2*i]
            # Betriebsmoment Welle
            T_ges = P * K_A / (2 * n_r * pi)
            m = sqrt \
                ( (2 * T_ges * 1.2) / ((delta_Hlim / 1.4) ** 2)
                * (u_tat + 1) / u_tat
                * (Z_E ** 2) * (Z_H2 ** 2)
                * 1 / (psi_d * z [2*i] ** 3)
                )
            stirnmodul.append (m)
            normalmodul.append (stirnmodul * cos (beta))
            n_r = n_r * z [1] / z [0]
    # end def genotype

    def evaluate (self, p, pop) :
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

# end class Gears

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
    pg = Gears (args)
    if args.check :
        z = [int (i) for i in args.check.split (',')]
        print ("Factor: %f" % pg.factor)
        print ("Gear Error: %12.9f%%" % pg.err (*z))
    else :
        pg.run ()

