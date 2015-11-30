# -*- coding: utf-8 -*-
# Ensure compatibility of Python 2 with Python 3 constructs
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)
from builtins import *

from pkg_resources import resource_filename

import numpy as np
from scipy.io import readsav

import sambuca_core as sbc


# @skip
class TestForwardModel(object):

    """Sambuca forward model test class
    """

    @classmethod
    def setup_class(cls):
        # load the test values
        filename = resource_filename(
            sbc.__name__,
            './tests/data/forward_model_test_data.sav')
        cls.data = readsav(filename)
        cls.unpack_parameters()
        cls.unpack_input_spectra()
        cls.unpack_input_params()
        cls.unpack_results()
        cls.unpack_substrates()
        # TODO: Why are my outputs not closer? IDL data appears to read in as single precision.
        cls.rtol = 1e-3
        cls.atol = 5e-4

    @classmethod
    def unpack_parameters(cls):
        # The IDL code has the parameters packed into a structure called ZZ.
        # Magic numbers here are drawn directly from the IDL code.
        zz = cls.data.zz
        cls.chl = zz[1]
        cls.cdom = zz[2]
        cls.nap = zz[3]
        cls.x_ph_lambda0x = zz[4]
        cls.x_nap_lambda0x = zz[5]
        cls.slope_cdom = zz[6]
        cls.slope_nap = zz[7]
        cls.a_nap_lambda0nap = zz[8]
        cls.slope_backscatter = zz[9]
        cls.q1 = zz[10]
        cls.q2 = zz[11]
        cls.q3 = zz[12]
        cls.H = zz[13]

    @classmethod
    def unpack_input_spectra(cls):
        s = cls.data.sambuca.input_spectra[0]
        cls.wav = s.wl[0]
        cls.awater = s.awater[0]
        # cls.bbwater = s.bbwater[0]
        cls.aphy_star = s.aphy_star[0]
        # cls.acdom_star = s.acdom_star[0]

    @classmethod
    def unpack_input_params(cls):
        p = cls.data.sambuca.input_params[0]
        cls.theta_air = p.theta_air
        cls.lambda0cdom = p.lambda0cdom
        cls.lambda0nap = p.lambda0tr
        cls.lambda0x = p.lambda0x

    @classmethod
    def unpack_substrates(cls):
        spectra = cls.data.sambuca.inputr[0].spectra[0]
        # it appears that in the test I set up, the substrates are both the same
        cls.substrate1 = spectra[:,0]
        cls.substrate2 = spectra[:,1]

    @classmethod
    def unpack_results(cls):
        r = cls.data.spectra
        cls.expected_substrate_r = r.substrater[0]
        cls.expected_rrs = r.rrs[0]
        cls.expected_rrsdp = r.rrsdp[0]
        cls.expected_kd = r.kd[0]
        cls.expected_kub = r.kub[0]
        cls.expected_kuc = r.kuc[0]
        cls.expected_a = r.a[0]
        cls.expected_bb = r.bb[0]

    def test_validate_data(self):
        assert self.data
        assert len(self.data.zz) == 15

        spectra = [
            self.wav,
            self.awater,
            self.aphy_star,
            self.expected_substrate_r,
            self.expected_rrs,
            self.expected_rrsdp,
            self.expected_kd,
            self.expected_kub,
            self.expected_kuc,
            self.substrate1,
            self.substrate2,
        ]
        for array in spectra:
            assert len(array) == 551

        assert self.lambda0cdom == 440

    def run_forward_model(self):
        return sbc.forward_model(
            self.chl,
            self.cdom,
            self.nap,
            self.H,
            self.substrate1,
            self.wav,
            self.awater,
            self.aphy_star,
            551,
            substrate_fraction=self.q1,
            substrate2=self.substrate2,
            x_ph_lambda0x=self.x_ph_lambda0x,
            x_nap_lambda0x=self.x_nap_lambda0x,
            slope_cdom=self.slope_cdom,
            slope_nap=self.slope_nap,
            a_nap_lambda0nap=self.a_nap_lambda0nap,
            slope_backscatter=self.slope_backscatter,
            lambda0cdom=self.lambda0cdom,
            lambda0nap=self.lambda0nap,
            theta_air=self.theta_air,
            water_refractive_index=1.333,  # The hard-coded IDL value
            # self.off_nadir,
        )

    def test_substrate_r(self):
        results = self.run_forward_model()
        assert np.allclose(results.r_substratum, self.expected_substrate_r)

    def test_substrate_r_single_substrate(self):
        # set the second substrate to None, and adjust the expectation that
        # results.r_substratum will equal substrate1
        self.substrate2 = None
        results = self.run_forward_model()
        assert np.allclose(results.r_substratum, self.substrate1)

    def test_rrs(self):
        results = self.run_forward_model()
        assert np.allclose(
            results.rrs,
            self.expected_rrs,
            atol=self.atol,
            rtol=self.rtol)

    def test_rrs_deep(self):
        results = self.run_forward_model()
        assert np.allclose(
            results.rrsdp,
            self.expected_rrsdp,
            atol=self.atol,
            rtol=self.rtol)

    def test_kd(self):
        results = self.run_forward_model()
        assert np.allclose(
            results.kd,
            self.expected_kd,
            atol=self.atol,
            rtol=self.rtol)

    def test_kub(self):
        results = self.run_forward_model()
        assert np.allclose(
            results.kub,
            self.expected_kub,
            atol=self.atol,
            rtol=self.rtol)

    def test_kuc(self):
        results = self.run_forward_model()
        assert np.allclose(
            results.kuc,
            self.expected_kuc,
            atol=self.atol,
            rtol=self.rtol)

    def test_a(self):
        results = self.run_forward_model()
        assert np.allclose(
            results.a,
            self.expected_a,
            atol=self.atol,
            rtol=self.rtol)

    def test_bb(self):
        results = self.run_forward_model()
        assert np.allclose(
            results.bb,
            self.expected_bb,
            atol=self.atol,
            rtol=self.rtol)
