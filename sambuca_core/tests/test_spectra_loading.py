# -*- coding: utf-8 -*-
# Ensure compatibility of Python 2 with Python 3 constructs
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)

from os.path import basename, splitext

import numpy as np
import pytest
from pkg_resources import resource_filename

import sambuca_core as sbc

def spectra_name(base_name, band_name):
    return '{0}:{1}'.format(base_name, band_name)

class TestSpectralLibrarySubstrateLoading(object):

    def test_valid_load_hi3(self):
        directory = resource_filename(
            sbc.__name__,
            'tests/data/substrates')
        base_name = 'HI_3'

        loaded_substrates = sbc.load_envi_spectral_library(
            directory,
            base_name)

        expected_spectra = [spectra_name(base_name, x) for x in ['Acropora', 'sand', 'Turf Algae']]
        assert isinstance(loaded_substrates, dict)

        for expected_name in expected_spectra:
            assert expected_name in loaded_substrates

        wavelengths, sand = loaded_substrates['HI_3:sand']

        assert len(wavelengths) == 602
        assert len(sand) == 602
        assert isinstance(wavelengths, np.ndarray)
        assert isinstance(sand, np.ndarray)
        assert np.allclose(wavelengths, (range(350, 952, 1)))

    def test_load_missing_file(self):
        directory = resource_filename(
            sbc.__name__,
            'tests/data/substrates')
        base_name = 'missing_file'

        with pytest.raises(FileNotFoundError):
            sbc.load_envi_spectral_library(directory, base_name)
