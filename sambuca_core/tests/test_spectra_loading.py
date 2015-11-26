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


class TestSubstrateLoading(object):

    def test_valid_load_spectral_library(self):
        directory = resource_filename(
            sbc.__name__,
            'tests/data/substrates')
        base_name = 'HI_3'

        loaded_substrates = sbc.load_envi_spectral_library(
            directory,
            base_name)

        expected_spectra = [spectra_name(base_name, x) for x in ['Acropora', 'sand', 'Turf Algae']]
        assert isinstance(loaded_substrates, dict)
        assert len(loaded_substrates) == 3

        for expected_name in expected_spectra:
            assert expected_name in loaded_substrates

        wavelengths, sand = loaded_substrates['HI_3:sand']

        assert len(wavelengths) == 602
        assert len(sand) == 602
        assert isinstance(wavelengths, np.ndarray)
        assert isinstance(sand, np.ndarray)
        assert np.allclose(wavelengths, (range(350, 952, 1)))

    def test_load_whole_directory(self):
        directory = resource_filename(
            sbc.__name__,
            'tests/data/substrates')

        loaded_substrates = sbc.load_all_spectral_libraries(directory)
        assert isinstance(loaded_substrates, dict)

        # we expect 3 spectra in the HI_3 file, 10 in the
        # Moreton_Bay_speclib_final file, and another 11 in the xlsx file
        assert len(loaded_substrates) == 24

        # HI_3 is tested above, lets check some values from the
        # Moreton Bay data ...
        mb_names = ['Zostera muelleri',
                    'Halophila ovalis',
                    'Halophila spinulosa',
                    'Syringodium isoetifolium',
                    'Cymodocea serrulata',
                    'green algae',
                    'brown algae',
                    'brown Mud',
                    'light brown Mud',
                    'white Sand']

        expected_names = [spectra_name('Moreton_Bay_speclib_final', x) for x in mb_names]
        for expected_name in expected_names:
            assert expected_name in loaded_substrates

        wavelengths, mud = loaded_substrates['Moreton_Bay_speclib_final:brown Mud']

        assert len(wavelengths) == 600
        assert len(mud) == 600
        assert isinstance(wavelengths, np.ndarray)
        assert isinstance(mud, np.ndarray)
        assert np.allclose(wavelengths, (range(350, 950, 1)))

    def test_load_missing_file(self):
        directory = resource_filename(
            sbc.__name__,
            'tests/data/substrates')
        base_name = 'missing_file'

        with pytest.raises(FileNotFoundError):
            sbc.load_envi_spectral_library(directory, base_name)

    def test_load_excel(self):
        filename = resource_filename(
            sbc.__name__,
            'tests/data/substrates/Moreton_Bay_speclib.xlsx')

        loaded_substrates = sbc.load_excel_spectral_library(filename)
        assert isinstance(loaded_substrates, dict)
        assert len(loaded_substrates) == 11

        mb_names = ['weird_substrate',  # loaded from second worksheet
                    'Zostera muelleri',
                    'Halophila ovalis',
                    'Halophila spinulosa',
                    'Syringodium isoetifolium',
                    'Cymodocea serrulata',
                    'green algae',
                    'brown algae',
                    'brown Mud',
                    'light brown Mud',
                    'white Sand']

        expected_names = [spectra_name('Moreton_Bay_speclib', x) for x in mb_names]
        for expected_name in expected_names:
            assert expected_name in loaded_substrates

        wavelengths, white_sand = loaded_substrates['Moreton_Bay_speclib:white Sand']

        assert len(wavelengths) == 600
        assert len(white_sand) == 600
        assert isinstance(wavelengths, np.ndarray)
        assert isinstance(white_sand, np.ndarray)
        assert np.allclose(wavelengths, (range(350, 950, 1)))
