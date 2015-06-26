# -*- coding: utf-8 -*-
# Ensure compatibility of Python 2 with Python 3 constructs
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)

import numpy as np
from pkg_resources import resource_filename

import sambuca_core as sbc


class TestExcelSensorFilterLoading(object):

    """ Sensor filter loading tests. """
    def test_unknown_worksheet_doesnt_throw(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        sbc.load_sensor_filters_excel(
            file,
            sheet_names=['non_existant'])

    def test_load_single_worksheet(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        expected_name = '3_band_350_900'
        loaded_filters = sbc.load_sensor_filters_excel(
            file,
            normalise=False,
            sheet_names=[expected_name])

        assert len(loaded_filters) == 1
        assert isinstance(loaded_filters, dict)
        assert expected_name in loaded_filters

    def test_load_all_worksheets(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        loaded_filters = sbc.load_sensor_filters_excel(file)
        assert len(loaded_filters) == 3

    def test_load_multiple_skips_invalid_and_missing_sheets(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        good_names = ['3_band_350_900', '5_band_400_800']
        missing_names = ['Monty_Hall', 'Monty_Python']
        invalid_names = ['Deliberately_Invalid', 'wavelengths_out_of_sequence']
        names = good_names + missing_names + invalid_names
        loaded_filters = sbc.load_sensor_filters_excel(
            file,
            normalise=False,
            sheet_names=names)

        assert len(loaded_filters) == len(good_names)

    def test_normalise(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        expected_name = '3_band_350_900'
        loaded_filters = sbc.load_sensor_filters_excel(
            file,
            normalise=True,
            sheet_names=[expected_name])
        actual_filter = loaded_filters[expected_name][1]

        assert np.allclose(actual_filter[0, ], 1.0/3.0)
        assert np.allclose(actual_filter[1, ], 2.0/3.0)
        assert np.allclose(actual_filter[2, ], 1.0)

    def test_valid_worksheet(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        expected_name = '3_band_350_900'
        loaded_filters = sbc.load_sensor_filters_excel(
            file,
            normalise=False,
            sheet_names=[expected_name])
        actual_filter = loaded_filters[expected_name][1]

        assert isinstance(actual_filter, np.ndarray)
        assert actual_filter.shape == (3, 551)
        assert np.allclose(actual_filter[0, ], 1)
        assert np.allclose(actual_filter[1, ], 2)
        assert np.allclose(actual_filter[2, ], 3)

    def test_undersampled_worksheet(self):
        """ Current functionality is to skip any sensor filters that are not
        already specified at exact 1nm bands.
        When interpolation is implemented, this test will need to change to
        reflect the new expectations.
        """
        # TODO: update test when band interpolation is implemented
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        expected_name = '10nm_3_bands_350_900'
        loaded_filters = sbc.load_sensor_filters_excel(
            file,
            normalise=False,
            sheet_names=[expected_name])

        assert len(loaded_filters) == 0

    def test_oversampled_worksheet(self):
        """ Current functionality is to skip any sensor filters that are not
        already specified at exact 1nm bands.
        When band-averaging is implemented, this test will need to change to
        reflect the new expectations.
        """
        # TODO: update test when band averaging is implemented
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        expected_name = '0.1nm_3_bands_350_355'
        loaded_filters = sbc.load_sensor_filters_excel(
            file,
            normalise=False,
            sheet_names=[expected_name])

        assert len(loaded_filters) == 0

    def test_wavelengths(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')
        expected_names = ['3_band_350_900', '5_band_400_800', '4_band_300_1000']
        loaded_filters = sbc.load_sensor_filters_excel(
            file,
            normalise=False,
            sheet_names=expected_names)
        expected_wavelengths = {
            expected_names[0]: (range(350, 901, 1)),
            expected_names[1]: (range(400, 801, 1)),
            expected_names[2]: (range(300, 1001, 1))}

        assert len(loaded_filters) == len(expected_names)

        for name in expected_names:
            expected = expected_wavelengths[name]
            actual = loaded_filters[name][0]
            assert len(expected) == len(actual)
            assert np.allclose(expected, actual)
