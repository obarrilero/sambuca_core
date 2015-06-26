# -*- coding: utf-8 -*-
# Ensure compatibility of Python 2 with Python 3 constructs
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)

import sambuca_core as sbc
import numpy as np
from pkg_resources import resource_filename


class TestExcelSensorFilterLoading(object):

    """ Sensor filter loading tests. """
    def test_unknown_worksheet_doesnt_throw(self):
        file = resource_filename(
            sbc.__name__,
            'tests/data/sensor_filters/sensor_filters.xlsx')

        sbc.load_sensor_filters_excel(
            file,
            sheet_names=['non_existant'])

    def test_valid_worksheet_load(self):
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

        actual_filter = loaded_filters[expected_name]

        assert isinstance(actual_filter, np.ndarray)
        assert actual_filter.shape == (3, 551)
        assert np.allclose(actual_filter[0, ], 1)
        assert np.allclose(actual_filter[1, ], 2)
        assert np.allclose(actual_filter[2, ], 3)
