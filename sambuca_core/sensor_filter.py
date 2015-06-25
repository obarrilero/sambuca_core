# -*- coding: utf-8 -*-
""" Contains functions for working with Sensor Filters. """

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)
from builtins import *

import numpy as np
import pandas as pd
import xlrd


def apply_sensor_filter(spectra, normalised_response_function):
    """Applies a sensor filter to a spectra using the given spectral
    response function.

    Args:
        spectra (array-like): The input spectra.
        normalised_response_function (matrix-like): The spectral sensitivity
            matrix.
            The first dimension determines the number of output bands.
            The second dimension represents the proportional contribution of
            each of the input bands to an output band. The size must match the
            number of bands in the input spectra.

    Returns:
        ndarray: The filtered spectra.

    """

    return np.dot(
        normalised_response_function,
        spectra) / normalised_response_function.sum(1)


def load_sensor_filters_excel(filename, normalise=False, sheet_names=None):
    """ Loads sensor filters from an Excel file. Both new style XLSX and
    old-style XLS formats are supported.

    Args:
        filename (str): full path to the Excel file.
        normalise (boolean): Determines whether the filter bands will be
            normalised after loading.
        sheet_names (list): Optional list of worksheet names to load.
            The default is to attempt to load all worksheets.

    Returns:
        dictionary: A dictionary of numpy.ndarrays containing all sensor
            filters, keyed by filter name inferred from the sheet name.
    """

    sensor_filters = {}
    with pd.ExcelFile(filename) as excel_file:
        # default is all sheets
        if not sheet_names:
            sheet_names = excel_file.sheet_names

        for sheet in sheet_names:
            try:
                filter_df = excel_file.parse(sheet)  # the sheet as a DataFrame
                # OK, we have the data frame. Let's process it...

                if normalise:
                    # normalise all bands relative to the strongest
                    # as this preserves the relative band strengths
                    filter_df = filter_df / max(filter_df.max())

                sensor_filters[sheet] = filter_df.values.transpose()
            except xlrd.biffh.XLRDError as xlrd_error:
                pass
                # TODO: log warning about invalid sheet

    return sensor_filters
