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

from .exceptions import UnsupportedDataFormatError
from .utility import list_files


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
        dict: A dictionary of 2-tuples of numpy.ndarrays.
            The first element contains the band centre wavelengths of the input
            bands, while the second element contains the filter.
            Dictionary is keyed by filter name inferred from the sheet name.
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

                sensor_filters[sheet] = (
                    filter_df.index,
                    filter_df.values.transpose())
            except xlrd.biffh.XLRDError as xlrd_error:
                pass
                # TODO: log warning about invalid sheet

    return sensor_filters

def merge_dictionary(target, new_items):
    """ Merges a dictionary of sensor new_filters into the master set,
    warning when a duplicate name is detected. Keys from new_items that
    are already present in target will generate warnings without modifying
    target.

    And yes, I know there are builtin methods to merge dictionaries (update),
    but I wanted finer control over handling for existing keys.

    Args:
        target (dictionary): The destination dictionary.
        new_items (dictionary): The dictionary of new items to merge.

    Returns:
        dictionary: target, with all unique items merged from new items.
    """

    for name, _filter in new_items.items():
        if name in target:
            # TODO: add logging
            # logging.getLogger(__name__).warn('Sensor filter %s already defined', name)
            pass
        else:
            target[name] = _filter

    return target

def load_sensor_filters(path):
    """" Loads all valid sensor filters from the given location.

    Args:
        path (str): The directory path to scan for sensor filters.

    Returns:
        dict: A dictionary of 2-tuples of numpy.ndarrays.
            The first element contains the band centre wavelengths of the input
            bands, while the second element contains the filter.
            Dictionary is keyed by filter name inferred from the sheet name.
    """
    # TODO: add logging
    # logging.getLogger(__name__).info(
    #     'Loading Sensor filters from %s', path)

    sensor_filters = {}
    new_filters = {}

    # excel files
    for file in list_files(path, ['xls', 'xlsx']):
        # logging.getLogger(__name__).info('\t %s', file)
        try:
            # logging.getLogger(__name__).info('Loading %s', file)
            new_filters = load_sensor_filters_excel(file)
        except UnsupportedDataFormatError as ex:
            # logging.getLogger(__name__).exception(ex)
            # TODO: logging
            pass

    merge_dictionary(sensor_filters, new_filters)
    new_filters.clear()

    # TODO: CSV files
    # TODO: Spectral Libraries

    return sensor_filters
