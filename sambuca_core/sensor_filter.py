# -*- coding: utf-8 -*-
""" Contains functions for working with Sensor Filters. """

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)
from builtins import *

import os

import numpy as np
import pandas as pd
import spectral.io.envi as envi
import spectral.io.spyfile as spyfile
import xlrd

from .exceptions import UnsupportedDataFormatError, DataValidationError
from .utility import list_files, strictly_increasing


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

def _validate_filter_dataframe(filter_df):
    """ Internal function to validate a sensor filter data frame.

    Args:
        filter_df (pandas.DataFrame): the sensor filter

    Returns:
        bool: True if the filter is valid; otherwise false.
    """

    wavelengths = filter_df.index

    # are the band-centre wavelengths strictly increasing?
    if not strictly_increasing(wavelengths):
        return False

    # Are the wavelength spacings acceptable?
    # For now, only sensor filters that are specified with exact
    # 1nm bands are supported.
    band_diffs = np.ediff1d(wavelengths)
    if band_diffs.min() < 1.0 or band_diffs.max() > 1.0:
        # TODO: log warning about interpolation/averaging not being supported
        return False

    # The dtype of every column needs to be a numpy-compatible number
    if len(filter_df.select_dtypes(include=[np.number]).columns) \
       != len(filter_df.columns):
        return False

    return True

def _normalise_df(df):
    # normalise all bands relative to the strongest
    # as this preserves the relative band strengths
    return df / max(df.max())
    # TODO: If per-band normalisation is required, this line will do it. Not that this loses the relative band strengths
    # return df / df.max()

def load_sensor_filter_spectral_library(
        directory,
        base_filename,
        normalise=False):
    """ Loads a single sensor filter from an ENVI spectral library.

    Args:
        directory (str): Directory containing the sensor filter file.
        base_filename (str): The filename without the extension or '.'
            preceeding the extension.
        normalise (bool): If true, the filter will be normalised.

    Returns:
        numpy.array: The band-centre wavelengths.
        numpy.array: The sensor filter.
    """

    base_filename = os.path.join(directory, base_filename)
    file_pattern = '{0}.{1}'

    # load the spectral library
    try:
        sl = envi.open(
            file_pattern.format(base_filename, 'hdr'),
            file_pattern.format(base_filename, 'lib'))
    except spyfile.FileNotFoundError as exception:
        raise FileNotFoundError(exception)

    # convert to a DataFrame
    df = pd.DataFrame(sl.spectra.transpose(), index=sl.bands.centers)
    df.columns = ['Band {0}'.format(x+1) for x in range(len(df.columns))]

    if not _validate_filter_dataframe(df):
        raise DataValidationError(
            'Spectral library {0} failed validation'.format(
                base_filename))

    if normalise:
        df = _normalise_df(df)

    return np.array(df.index), df.values.transpose()

# TODO: do I want an option to clip the filters to a specific range of 1nm bands?
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
                df = excel_file.parse(sheet)  # the sheet as a DataFrame
                # OK, we have the data frame. Let's process it...
                if not _validate_filter_dataframe(df):
                    continue

                if normalise:
                    df = _normalise_df(df)

                sensor_filters[sheet] = (
                    np.array(df.index),
                    df.values.transpose())

            except xlrd.biffh.XLRDError as xlrd_error:
                continue
                # TODO: log warning about invalid sheet

    return sensor_filters

def _merge_dictionary(target, new_items):
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
        dict: target, with all unique items merged from new items.
    """

    for name, _filter in new_items.items():
        if name in target:
            # TODO: add logging
            # logging.getLogger(__name__).warn('Sensor filter %s already defined', name)
            pass
        else:
            target[name] = _filter

    return target

def load_sensor_filters(
        path,
        normalise=False,
        spectral_library_name_parser=None):
    """" Loads all valid sensor filters from the given location.

    Args:
        path (str): The directory path to scan for sensor filters.
        normalise (boolean): Determines whether the filter bands will be
            normalised after loading.
        spectral_library_name_parser (function): If supplied, this function
            accepts a single string argument (the full path to a spectral
            library file) and returns the sensor filter name that will be used
            in the dictionary of results.

    Returns:
        dict: A dictionary of 2-tuples of numpy.ndarrays.
            The first element contains the band centre wavelengths of the input
            bands, while the second element contains the filter.
            Dictionary is keyed by filter name inferred from the sheet name.

            Note that names are not disambiguated, so that if more than one
            filter has the same name, only the first will be returned and no
            error will be raised.
    """
    # TODO: add logging
    # logging.getLogger(__name__).info(
    #     'Loading Sensor filters from %s', path)

    sensor_filters = {}
    new_filters = {}

    # excel files
    for file in list_files(path, ['xls', 'xlsx']):
        try:
            new_filters = load_sensor_filters_excel(file, normalise=normalise)
        except UnsupportedDataFormatError as ex:
            # logging.getLogger(__name__).exception(ex)
            # TODO: logging
            pass
        _merge_dictionary(sensor_filters, new_filters)

    # Spectral Libraries
    for file in list_files(path, ['lib']):
        try:
            base_name, _ = os.path.splitext(os.path.basename(file))

            if spectral_library_name_parser:
                name = spectral_library_name_parser(file)
            else:
                name = base_name

            loaded_filter = load_sensor_filter_spectral_library(
                path,
                base_name,
                normalise=normalise)

            if name not in sensor_filters:
                sensor_filters[name] = loaded_filter
        except UnsupportedDataFormatError as ex:
            # TODO: logging.getLogger(__name__).exception(ex)
            pass
        _merge_dictionary(sensor_filters, new_filters)

    return sensor_filters
