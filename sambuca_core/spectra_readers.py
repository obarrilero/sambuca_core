# -*- coding: utf-8 -*-
""" Contains functions for loading collections of spectra from Sambuca
spectral database directories. """

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
from .utility import list_files, strictly_increasing, merge_dictionary

def _validate_spectra_dataframe(spectra_dataframe):
    """ Internal function to validate a spectra data frame.

    Args:
        spectra_dataframe (pandas.DataFrame): the

    Returns:
        bool: True if the spectra is valid; otherwise false.
    """

    wavelengths = spectra_dataframe.index

    # are the band-centre wavelengths strictly increasing?
    if not strictly_increasing(wavelengths):
        return False

    # Are the wavelength spacings acceptable?
    # For now, only spectra that are specified with exact
    # 1nm bands are supported.
    band_diffs = np.ediff1d(wavelengths)
    if band_diffs.min() < 1.0 or band_diffs.max() > 1.0:
        # TODO: log warning about interpolation/averaging not being supported
        return False

    # The dtype of every column needs to be a numpy-compatible number
    if len(spectra_dataframe.select_dtypes(include=[np.number]).columns) \
       != len(spectra_dataframe.columns):
        return False

    return True


def load_envi_spectral_library(
        directory,
        base_filename):
    """ Loads spectra from an ENVI spectral library.

    Args:
        directory (str): Directory containing the spectral library file.
        base_filename (str): The filename without the extension or '.'
            preceeding the extension.

    Returns:
        dict: A dictionary of 2-tuples of numpy.ndarrays.
            The first element contains the band centre wavelengths,
            while the second element contains the spectra.
            The dictionary is keyed by spectra name, formed by concatenation
            of the file and band names. This allows multiple spectra from multiple
            files to be unambigiously collected into a dictionary.
    """

    full_filename = os.path.join(directory, base_filename)
    file_pattern = '{0}.{1}'

    # load the spectral library
    try:
        spectral_library = envi.open(
            file_pattern.format(full_filename, 'hdr'),
            file_pattern.format(full_filename, 'lib'))
    except spyfile.FileNotFoundError as exception:
        raise FileNotFoundError(exception)

    # convert to a DataFrame for processing
    dataframe = pd.DataFrame(
        spectral_library.spectra.transpose(),
        index=spectral_library.bands.centers)
    dataframe.columns = spectral_library.names

    if not _validate_spectra_dataframe(dataframe):
        raise DataValidationError(
            'Spectral library {0} failed validation'.format(
                base_filename))

    # merge the spectra into a dictionary
    all_spectra = {}
    for column in dataframe:
        all_spectra['{0}:{1}'.format(base_filename, column)] = (
            np.array(dataframe.index),
            dataframe[column].values)

    return all_spectra


def load_all_spectra(path):
    """" Loads all valid spectra from the given location.

    Args:
        path (str): The directory path to scan for supported spectra files.

    Returns:
        dict: A dictionary of 2-tuples of numpy.ndarrays.
            The first element contains the band centre wavelengths of the input
            bands, while the second element contains the spectra values.
            Dictionary is keyed by spectra name built from the file and
            band/sheet names, separated by a colon.

            Note that names are not disambiguated, so that if more than one
            filter has the same name, only the first will be returned and no
            error will be raised (although it will be logged).
    """
    # TODO: add logging
    # logging.getLogger(__name__).info(
    #     'Loading Sensor filters from %s', path)

    all_spectra = {}
    new_spectra = {}

    # excel files
    # for file in list_files(path, ['xls', 'xlsx']):
    #     try:
    #         new_spectra = load_spectra_excel(file)
    #     except UnsupportedDataFormatError:
    #         pass
    #         # except UnsupportedDataFormatError as ex:
    #         # logging.getLogger(__name__).exception(ex)
    #         # TODO: logging
    #     _merge_dictionary(all_spectra, new_spectra)

    # Spectral Libraries
    for file in list_files(path, ['lib']):
        try:
            base_name, _ = os.path.splitext(os.path.basename(file))
            new_spectra = load_envi_spectral_library(path, base_name)
        except UnsupportedDataFormatError:
            pass
            # except UnsupportedDataFormatError as ex:
            # TODO: logging.getLogger(__name__).exception(ex)
        merge_dictionary(all_spectra, new_spectra)

    return all_spectra
