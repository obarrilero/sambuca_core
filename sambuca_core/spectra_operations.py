# -*- coding: utf-8 -*-
""" Contains functions for manipulating the (wavelength, value) tuples returned
by the spectra readers.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)
from builtins import *

import numpy as np


def spectra_find_common_wavelengths(wavs_a, wavs_b):
    """ Finds the common subset of wavelengths for the given pair.

    Args:
        wavs_a (array-like): a vector of wavelength values.
        wavs_b (array-like): a vector of wavelength values.

    Returns:
        numpy.ndarray: The common subset of values.
    """
    return np.intersect1d(wavs_a, wavs_b)


def spectra_apply_wavelength_mask(spectra, mask):
    """ Applies a wavelength mask to a spectra ((wavelengths, values) tuple).
    All values in the spectra that are not in the mask will be removed in the
    returned values. The input spectra is not modified.

    Args:
        spectra (tuple): the (wavelengths, values) spectra tuple.
        mask (array-like): The wavelength values that should be retained.

    Returns:
        The masked tuple of (wavelengths, values).
    """

    boolean_mask = (spectra[0] >= mask.min()) & (spectra[0] <= mask.max())
    return spectra[0][boolean_mask], spectra[1][boolean_mask]
