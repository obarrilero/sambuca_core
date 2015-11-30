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


def spectra_find_common_wavelengths(one, two):
    """ Finds the common subset of wavelengths for the given pair of spectra
    (wavelengths, values) tuples.
    """
    return np.intersect1d(one[0], two[0])


def spectra_apply_wavelength_mask(spectra, mask):
    """ Applies a wavelength mask to a spectra ((wavelengths, values) tuple).
    All values in spectra that are not in the mask will be removed.

    Args:
        spectra (tuple): the (wavelengths, values) spectra tuple.
        mask (array-like): The mask of wavelength values.

    Returns:
        The masked tuple of (wavelengths, values).
    """

    boolean_mask = (spectra[0] >= mask.min()) & (spectra[0] <= mask.max())
    return (spectra[0][boolean_mask], spectra[1][boolean_mask])
