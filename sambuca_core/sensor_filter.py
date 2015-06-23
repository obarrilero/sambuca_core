""" Contains functions for working with Sensor Filters.
"""


from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)
from builtins import *

import numpy as np


# TODO: check for normalisation when loading the response_function
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
