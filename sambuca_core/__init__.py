# -*- coding: utf-8 -*-
""" Core components of the Sambuca modeling system """

from .exceptions import (
    SambucaException,
    UnsupportedDataFormatError,
    DataValidationError
)
from .forward_model import forward_model, ForwardModelResults
from .sensor_filter import (
    apply_sensor_filter,
    load_sensor_filters,
    load_sensor_filters_excel,
    load_sensor_filter_spectral_library,
)

__author__ = 'Daniel Collins'
__email__ = 'daniel.collins@csiro.au'

# Versioning: major.minor.patch
# major: increment on a major version. Must be changed when the API changes in
# an imcompatible way.
# minor: new functionality that does not break the
# existing API.
# patch: bug-fixes that do not change the public API
__version__ = '0.1.0'
