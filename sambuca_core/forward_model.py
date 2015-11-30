# -*- coding: utf-8 -*-
"""Semi-analytical Lee/Sambuca forward model. """

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals)
from builtins import *
import math
from collections import namedtuple

import numpy as np

from .constants import REFRACTIVE_INDEX_SEAWATER

ForwardModelResults = namedtuple('ForwardModelResults',
                                 [
                                     'r_substratum',
                                     'rrs',
                                     'rrsdp',
                                     'kd',
                                     'kub',
                                     'kuc',
                                     'a',
                                     'bb',
                                 ])
""" namedtuple containing the forward model results.

Attributes:
    r_substratum (numpy.ndarray): The combined substrate.
    rrs (numpy.ndarray): Modelled remotely-sensed reflectance.
    rrsdp (numpy.ndarray): Modelled optically-deep remotely-sensed reflectance.
    kd (numpy.ndarray): TODO
    kub (numpy.ndarray): TODO
    kuc (numpy.ndarray): TODO
    a (numpy.ndarray): Total absorption.
    bb (numpy.ndarray): Total backscatter.
"""


# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

# Disabling invalid-name as many of the common (and published) variable names
# in the Sambuca model are invalid according to Python conventions.
# pylint: disable=invalid-name
def forward_model(
        chl,
        cdom,
        nap,
        depth,
        substrate1,
        wavelengths,
        awater,
        aphy_star,
        num_bands,
        substrate_fraction=1,
        substrate2=None,
        slope_cdom=0.0168052,
        slope_nap=0.00977262,
        slope_backscatter=0.878138,
        lambda0cdom=550.0,
        lambda0nap=550.0,
        lambda0x=546.0,
        x_ph_lambda0x=0.00157747,
        x_nap_lambda0x=0.0225353,
        a_cdom_lambda0cdom=1.0,
        a_nap_lambda0nap=0.00433,
        bb_lambda_ref=550,
        water_refractive_index=REFRACTIVE_INDEX_SEAWATER,
        theta_air=30.0,
        off_nadir=0.0):
    """Semi-analytical Lee/Sambuca forward model.

    TODO: Extended description goes here.

    TODO: For those arguments which have units, the units should be stated.

    Args:
        chl (float): Concentration of chlorophyll (algal organic particles).
        cdom (float): Concentration of coloured dissolved organic particulates.
        nap (float): Concentration of non-algal particles,
            (also known as Tripton/tr in some literature).
        depth (float): Water column depth.
        substrate1 (array-like): A benthic substrate.
        wavelengths (array-like): Central wavelengths of the modelled
            spectral bands.
        awater (array-like): Absorption coefficient of pure water
        aphy_star (array-like): Specific absorption of phytoplankton.
        num_bands (int): The number of spectral bands.
        substrate_fraction (float): Substrate proportion, used to generate a
            convex combination of substrate1 and substrate2.
        substrate2 (array-like, optional): A benthic substrate.
        slope_cdom (float, optional): slope of cdom absorption
        slope_nap (float, optional): slope of NAP absorption
        slope_backscatter (float, optional): TODO
        lambda0cdom (float, optional): TODO
        lambda0nap (float, optional): TODO
        lambda0x (float, optional): TODO
        x_ph_lambda0x (float, optional): specific backscatter of chlorophyl
            at lambda0x.
        x_nap_lambda0x (float, optional): specific backscatter of tripton
            at lambda0x.
        a_cdom_lambda0cdom (float, optional): TODO
        a_nap_lambda0nap (float, optional): TODO
        bb_lambda_ref (float, optional): TODO
        water_refractive_index (float, optional): refractive index of water.
        theta_air (float, optional): solar zenith angle in degrees
        off_nadir (float, optional): off-nadir angle

    Returns:
        ForwardModelResults: A namedtuple containing the model outputs.
    """

    assert len(substrate1) == num_bands
    if substrate2 is not None:
        assert len(substrate2) == num_bands
    assert len(wavelengths) == num_bands
    assert len(awater) == num_bands
    assert len(aphy_star) == num_bands

    # Sub-surface solar zenith angle in radians
    inv_refractive_index = 1.0 / water_refractive_index
    thetaw = math.asin(inv_refractive_index * math.sin(math.radians(theta_air)))

    # Sub-surface viewing angle in radians
    thetao = math.asin(inv_refractive_index * math.sin(math.radians(off_nadir)))

    # Calculate derived SIOPS, based on
    # Mobley, Curtis D., 1994: Radiative Transfer in natural waters.
    bbwater = (0.00194 / 2.0) * np.power(bb_lambda_ref / wavelengths, 4.32)
    acdom_star = a_cdom_lambda0cdom * \
        np.exp(-slope_cdom * (wavelengths - lambda0cdom))
    atr_star = a_nap_lambda0nap * \
        np.exp(-slope_nap * (wavelengths - lambda0nap))

    # Calculate backscatter
    backscatter = np.power(lambda0x / wavelengths, slope_backscatter)
    # backscatter due to phytoplankton
    bbph_star = x_ph_lambda0x * backscatter
    # backscatter due to tripton
    bbtr_star = x_nap_lambda0x * backscatter

    # Total absorption
    a = awater + chl * aphy_star + cdom * acdom_star + nap * atr_star
    # Total backscatter
    bb = bbwater + chl * bbph_star + nap * bbtr_star

    # Calculate total bottom reflectance from the two substrates
    r_substratum = substrate1
    if substrate2 is not None:
        r_substratum = substrate_fraction * substrate1 + \
        (1. - substrate_fraction) * substrate2

    # TODO: what are u and kappa?
    kappa = a + bb
    u = bb / kappa

    # Optical path elongation for scattered photons
    # elongation from water column
    # TODO: reference to the paper from which these equations are derived
    du_column = 1.03 * np.power(1.00 + (2.40 * u), 0.50)
    # elongation from bottom
    du_bottom = 1.04 * np.power(1.00 + (5.40 * u), 0.50)

    # Remotely sensed sub-surface reflectance for optically deep water
    rrsdp = (0.084 + 0.17 * u) * u

    # common terms in the following calculations
    inv_cos_thetaw = 1.0 / math.cos(thetaw)
    inv_cos_theta0 = 1.0 / math.cos(thetao)
    du_column_scaled = du_column * inv_cos_theta0
    du_bottom_scaled = du_bottom * inv_cos_theta0

    # TODO: descriptions of kd, kuc, kub
    kd = kappa * inv_cos_thetaw
    kuc = kappa * du_column_scaled
    kub = kappa * du_bottom_scaled

    # Remotely sensed reflectance
    kappa_d = kappa * depth
    rrs = (rrsdp *
           (1.0 - np.exp(-(inv_cos_thetaw + du_column_scaled) * kappa_d)) +
           ((1.0 / math.pi) * r_substratum *
            np.exp(-(inv_cos_thetaw + du_bottom_scaled) * kappa_d)))

    return ForwardModelResults(
        r_substratum,
        rrs,
        rrsdp,
        kd,
        kub,
        kuc,
        a,
        bb,
    )

# pylint: enable=too-many-arguments
# pylint: enable=invalid-name
# pylint: enable=too-many-locals
