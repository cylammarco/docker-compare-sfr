import os
import sys

from astropy import constants
from astropy import units
from astropy.io import fits
from matplotlib import pyplot as plt

import numpy as np
from scipy.optimize import minimize
from scipy import interpolate as itp
from scipy import special

from dtd_functions import get_dtd, get_tophat_dtd


def likelihood_voronoi(
    dtd_guess,
    mass_grid,
    mass_grid_with_sn,
    n_sn_flatten,
):
    # force the solution to go negative when it multiplied with ZERO
    # Eq. 2, the lamb is for each galaxy
    _dtd_guess = 10.0**dtd_guess * t_obs * epsilon
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    lamb = np.sum(_dtd_guess @ mass_grid)
    # Eq. 6 simplified
    lamb_with_sn = _dtd_guess @ mass_grid_with_sn
    x_ln_lamb = n_sn_flatten @ np.log(lamb_with_sn)
    neg_ln_like = lamb - x_ln_lamb
    return neg_ln_like


def likelihood_voronoi_no_sn(
    dtd_guess,
    mass_grid,
):
    # force the solution to go negative when it multiplied with ZERO
    # Eq. 2, the lamb is for each galaxy
    _dtd_guess = 10.0**dtd_guess * t_obs * epsilon
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    lamb = np.sum(_dtd_guess @ mass_grid)
    neg_ln_like = lamb
    return neg_ln_like


def likelihood_zero_inflated_voronoi(
    dtd_guess_and_theta,
    mass_grid_with_sn,
    mass_grid_without_sn,
    n_sn_flatten,
    sample_size,
    n_time_bin,
):
    dtd_guess = dtd_guess_and_theta[:n_time_bin]
    # theta is for each spexel, NOT each time bin
    theta = np.exp(dtd_guess_and_theta[n_time_bin])
    # force the solution to go negative when it multiplied with ZERO
    # Eq. 2, the lamb is for each galaxy
    _dtd_guess = 10.0**dtd_guess * t_obs * epsilon
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    # sum over axis=1 means the array size is the number of SN-host spexel
    lamb_with_sn = _dtd_guess @ mass_grid_with_sn
    lamb_without_sn = _dtd_guess @ mass_grid_without_sn
    # zero inflation y > 0 term
    ln_like_with_sn = np.sum(
        n_sn_flatten * np.log(lamb_with_sn) - lamb_with_sn
    )
    # zero inflation y == 0 term
    ln_like_without_sn = np.sum(np.log(theta + np.exp(-lamb_without_sn)))
    # zero inflation mutual term
    ln_like_mutual_term = np.sum(np.log(1 + theta)) * sample_size
    neg_ln_like = -(ln_like_without_sn + ln_like_with_sn - ln_like_mutual_term)
    print(dtd_guess, theta)
    return neg_ln_like


def likelihood_zero_inflated_voronoi_2(
    dtd_guess_and_theta,
    mass_grid,
    mass_grid_with_sn,
    n_sn_flatten,
    size_without_sn,
    n_time_bin,
):
    dtd_guess = dtd_guess_and_theta[:n_time_bin]
    # theta is for each spexel, NOT each time bin
    theta = dtd_guess_and_theta[n_time_bin]
    if (theta < 0) or (theta > 1):
        return np.inf
    # force the solution to go negative when it multiplied with ZERO
    # Eq. 2, the lamb is for each galaxy
    _dtd_guess = 10.0**dtd_guess * t_obs * epsilon
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    # sum over axis=1 means the array size is the number of SN-host spexel
    lamb = np.sum(_dtd_guess @ mass_grid)
    # Eq. 6 simplified
    lamb_with_sn = _dtd_guess @ mass_grid_with_sn
    x_ln_lamb = n_sn_flatten @ np.log(lamb_with_sn)
    # the zero-inflated components
    ln_theta_with_sn = len(n_sn_flatten) * np.log(theta)
    ln_theta_without_sn = size_without_sn * np.log(1 - theta)
    neg_ln_like = lamb - x_ln_lamb - ln_theta_with_sn - ln_theta_without_sn
    # print(dtd_guess, theta)
    return neg_ln_like


# 140e-14 SNe / Msun / yr at 0.21 Gyr
# delay time in 50 Myr
gap = 5e6
beta = -1.0


# Get the SFH from the firefly data
data_firefly = fits.open("../firefly/manga-firefly-v3_1_1-miles.fits.gz")
"""
firefly_mask = np.where(
    data_firefly["GALAXY_INFO"].data["PLATEIFU"] == "9881-9102"
)[0][0]

# speed of light
c = constants.c.to("km/s").value
# Data are at z=0.025
firefly_z = data_firefly[1].data["REDSHIFT"]

z = firefly_z[firefly_mask]
"""

input_age = np.array(
    [
        6.5000010e06,
        7.0000000e06,
        7.5000005e06,
        8.0000005e06,
        8.5000020e06,
        8.9999990e06,
        9.5000000e06,
        1.0000000e07,
        1.5000002e07,
        2.0000002e07,
        2.5000002e07,
        3.0000000e07,
        3.5000004e07,
        3.9999996e07,
        4.5000000e07,
        4.9999992e07,
        5.4999996e07,
        6.0000004e07,
        6.4999992e07,
        6.9999992e07,
        7.5000000e07,
        8.0000008e07,
        8.5000000e07,
        8.9999992e07,
        9.5000000e07,
        1.0000000e08,
        1.9999998e08,
        2.9999997e08,
        4.0000000e08,
        4.9999997e08,
        5.9999994e08,
        7.0000000e08,
        8.0000000e08,
        9.0000000e08,
        1.0000000e09,
        1.5000000e09,
        2.0000000e09,
        3.0000000e09,
        4.0000005e09,
        5.0000000e09,
        6.0000005e09,
        6.9999995e09,
        8.0000000e09,
        8.9999995e09,
        1.0000000e10,
        1.1000001e10,
        1.1999999e10,
        1.2999999e10,
    ],
    dtype="float32",
)
input_age = np.around(input_age, decimals=-5)
# log_age_bin_size = input_age[1] - input_age[0]
# time_bin_duration = 10.0 ** (input_age + log_age_bin_size / 2.0) - 10.0 ** (
#    input_age - log_age_bin_size / 2.0
# )
#
# input_age_bounds = 10.0 ** (input_age - log_age_bin_size / 2.0)
# input_age_bounds = np.append(
#    input_age_bounds, 10.0 ** (input_age[-1] + log_age_bin_size / 2.0)
# )

if os.path.exists("sfh_voronoi.npy"):
    sfh_voronoi = np.load("sfh_voronoi.npy")

else:
    # firefly_voronoi_id_list = data_firefly["SPATIAL_BINID"].data
    firefly_sfh_list = data_firefly["STAR_FORMATION_HISTORY"].data
    firefly_stellar_mass_list = data_firefly["STELLAR_MASS_VORONOI"].data
    # firefly_remnant_mass_list = data_firefly["STELLAR_MASS_REMNANT"].data
    firefly_spatial_info_list = data_firefly["SPATIAL_INFO"].data
    n_firefly = len(firefly_spatial_info_list)
    sfh_voronoi = []

    for i, (sfh, spatial, stellar_mass) in enumerate(
        zip(
            firefly_sfh_list,
            firefly_spatial_info_list,
            firefly_stellar_mass_list,
            # firefly_remnant_mass_list,
        )
    ):
        print(f"Galaxy {i+1} of {n_firefly}.")
        voronoi_id_list = np.array(list(set(spatial[:, 0]))).astype("int")
        voronoi_id_list = voronoi_id_list[voronoi_id_list >= 0]
        # shape = (2800, 8)
        _sfh_voronoi = (sfh[:, :, 2].T * 10.0 ** stellar_mass[:, 2]).T
        _age_voronoi = np.around(10.0 ** sfh[:, :, 0] * 1e9, decimals=-5)
        for voronoi_id in np.sort(voronoi_id_list):
            sfh_voronoi_i = _sfh_voronoi[voronoi_id]
            age_voronoi_i = _age_voronoi[voronoi_id]
            sfh_voronoi_i_vid = np.zeros_like(input_age)
            # need to loop because multiple SSP can have the same age
            # but different metallicity
            for j, a in enumerate(age_voronoi_i):
                idx = np.where(input_age == a)[0][0]
                sfh_voronoi_i_vid[idx] += sfh_voronoi_i[j]
            sfh_voronoi.append(sfh_voronoi_i_vid)

    np.save("sfh_voronoi.npy", sfh_voronoi)


nudge_factor_list = [1.0, 10.0, 100.0]

for nudge_factor in nudge_factor_list:
    t1 = gap / 1e9
    t2 = 0.21
    snr_t2 = 140e-14 * nudge_factor

    # find the normalsation at the peak SN production
    snr_t1 = snr_t2 * t1**beta / t2**beta

    # Return the DTD as a function of time (year)
    dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)
    dtd_tophat_itp_1 = get_tophat_dtd(5e7, 5e8, normalisation=snr_t1)
    dtd_tophat_itp_2 = get_tophat_dtd(1e8, 1e9, normalisation=snr_t1)
    dtd_tophat_itp_3 = get_tophat_dtd(5e8, 1e9, normalisation=snr_t1)

    sn_list = np.load(
        os.path.join("output", f"sn_list_rate_firefly_{int(nudge_factor)}.npy")
    )
    sn_list_tophat_1 = np.load(
        os.path.join(
            "output",
            f"sn_list_rate_firefly_{int(nudge_factor)}_tophat_dtd_1.npy",
        )
    )
    sn_list_tophat_2 = np.load(
        os.path.join(
            "output",
            f"sn_list_rate_firefly_{int(nudge_factor)}_tophat_dtd_2.npy",
        )
    )
    sn_list_tophat_3 = np.load(
        os.path.join(
            "output",
            f"sn_list_rate_firefly_{int(nudge_factor)}_tophat_dtd_3.npy",
        )
    )

    sn_mask = sn_list > 0
    sn_mask_tophat_1 = sn_list_tophat_1 > 0
    sn_mask_tophat_2 = sn_list_tophat_2 > 0
    sn_mask_tophat_3 = sn_list_tophat_3 > 0

    # time of observation
    t_obs = 10.0
    # discovery efficiency
    epsilon = 1.0
    # a flat input DTD function
    dtd_bin = np.ones_like(input_age) * -10

    sfh_voronoi_with_sn = np.array(sfh_voronoi)[sn_mask]
    sfh_voronoi_with_sn_tophat_1 = np.array(sfh_voronoi)[sn_mask_tophat_1]
    sfh_voronoi_with_sn_tophat_2 = np.array(sfh_voronoi)[sn_mask_tophat_2]
    sfh_voronoi_with_sn_tophat_3 = np.array(sfh_voronoi)[sn_mask_tophat_3]

    sfh_voronoi_without_sn = np.array(sfh_voronoi)[~sn_mask]
    sfh_voronoi_without_sn_tophat_1 = np.array(sfh_voronoi)[~sn_mask_tophat_1]
    sfh_voronoi_without_sn_tophat_2 = np.array(sfh_voronoi)[~sn_mask_tophat_2]
    sfh_voronoi_without_sn_tophat_3 = np.array(sfh_voronoi)[~sn_mask_tophat_3]
    answer = minimize(
        likelihood_voronoi,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi, axis=0),
            sfh_voronoi_with_sn.T,
            sn_list[sn_mask],
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer)
    answer_tophat_1 = minimize(
        likelihood_voronoi,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi, axis=0),
            sfh_voronoi_with_sn_tophat_1.T,
            sn_list_tophat_1[sn_mask_tophat_1],
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_tophat_1)
    answer_tophat_2 = minimize(
        likelihood_voronoi,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi, axis=0),
            sfh_voronoi_with_sn_tophat_2.T,
            sn_list_tophat_2[sn_mask_tophat_2],
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_tophat_2)
    answer_tophat_3 = minimize(
        likelihood_voronoi,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi, axis=0),
            sfh_voronoi_with_sn_tophat_3.T,
            sn_list_tophat_3[sn_mask_tophat_3],
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_tophat_3)
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}",
        ),
        answer,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_1",
        ),
        answer_tophat_1,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_2",
        ),
        answer_tophat_2,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_3",
        ),
        answer_tophat_3,
    )
    """
    dtd_bin_and_theta = np.concatenate((dtd_bin, [1e-2]))
    n_spexel = (np.shape(sfh_voronoi)[0],)
    n_time_bin = len(dtd_bin_and_theta) - 1

    answer_zero_inflated = minimize(
        likelihood_zero_inflated_voronoi_2,
        dtd_bin_and_theta,
        args=(
            sfh_voronoi.T,
            sfh_voronoi_with_sn.T,
            sn_list[sn_mask],
            len(sfh_voronoi_without_sn),
            n_time_bin,
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_zero_inflated)
    answer_zero_inflated_tophat_1 = minimize(
        likelihood_zero_inflated_voronoi_2,
        dtd_bin_and_theta,
        args=(
            sfh_voronoi.T,
            sfh_voronoi_with_sn_tophat_1.T,
            sn_list_tophat_1[sn_mask_tophat_1],
            len(sfh_voronoi_without_sn_tophat_1),
            n_time_bin,
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_zero_inflated_tophat_1)
    answer_zero_inflated_tophat_2 = minimize(
        likelihood_zero_inflated_voronoi_2,
        dtd_bin_and_theta,
        args=(
            sfh_voronoi.T,
            sfh_voronoi_with_sn_tophat_2.T,
            sn_list_tophat_2[sn_mask_tophat_2],
            len(sfh_voronoi_without_sn_tophat_2),
            n_time_bin,
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_zero_inflated_tophat_3)
    answer_zero_inflated_tophat_3 = minimize(
        likelihood_zero_inflated_voronoi_3,
        dtd_bin_and_theta,
        args=(
            sfh_voronoi.T,
            sfh_voronoi_with_sn_tophat_3.T,
            sn_list_tophat_3[sn_mask_tophat_3],
            len(sfh_voronoi_without_sn_tophat_3),
            n_time_bin,
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_zero_inflated_tophat_3)
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_zero_inflated",
        ),
        answer_zero_inflated,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_zero_inflated_tophat_1",
        ),
        answer_zero_inflated_tophat_1,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_zero_inflated_tophat_2",
        ),
        answer_zero_inflated_tophat_2,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_zero_inflated_tophat_3",
        ),
        answer_zero_inflated_tophat_3,
    )

    answer_no_sn = minimize(
        likelihood_voronoi_no_sn,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi, axis=0),
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_no_sn)
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_all_SN_removed",
        ),
        answer_no_sn,
    )
    answer_no_sn_tophat_1 = minimize(
        likelihood_voronoi_no_sn,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi, axis=0),
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_no_sn_tophat_1)
    answer_no_sn_tophat_2 = minimize(
        likelihood_voronoi_no_sn,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi, axis=0),
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_no_sn_tophat_2)
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_all_SN_removed",
        ),
        answer_no_sn,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_all_SN_removed_1",
        ),
        answer_no_sn_tophat_1,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_all_SN_removed_2",
        ),
        answer_no_sn_tophat_2,
    )
    """

    """
    answer_no_sn_zero_inflated = minimize(
        likelihood_zero_inflated_voronoi_2,
        dtd_bin_and_theta,
        args=(
            np.zeros_like(sfh_voronoi_with_sn),
            np.sum(np.zeros_like(sfh_voronoi_without_sn), axis=0),
            np.zeros_like(sn_list[sn_mask])[np.newaxis].T,
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_no_sn_zero_inflated)
    answer_no_sn_zero_inflated_tophat_1 = minimize(
        likelihood_zero_inflated_voronoi_2,
        dtd_bin_and_theta,
        args=(
            np.zeros_like(sfh_voronoi_with_sn_tophat_1),
            np.sum(np.zeros_like(sfh_voronoi_without_sn_tophat_1), axis=0),
            np.zeros_like(sn_list_tophat_1[sn_mask_tophat_1])[np.newaxis].T,
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_no_sn_zero_inflated_tophat_1)
    answer_no_sn_zero_inflated_tophat_2 = minimize(
        likelihood_zero_inflated_voronoi_2,
        dtd_bin_and_theta,
        args=(
            np.zeros_like(sfh_voronoi_with_sn_tophat_2),
            np.sum(np.zeros_like(sfh_voronoi_without_sn_tophat_2), axis=0),
            np.zeros_like(sn_list_tophat_2[sn_mask_tophat_2])[np.newaxis].T,
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_no_sn_zero_inflated_tophat_2)
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_zero_inflated_all_SN_removed",
        ),
        answer_no_sn_zero_inflated,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_zero_inflated_tophat_all_SN_removed_1",
        ),
        answer_no_sn_zero_inflated_tophat_1,
    )
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_zero_inflated_tophat_all_SN_removed_2",
        ),
        answer_no_sn_zero_inflated_tophat_2,
    )
    sfh_voronoi_no_sfh_lt_1e8 = sfh_voronoi.copy()
    sfh_voronoi_with_sn_no_sfh_lt_1e8 = sfh_voronoi_with_sn.copy()

    sfh_voronoi_no_sfh_lt_1e8[:,input_age<=1e8] = 0.0
    sfh_voronoi_with_sn_no_sfh_lt_1e8[:,input_age<=1e8] = 0.0
    answer_no_sfh_lt_1e8 = minimize(
        likelihood_voronoi,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi_no_sfh_lt_1e8, axis=0),
            sfh_voronoi_with_sn_no_sfh_lt_1e8,
            sn_list[sn_mask],
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_no_sfh_lt_1e8)
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_all_young_SFH_removed",
        ),
        answer_no_sfh_lt_1e8,
    )



    sfh_voronoi_1percent_sfh_lt_1e8 = sfh_voronoi.copy()
    sfh_voronoi_with_sn_1percent_sfh_lt_1e8 = sfh_voronoi_with_sn.copy()

    sfh_voronoi_1percent_sfh_lt_1e8[:,input_age<=1e8] *= 0.01
    sfh_voronoi_with_sn_1percent_sfh_lt_1e8[:,input_age<=1e8] *= 0.01
    answer_1percent_sfh_lt_1e8 = minimize(
        likelihood_voronoi,
        dtd_bin,
        args=(
            np.sum(sfh_voronoi_1percent_sfh_lt_1e8, axis=0),
            sfh_voronoi_with_sn_1percent_sfh_lt_1e8,
            sn_list[sn_mask],
        ),
        method="Powell",
        tol=1e-10,
        options={"maxiter": 100000, "xtol": 1e-10, "ftol": 1e-10},
    )
    print(answer_1percent_sfh_lt_1e8)
    np.save(
        os.path.join(
            "output",
            f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_1percent_young_SFH",
        ),
        answer_1percent_sfh_lt_1e8,
    )

    """
    sum_m_psi = np.sum(sfh_voronoi * 10.0**answer.x, axis=0)
    sum_m_psi_tophat_1 = np.sum(
        sfh_voronoi * 10.0**answer_tophat_1.x, axis=0
    )
    sum_m_psi_tophat_2 = np.sum(
        sfh_voronoi * 10.0**answer_tophat_2.x, axis=0
    )
    sum_m_psi_tophat_3 = np.sum(
        sfh_voronoi * 10.0**answer_tophat_3.x, axis=0
    )
    """
    sum_m_psi_zero_inflated = np.sum(
        sfh_voronoi * 10.0 ** answer_zero_inflated.x[: len(dtd_bin)], axis=0
    )
    sum_m_psi_zero_inflated_tophat_1 = np.sum(
        sfh_voronoi * 10.0 ** answer_zero_inflated_tophat_1.x[: len(dtd_bin)],
        axis=0,
    )
    sum_m_psi_zero_inflated_tophat_2 = np.sum(
        sfh_voronoi * 10.0 ** answer_zero_inflated_tophat_2.x[: len(dtd_bin)],
        axis=0,
    )
    sum_m_psi_zero_inflated_tophat_3 = np.sum(
        sfh_voronoi * 10.0 ** answer_zero_inflated_tophat_3.x[: len(dtd_bin)],
        axis=0,
    )
    """
    dtd_err = (
        np.sum(sn_list) * sum_m_psi / np.sum(sum_m_psi)
    ) ** -0.5 * 10.0**answer.x
    dtd_err_tophat_1 = (
        np.sum(sn_list_tophat_1)
        * sum_m_psi_tophat_1
        / np.sum(sum_m_psi_tophat_1)
    ) ** -0.5 * 10.0**answer_tophat_1.x
    dtd_err_tophat_2 = (
        np.sum(sn_list_tophat_2)
        * sum_m_psi_tophat_2
        / np.sum(sum_m_psi_tophat_2)
    ) ** -0.5 * 10.0**answer_tophat_2.x
    dtd_err_tophat_3 = (
        np.sum(sn_list_tophat_3)
        * sum_m_psi_tophat_3
        / np.sum(sum_m_psi_tophat_3)
    ) ** -0.5 * 10.0**answer_tophat_3.x
    """
    dtd_err_zero_inflated = (
        np.sum(sn_list)
        * sum_m_psi_zero_inflated
        / np.sum(sum_m_psi_zero_inflated)
    ) ** -0.5 * 10.0 ** answer_zero_inflated.x[: len(dtd_bin)]
    dtd_err_zero_inflated_tophat_1 = (
        np.sum(sn_list_tophat_1)
        * sum_m_psi_zero_inflated_tophat_1
        / np.sum(sum_m_psi_zero_inflated_tophat_1)
    ) ** -0.5 * 10.0 ** answer_zero_inflated_tophat_1.x[: len(dtd_bin)]
    dtd_err_zero_inflated_tophat_2 = (
        np.sum(sn_list_tophat_2)
        * sum_m_psi_zero_inflated_tophat_2
        / np.sum(sum_m_psi_zero_inflated_tophat_2)
    ) ** -0.5 * 10.0 ** answer_zero_inflated_tophat_2.x[: len(dtd_bin)]
    dtd_err_zero_inflated_tophat_3 = (
        np.sum(sn_list_tophat_3)
        * sum_m_psi_zero_inflated_tophat_3
        / np.sum(sum_m_psi_zero_inflated_tophat_3)
    ) ** -0.5 * 10.0 ** answer_zero_inflated_tophat_3.x[: len(dtd_bin)]
    """
    # compute the curvature matrix (alpha) here
    alpha = np.zeros((len(input_age), len(input_age)))
    for i, sfh_i in enumerate(sfh_voronoi_with_sn):
        n_i = sn_list[sn_mask][i]
        lamb_i = np.sum(sfh_i * 10.0**answer.x) * t_obs * epsilon
        for j, m_j in enumerate(sfh_i):
            for k, m_k in enumerate(sfh_i):
                alpha[j][k] += (n_i / lamb_i - 1) ** 2.0 * m_j * m_k

    alpha *= (t_obs * epsilon) ** 2
    # compute the covariance matrix here
    covariance = np.linalg.pinv(alpha)

    plt.ion()
    plt.figure(1, figsize=(8, 8))
    plt.clf()

    plt.errorbar(
        input_age,
        10.0**answer.x,
        yerr=dtd_err,
        label="poisson",
        fmt=".",
    )
    """
    plt.errorbar(
        input_age,
        10.0 ** answer_zero_inflated.x[: len(dtd_bin)],
        yerr=dtd_err_zero_inflated,
        label="zero inflated poisson",
        fmt=".",
    )
    """
    #
    # _y = 10.0**answer_no_sn.x
    # _y_zero_inflated = 10.0**answer_no_sn_zero_inflated.x[:len(dtd_bin)]
    # _y[_y < 1e-29] = 1e-29
    # _y_zero_inflated[_y_zero_inflated < 1e-29] = 1e-29
    # plt.scatter(input_age, _y, label="All SN removed", marker="v", color="C2")
    # plt.scatter(
    #    input_age,
    #    _y_zero_inflated,
    #    label="All SN removed (zero inflated)",
    #    marker="v",
    #    color="C3",
    # )
    # plt.scatter(input_age, 10.0**answer_no_sfh_lt_1e8.x, label="SFH(t <= 10E8) removed", marker="o", color="C3")
    # plt.scatter(input_age, 10.0**answer_1percent_sfh_lt_1e8.x, label="SFH(t <= 10E8) x 0.01", marker="o", color="C4")

    plt.scatter(
        input_age,
        10.0**dtd_bin,
        color="black",
        marker="+",
        label="Initial Condition",
    )
    plt.plot(input_age, dtd_itp(input_age), label="Input DTD")

    plt.grid()
    plt.xlim(1e6, 3.0e10)
    plt.ylim(0.9e-30, 1.1e-8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Delay Time (Gyr)")
    plt.ylabel(r"SN / yr / M$_\odot$")
    plt.legend(loc="upper right")
    plt.title("Non-parametric fit")
    plt.tight_layout()
    plt.savefig(
        f"best_fit_dtd_rate_firefly_multiplier_{int(nudge_factor)}.png"
    )

    plt.figure(11, figsize=(8, 8))
    plt.clf()

    plt.errorbar(
        input_age,
        10.0**answer_tophat_1.x,
        yerr=dtd_err_tophat_1,
        label="poisson",
        fmt=".",
    )
    """
    plt.errorbar(
        input_age,
        10.0 ** answer_zero_inflated_tophat_1.x[: len(dtd_bin)],
        yerr=dtd_err_zero_inflated_tophat_1,
        label="zero inflated poisson",
        fmt=".",
    )
    """

    # _y_tophat_1 = 10.0**answer_no_sn_tophat_1.x
    # _y_zero_inflated_tophat_1 = 10.0**answer_no_sn_zero_inflated_tophat_1.x[:-1]
    # _y_tophat_1[_y_tophat_1 < 1e-29] = 1e-29
    # _y_zero_inflated_tophat_1[_y_zero_inflated_tophat_1 < 1e-29] = 1e-29
    # plt.scatter(
    #    input_age, _y_tophat_1, label="All SN removed", marker="v", color="C2"
    # )
    # plt.scatter(
    #    input_age,
    #    _y_zero_inflated_tophat_1,
    #    label="All SN removed (zero inflated)",
    #    marker="v",
    #    color="C3",
    # )
    # plt.scatter(input_age, 10.0**answer_no_sfh_lt_1e8.x, label="SFH(t <= 10E8) removed", marker="o", color="C3")
    # plt.scatter(input_age, 10.0**answer_1percent_sfh_lt_1e8.x, label="SFH(t <= 10E8) x 0.01", marker="o", color="C4")

    plt.scatter(
        input_age,
        10.0**dtd_bin,
        color="black",
        marker="+",
        label="Initial Condition",
    )
    plt.plot(input_age, dtd_tophat_itp_1(input_age), label="Input DTD")

    plt.grid()
    plt.xlim(1e6, 3.0e10)
    plt.ylim(0.9e-30, 1.1e-8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Delay Time (Gyr)")
    plt.ylabel(r"SN / yr / M$_\odot$")
    plt.legend(loc="upper right")
    plt.title("Non-parametric fit")
    plt.tight_layout()
    plt.savefig(
        f"best_fit_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_1.png"
    )

    plt.figure(21, figsize=(8, 8))
    plt.clf()

    plt.errorbar(
        input_age,
        10.0**answer_tophat_2.x,
        yerr=dtd_err_tophat_2,
        label="poisson",
        fmt=".",
    )
    """
    plt.errorbar(
        input_age,
        10.0 ** answer_zero_inflated_tophat_2.x[: len(dtd_bin)],
        yerr=dtd_err_zero_inflated_tophat_2,
        label="zero inflated poisson",
        fmt=".",
    )
    """

    # _y_tophat_2 = 10.0**answer_no_sn_tophat_2.x
    # _y_zero_inflated_tophat_2 = 10.0**answer_no_sn_zero_inflated_tophat_2.x[:len(dtd_bin)]
    # _y_tophat_2[_y_tophat_2 < 1e-29] = 1e-29
    # _y_zero_inflated_tophat_2[_y_zero_inflated_tophat_2 < 1e-29] = 1e-29
    # plt.scatter(
    #    input_age, _y_tophat_2, label="All SN removed", marker="v", color="C2"
    # )
    # plt.scatter(
    #    input_age,
    #    _y_zero_inflated_tophat_2,
    #    label="All SN removed (zero inflated)",
    #    marker="v",
    #    color="C3",
    # )
    # plt.scatter(input_age, 10.0**answer_no_sfh_lt_1e8.x, label="SFH(t <= 10E8) removed", marker="o", color="C3")
    # plt.scatter(input_age, 10.0**answer_1percent_sfh_lt_1e8.x, label="SFH(t <= 10E8) x 0.01", marker="o", color="C4")

    plt.scatter(
        input_age,
        10.0**dtd_bin,
        color="black",
        marker="+",
        label="Initial Condition",
    )
    plt.plot(input_age, dtd_tophat_itp_2(input_age), label="Input DTD")

    plt.grid()
    plt.xlim(1e6, 3.0e10)
    plt.ylim(0.9e-30, 1.1e-8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Delay Time (Gyr)")
    plt.ylabel(r"SN / yr / M$_\odot$")
    plt.legend(loc="upper right")
    plt.title("Non-parametric fit")
    plt.tight_layout()
    plt.savefig(
        f"best_fit_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_2.png"
    )

    plt.figure(31, figsize=(8, 8))
    plt.clf()

    plt.errorbar(
        input_age,
        10.0**answer_tophat_3.x,
        yerr=dtd_err_tophat_3,
        label="poisson",
        fmt=".",
    )
    """
    plt.errorbar(
        input_age,
        10.0 ** answer_zero_inflated_tophat_3.x[: len(dtd_bin)],
        yerr=dtd_err_zero_inflated_tophat_3,
        label="zero inflated poisson",
        fmt=".",
    )
    """

    # _y_tophat_3 = 10.0**answer_no_sn_tophat_3.x
    # _y_zero_inflated_tophat_3 = 10.0**answer_no_sn_zero_inflated_tophat_3.x[:len(dtd_bin)]
    # _y_tophat_3[_y_tophat_3 < 1e-29] = 1e-29
    # _y_zero_inflated_tophat_3[_y_zero_inflated_tophat_3 < 1e-29] = 1e-29
    # plt.scatter(
    #    input_age, _y_tophat_3, label="All SN removed", marker="v", color="C2"
    # )
    # plt.scatter(
    #    input_age,
    #    _y_zero_inflated_tophat_3,
    #    label="All SN removed (zero inflated)",
    #    marker="v",
    #    color="C3",
    # )
    # plt.scatter(input_age, 10.0**answer_no_sfh_lt_1e8.x, label="SFH(t <= 10E8) removed", marker="o", color="C3")
    # plt.scatter(input_age, 10.0**answer_1percent_sfh_lt_1e8.x, label="SFH(t <= 10E8) x 0.01", marker="o", color="C4")

    plt.scatter(
        input_age,
        10.0**dtd_bin,
        color="black",
        marker="+",
        label="Initial Condition",
    )
    plt.plot(input_age, dtd_tophat_itp_3(input_age), label="Input DTD")

    plt.grid()
    plt.xlim(1e6, 3.0e10)
    plt.ylim(0.9e-30, 1.1e-8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Delay Time (Gyr)")
    plt.ylabel(r"SN / yr / M$_\odot$")
    plt.legend(loc="upper right")
    plt.title("Non-parametric fit")
    plt.tight_layout()
    plt.savefig(
        f"best_fit_dtd_rate_firefly_multiplier_{int(nudge_factor)}_tophat_3.png"
    )

    """
    plt.figure(2)
    plt.clf()
    plt.plot(input_age, np.sum(sfh_voronoi, axis=0), label='all SFH')
    plt.plot(input_age, np.sum(sfh_voronoi_with_sn, axis=0), label='only SFH with SN')

    plt.xscale("log")
    plt.yscale("log")

    """
    plt.figure(3, figsize=(8, 8))
    plt.clf()
    # for s in sfh_voronoi[sn_list.astype("bool")]:
    #    plt.scatter(input_age, s, s=2, color="grey")

    plt.plot(
        input_age,
        np.sum(sfh_voronoi[sn_list.astype("bool")], axis=0),
        label="SN host",
    )
    plt.plot(input_age, np.sum(sfh_voronoi, axis=0), label="total")
    plt.xlabel("Time (yr)")
    plt.ylabel("Mass formed at the given age")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(
        "Stars Formed at the given age (where SNe were found)in that voronoi"
        " cell"
    )
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"sfh_with_sn_{int(nudge_factor)}.png")

    """
    decades_val = np.array([1e7, 1e8, 1e9, 1e10])
    decades_arg = np.where(np.in1d(input_age, decades_val))[0]


    plt.figure(4, figsize=(9.5, 8))
    plt.clf()
    plt.imshow(np.log10(covariance), aspect="auto", origin="lower")
    plt.xticks(decades_arg, ["1e7", "1e8", "1e9", "1e10"])
    plt.yticks(decades_arg, ["1e7", "1e8", "1e9", "1e10"])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"dtd_covariance_matrix_{int(nudge_factor)}.png")


    sigma = np.sqrt(np.diagonal(covariance))
    correlation = np.zeros_like(covariance)
    for i, c_i in enumerate(sigma):
        for j, c_j in enumerate(sigma):
            correlation[i][j] = covariance[i][j] / c_i / c_j


    plt.figure(5, figsize=(9.5, 8))
    plt.clf()
    plt.imshow(correlation, aspect="auto", origin="lower")
    plt.xticks(decades_arg, ["1e7", "1e8", "1e9", "1e10"])
    plt.yticks(decades_arg, ["1e7", "1e8", "1e9", "1e10"])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"dtd_correlation_matrix_{int(nudge_factor)}.png")
    """
