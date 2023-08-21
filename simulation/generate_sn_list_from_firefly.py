import os
import sys

from astropy import units
from astropy import constants
from astropy import cosmology
from astropy.io import fits
import numpy as np
from scipy import interpolate as itp

from dtd_functions import get_dtd, get_tophat_dtd

cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)
age_universe = cosmo.age(0).to_value() * 1e9
log_age_universe = np.log10(age_universe)
# years of observation
detection_window = 10.0


def sn_rate(tau, dtd, sfr):
    """
    Return the SN rate in the given time.
    Parameters
    ----------
    tau : float
        Lookback time in unit of yr.
    dtd : callable
        The delay time distribution function (look-back-time). In
        the unit of number of supernovae per solar mass formed per year.
    sfr : callable
        The star formation rate function (look-back-time). In the unit of
        solar mass formed per year.
    """
    return sfr(tau) * dtd(tau)


# 140e-14 SNe / Msun / yr at 0.21 Gyr
# no SN in the first 50 Myr
gap = 5e6
beta = -1.0

nudge_factor_list = [1.0, 10.0, 100.0]

# Get the SFH from the firefly data
data_firefly = fits.open("../firefly/manga-firefly-v3_1_1-miles.fits.gz")

# speed of light
c = constants.c.to("km/s").value
# Data are at z=0.025
firefly_z = data_firefly[1].data["REDSHIFT"]

firefly_voronoi_id_list = data_firefly["SPATIAL_BINID"].data
firefly_sfh_list = data_firefly["STAR_FORMATION_HISTORY"].data
firefly_stellar_mass_list = data_firefly["STELLAR_MASS_VORONOI"].data
# firefly_remnant_mass_list = data_firefly["STELLAR_MASS_REMNANT"].data
firefly_spatial_info_list = data_firefly["SPATIAL_INFO"].data
n_firefly = len(firefly_spatial_info_list)

for nudge_factor in nudge_factor_list:
    t1 = gap / 1e9
    t2 = 0.21
    snr_t2 = 140e-14 * nudge_factor

    # find the normalsation at the peak SN production
    snr_t1 = snr_t2 * t1**beta / t2**beta

    dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)
    dtd_tophat_itp_1 = get_tophat_dtd(5e7, 5e8, normalisation=snr_t1)
    dtd_tophat_itp_2 = get_tophat_dtd(1e8, 1e9, normalisation=snr_t1)
    dtd_tophat_itp_3 = get_tophat_dtd(5e8, 1e9, normalisation=snr_t1)

    sn_list = []
    sn_list_tophat_1 = []
    sn_list_tophat_2 = []
    sn_list_tophat_3 = []
    N_sn = 0
    N_sn_tophat_1 = 0
    N_sn_tophat_2 = 0
    N_sn_tophat_3 = 0
    for i, (sfh, spatial, stellar_mass) in enumerate(
        zip(
            firefly_sfh_list,
            firefly_spatial_info_list,
            firefly_stellar_mass_list,
        )
    ):
        print(f"Galaxy {i+1} of {n_firefly}.")
        voronoi_id_list = np.array(list(set(spatial[:, 0]))).astype("int")
        voronoi_id_list = np.sort(voronoi_id_list[voronoi_id_list >= 0])
        # shape = (2800, 8)
        sfh_voronoi = (sfh[:, :, 2].T * 10.0 ** stellar_mass[:, 2]).T
        age_voronoi = 10.0 ** sfh[:, :, 0] * 1e9
        for voronoi_id in voronoi_id_list:
            sfh_voronoi_i = sfh_voronoi[voronoi_id]
            age_voronoi_i = age_voronoi[voronoi_id]
            # Toggle this to remove the youngest stellar populations
            # sfh_voronoi_i[age_voronoi_i < 50e6] = 0.0
            # Eq. 2 & 3
            lamb = dtd_itp(age_voronoi_i) @ sfh_voronoi_i * detection_window
            lamb_tophat_1 = (
                dtd_tophat_itp_1(age_voronoi_i)
                @ sfh_voronoi_i
                * detection_window
            )
            lamb_tophat_2 = (
                dtd_tophat_itp_2(age_voronoi_i)
                @ sfh_voronoi_i
                * detection_window
            )
            lamb_tophat_3 = (
                dtd_tophat_itp_3(age_voronoi_i)
                @ sfh_voronoi_i
                * detection_window
            )
            # Eq. 4
            # Get the Possion probability of NOT observing an SN in that voronois
            # within the detection window
            N = np.random.poisson(lamb)
            N_sn += N
            sn_list.append(N)
            # Tophat case 1
            N_tophat_1 = np.random.poisson(lamb_tophat_1)
            N_sn_tophat_1 += N_tophat_1
            sn_list_tophat_1.append(N_tophat_1)
            # Tophat case 2
            N_tophat_2 = np.random.poisson(lamb_tophat_2)
            N_sn_tophat_2 += N_tophat_2
            sn_list_tophat_2.append(N_tophat_2)
            # Tophat case 3
            N_tophat_3 = np.random.poisson(lamb_tophat_3)
            N_sn_tophat_3 += N_tophat_3
            sn_list_tophat_3.append(N_tophat_3)
        print(f"Power-law DTD: After galaxy {i}, the total no. of SN: {N_sn}.")
        print(
            f"Tophat DTD 1: After galaxy {i}, the total no. of SN:"
            f" {N_sn_tophat_1}."
        )
        print(
            f"Tophat DTD 2: After galaxy {i}, the total no. of SN:"
            f" {N_sn_tophat_2}."
        )
        print(
            f"Tophat DTD 3: After galaxy {i}, the total no. of SN:"
            f" {N_sn_tophat_3}."
        )

    np.save(
        os.path.join("output", f"sn_list_rate_firefly_{int(nudge_factor)}"),
        sn_list,
    )
    np.save(
        os.path.join(
            "output", f"sn_list_rate_firefly_{int(nudge_factor)}_tophat_dtd_1"
        ),
        sn_list_tophat_1,
    )
    np.save(
        os.path.join(
            "output", f"sn_list_rate_firefly_{int(nudge_factor)}_tophat_dtd_2"
        ),
        sn_list_tophat_2,
    )
    np.save(
        os.path.join(
            "output", f"sn_list_rate_firefly_{int(nudge_factor)}_tophat_dtd_3"
        ),
        sn_list_tophat_3,
    )
