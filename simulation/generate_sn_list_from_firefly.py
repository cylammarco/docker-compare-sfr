import os
import sys

from astropy import units
from astropy import constants
from astropy import cosmology
from astropy.io import fits
import numpy as np
from scipy import interpolate as itp

cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)
age_universe = cosmo.age(0).to_value() * 1e9
log_age_universe = np.log10(age_universe)
# years of observation
detection_window = 10.0


def get_dtd(gap, gradient, normalisation=1.0):
    """
    Return an interpolated function of a delay time distribution
    function based on the input delay time and gradient. The returned
    function takes t which is the lockback time in yr, in the unit of
    SN per year per solar mass
    Parameters
    ----------
    gap : array_like
        The time during which no SN is formed, in yr.
    gradient : array_like
        The power-law gradient of the delay time distribution.
    normalisation : float, optional
        The normalisation (at the gap time) of the delay time distribution.
        The default is 1.0.
    """
    if gradient > 0:
        raise ValueError("Gradient must be negative.")
    t = 10.0 ** np.linspace(1.0, 11.0, 10001)
    dtd = np.ones_like(t)
    mask = t > gap
    dtd[mask] = (t[mask] * 1e-9) ** gradient
    dtd[~mask] *= 1e-30
    dtd /= max(dtd)
    dtd *= normalisation
    dtd_itp = itp.interp1d(t, dtd, kind="linear", fill_value="extrapolate")
    return dtd_itp


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
gap = 50e6
beta = -1.1
try:
    nudge_factor = float(sys.argv[1])
except:
    nudge_factor = 1.0

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * nudge_factor

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

# Get the SFH from the firefly data
data_firefly = fits.open("../firefly/manga-firefly-v3_1_1-miles.fits.gz")

# speed of light
c = constants.c.to("km/s").value
# Data are at z=0.025
firefly_z = data_firefly[1].data["REDSHIFT"]

firefly_voronoi_id_list = data_firefly["SPATIAL_BINID"].data
firefly_sfh_list = data_firefly["STAR_FORMATION_HISTORY"].data
firefly_stellar_mass_list = data_firefly["STELLAR_MASS_VORONOI"].data
firefly_spatial_info_list = data_firefly["SPATIAL_INFO"].data
n_firefly = len(firefly_spatial_info_list)

sn_list = []
N_sn = 0

for i, (sfh, spatial, stellar_mass) in enumerate(
    zip(firefly_sfh_list, firefly_spatial_info_list, firefly_stellar_mass_list)
):
    print(f"Galaxy {i+1} of {n_firefly}.")
    voronoi_id_list = np.array(list(set(spatial[:, 0]))).astype("int")
    voronoi_id_list = voronoi_id_list[voronoi_id_list >= 0]
    for voronoi_id in voronoi_id_list:
        arg = np.where(spatial[:, 0].astype("int") == voronoi_id)[0][0]
        mass_voronoi_i = 10.0 ** stellar_mass[arg][2]
        sfh_voronoi_i = sfh[arg][:, 2] * mass_voronoi_i
        age_voronoi_i = 10.0 ** sfh[arg][:, 0] * 1e9
        # Toggle this to remove the youngest stellar populations
        sfh_voronoi_i[age_voronoi_i < 50e6] = 0.0
        # Eq. 2 & 3
        lamb = (
            np.nansum(dtd_itp(age_voronoi_i) * sfh_voronoi_i)
            * detection_window
        )
        # Eq. 4
        # Get the Possion probability of NOT observing an SN in that voronois
        # within the detection window
        N = np.random.poisson(lamb)
        N_sn += N
        sn_list.append(N)
        if N >0:
            print(f"BOOM! {N} SN! Bringing the total to {N_sn}")

np.save(
    os.path.join("output", f"sn_list_rate_firefly_{int(nudge_factor)}"),
    sn_list,
)
