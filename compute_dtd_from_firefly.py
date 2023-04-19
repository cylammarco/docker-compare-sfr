import glob
import os
import sys

from astropy import coordinates as coords
from astropy import constants
from astropy import units
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import pandas
from scipy.optimize import minimize
from scipy import interpolate as itp
from scipy import special



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
    dtd_itp = itp.interp1d(t, dtd, kind='linear', fill_value="extrapolate")
    return dtd_itp


def likelihood_voronoi(
    dtd_guess,
    mass_grid,
    mass_grid_with_sn,
    n_sn_flatten,
    n_sn_factorial_flatten,
):
    # force the solution to go negative when it multiplied with ZERO
    # Eq. 2, the lamb is for each galaxy
    if (dtd_guess > -6.0).any() or (dtd_guess < -50.0).any():
        return np.inf
    _dtd_guess = 10.0**dtd_guess
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    lamb = np.sum(_dtd_guess * mass_grid) * t_obs * epsilon
    # Eq. 6, currently assuming 0, 1, 2, 3 or 4 SN(e) per voronoi cell
    lamb_with_sn = np.sum(
        np.log(
            (np.sum(_dtd_guess * mass_grid_with_sn, axis=1) * t_obs * epsilon)
            ** n_sn_flatten
            / n_sn_factorial_flatten
        )
    )
    ln_like = lamb - lamb_with_sn
    return ln_like


# 140e-14 SNe / Msun / yr at 0.21 Gyr
# delay time in 50 Myr
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

# Return the DTD as a function of time (year)
dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

# Get the SFH from the firefly data
data_firefly = fits.open("firefly/manga-firefly-v3_1_1-miles.fits.gz")
firefly_mask = np.where(
    data_firefly["GALAXY_INFO"].data["PLATEIFU"] == "9881-9102"
)[0][0]

# speed of light
c = constants.c.to("km/s").value
# Data are at z=0.025
firefly_z = data_firefly[1].data["REDSHIFT"]

z = firefly_z[firefly_mask]
firefly_voronoi_id_list = data_firefly["SPATIAL_BINID"].data
firefly_sfh_list = data_firefly["STAR_FORMATION_HISTORY"].data
firefly_stellar_mass_list = data_firefly["STELLAR_MASS_VORONOI"].data
firefly_remnant_mass_list = data_firefly["STELLAR_MASS_REMNANT"].data
firefly_spatial_info_list = data_firefly["SPATIAL_INFO"].data
n_firefly = len(firefly_spatial_info_list)

input_age = np.array([
    6.5000010e+06, 7.0000000e+06, 7.5000005e+06, 8.0000005e+06,
    8.5000020e+06, 8.9999990e+06, 9.5000000e+06, 1.0000000e+07,
    1.5000002e+07, 2.0000002e+07, 2.5000002e+07, 3.0000000e+07,
    3.5000004e+07, 3.9999996e+07, 4.5000000e+07, 4.9999992e+07,
    5.4999996e+07, 6.0000004e+07, 6.4999992e+07, 6.9999992e+07,
    7.5000000e+07, 8.0000008e+07, 8.5000000e+07, 8.9999992e+07,
    9.5000000e+07, 1.0000000e+08, 1.9999998e+08, 2.9999997e+08,
    4.0000000e+08, 4.9999997e+08, 5.9999994e+08, 7.0000000e+08,
    8.0000000e+08, 9.0000000e+08, 1.0000000e+09, 1.5000000e+09,
    2.0000000e+09, 3.0000000e+09, 4.0000005e+09, 5.0000000e+09,
    6.0000005e+09, 6.9999995e+09, 8.0000000e+09, 8.9999995e+09,
    1.0000000e+10, 1.1000001e+10, 1.1999999e+10, 1.2999999e+10
], dtype='float32')
input_age = np.around(input_age, decimals=-5)
log_age_bin_size = input_age[1] - input_age[0]
time_bin_duration = 10.0 ** (input_age + log_age_bin_size / 2.0) - 10.0 ** (
    input_age - log_age_bin_size / 2.0
)

input_age_bounds = 10.0 ** (input_age - log_age_bin_size / 2.0)
input_age_bounds = np.append(
    input_age_bounds, 10.0 ** (input_age[-1] + log_age_bin_size / 2.0)
)




# Load the SNe from the tables downloaded from TNS
filelist = glob.glob("tns-sn1a-20230416/tns_search*.csv")
_file = []
for filename in filelist:
    _file.append(pandas.read_csv(filename))

sn_table = pandas.concat(_file)

red_shift = sn_table["Redshift"].values
host_red_shift = sn_table["Host Redshift"].values
red_shift_mask = (red_shift <= 0.125) | (host_red_shift <= 0.125)

# sn_list is of size N_galaxy * N_voronoi
sn_list = []

# of size N_galaxy * N_voronoi
galaxy_centre_ra = data_firefly["GALAXY_INFO"].data['OBJRA']
galaxy_centre_dec = data_firefly["GALAXY_INFO"].data['OBJDEC']

# of size number of SN
sn_ra = sn_table["RA"].values
sn_dec = sn_table["DEC"].values

# looping through N_galaxy
for i, (spatial, g_ra, g_dec) in enumerate(zip(firefly_spatial_info_list, galaxy_centre_ra, galaxy_centre_dec)):
    print(f"Galaxy {i+1} of {n_firefly}.")
    galaxy_voronoi_ra_offset = spatial[:, 1]
    galaxy_voronoi_dec_offset = spatial[:, 2]
    voronoi_ra = g_ra + spatial[:, 1]
    voronoi_dec = g_dec + spatial[:, 2]
    # First find the SN with the minimum separation from the galaxy
    g_pos = coords.SkyCoord(g_ra, g_dec, frame="icrs", unit=(units.deg, units.deg))
    for s_ra, s_dec in zip(sn_ra, sn_dec):
        s_pos = coords.SkyCoord(s_ra, s_dec, frame="icrs", unit=(units.hourangle, units.deg))





i = 0

if os.path.exists("sfh_voronoi.npy"):
    sfh_voronoi = np.load("sfh_voronoi.npy")

else:
    sfh_voronoi = []

    for i, (sfh, spatial, stellar_mass, remnant_mass) in enumerate(zip(firefly_sfh_list, firefly_spatial_info_list, firefly_stellar_mass_list, firefly_remnant_mass_list)):
        print(f"Galaxy {i+1} of {n_firefly}.")
        voronoi_id_list = np.array(list(set(spatial[:, 0]))).astype('int')
        voronoi_id_list = voronoi_id_list[voronoi_id_list>=0]
        for voronoi_id in voronoi_id_list:
            arg = np.where(spatial[:, 0].astype('int') == voronoi_id)[0][0]
            mass_voronoi_i = 10.0**stellar_mass[arg][2] + 10.0**remnant_mass[arg][2]
            sfh_voronoi_i = np.array(sfh[arg][:, 2])
            sfh_mask = sfh_voronoi_i > 0.0
            age = np.around(np.array(10.0**sfh[arg][:, 0][sfh_mask] * 1e9), decimals=-5)
            sfh_voronoi_i_vid = np.zeros_like(input_age)
            for j, a in enumerate(age):
                idx = np.where(input_age==a)[0][0]
                sfh_voronoi_i_vid[idx] = sfh_voronoi_i[sfh_mask][j] * mass_voronoi_i
            sfh_voronoi.append(sfh_voronoi_i_vid)
    
    np.save("sfh_voronoi.npy", sfh_voronoi)

# toggle this line to remove the most recent star formation
#sfh_voronoi[:,:10] = 1e-30
























sn_list = np.load(os.path.join("output", f"sn_list_rate_firefly_{int(nudge_factor)}.npy"))
sn_mask = sn_list > 0

# time of observation
t_obs = 10.0
# discovery efficiency
epsilon = 1.0
# a flat input DTD function
dtd_bin = dtd_itp(input_age)

dtd_bin = np.log10(dtd_bin)
dtd_bin[~np.isfinite(dtd_bin)] = -18

sfh_voronoi_with_sn = np.array(sfh_voronoi)[sn_mask]

answer = minimize(
    likelihood_voronoi,
    dtd_bin * 1.01,
    args=(
        np.sum(sfh_voronoi, axis=0),
        sfh_voronoi_with_sn,
        sn_list[sn_mask],
        special.factorial(sn_list[sn_mask]),
    ),
    method="Powell",
    tol=1e-50,
    options={"maxiter": 10000, "xtol": 1e-50},
)
print(answer)
np.save(os.path.join("output", f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}"), answer)


sum_m_psi = np.sum(sfh_voronoi * 10.**answer.x, axis=0)

dtd_err = (np.sum(sn_list) * sum_m_psi / np.sum(sum_m_psi))**-0.5 * 10.**answer.x

plt.ion()
plt.figure(1, figsize=(8, 8))
plt.clf()

plt.errorbar(
    input_age,
    10.0**answer.x,
    yerr=dtd_err,
    label="leakless",
    fmt='o',
)

plt.scatter(
    input_age,
    10.0**(dtd_bin * 1.01),
    color="black",
    marker="+",
    label="Initial Condition",
)
plt.plot(input_age, dtd_itp(input_age), label="Input DTD")

plt.grid()
plt.xlim(1e6, 3.0e10)
plt.ylim(0.9e-20, 1.1e-8)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Delay Time (Gyr)")
plt.ylabel(r"SN / yr / M$_\odot$")
plt.legend(loc="upper right")
plt.title("Non-parametric fit")
plt.tight_layout()
plt.savefig(f"best_fit_dtd_rate_firefly_multiplier_{int(nudge_factor)}.png")


plt.figure(2)
plt.clf()
plt.plot(input_age, np.sum(sfh_voronoi, axis=0), label='all SFH')
plt.plot(input_age, np.sum(sfh_voronoi_with_sn, axis=0), label='only SFH with SN')

plt.xscale("log")
plt.yscale("log")


"""
plt.figure(3)
plt.clf()
for i in range(100):
    plt.plot(
        10.0**input_age_20_bin,
        mass_grid_20_bin[i][0],
        zorder=0,
        color="grey",
        alpha=0.1,
    )


plt.xscale("log")
plt.yscale("log")
"""
