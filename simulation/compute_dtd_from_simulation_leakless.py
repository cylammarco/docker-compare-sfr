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
    dtd[~mask] *= 1e-100
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
    _dtd_guess = 10.0**dtd_guess
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    lamb = np.nansum(_dtd_guess * mass_grid) * t_obs * epsilon
    # Eq. 6, currently assuming 0 to 6 SN(e) per voronoi cell
    lamb_with_sn = np.nansum(
        np.log(
            (np.nansum(_dtd_guess * mass_grid_with_sn, axis=1) * t_obs * epsilon)
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
    nudge_factor = 100.0

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * nudge_factor

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

# Return the DTD as a function of time (year)
dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

# Get the SFH from the firefly data
data_firefly = fits.open("../firefly/manga-firefly-v3_1_1-miles.fits.gz")
firefly_mask = np.where(
    data_firefly["GALAXY_INFO"].data["PLATEIFU"] == "9881-9102"
)[0][0]

"""
# speed of light
c = constants.c.to("km/s").value
# Data are at z=0.025
firefly_z = data_firefly[1].data["REDSHIFT"]

z = firefly_z[firefly_mask]
"""

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

if os.path.exists("sfh_voronoi.npy"):
    sfh_voronoi = np.load("sfh_voronoi.npy")

else:
    #firefly_voronoi_id_list = data_firefly["SPATIAL_BINID"].data
    firefly_sfh_list = data_firefly["STAR_FORMATION_HISTORY"].data
    firefly_stellar_mass_list = data_firefly["STELLAR_MASS_VORONOI"].data
    firefly_remnant_mass_list = data_firefly["STELLAR_MASS_REMNANT"].data
    firefly_spatial_info_list = data_firefly["SPATIAL_INFO"].data
    n_firefly = len(firefly_spatial_info_list)
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
dtd_bin = np.ones_like(input_age) * -15

sfh_voronoi_with_sn = np.array(sfh_voronoi)[sn_mask]

answer = minimize(
    likelihood_voronoi,
    dtd_bin,
    args=(
        np.sum(sfh_voronoi, axis=0),
        sfh_voronoi_with_sn,
        sn_list[sn_mask],
        special.factorial(sn_list[sn_mask]),
    ),
    method="Powell",
    tol=1e-100,
    options={"maxiter": 100000, "xtol": 1e-100, "ftol": 1e-100},
)
print(answer)
np.save(os.path.join("output", f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}"), answer)


answer_no_sn = minimize(
    likelihood_voronoi,
    dtd_bin,
    args=(
        np.sum(sfh_voronoi, axis=0),
        np.ones_like(sfh_voronoi_with_sn) * 1e-10,
        np.zeros_like(sn_list[sn_mask]),
        np.ones_like(sn_list[sn_mask]),
    ),
    method="Powell",
    tol=1e-100,
    options={"maxiter": 100000, "xtol": 1e-100, "ftol": 1e-100},
)
print(answer_no_sn)
np.save(os.path.join("output", f"recovered_dtd_rate_firefly_multiplier_{int(nudge_factor)}_all_SN_removed"), answer_no_sn)

sum_m_psi = np.sum(sfh_voronoi * 10.**answer.x, axis=0)

dtd_err = (np.sum(sn_list) * sum_m_psi / np.sum(sum_m_psi))**-0.5 * 10.**answer.x


# compute the curvature matrix (alpha) here
alpha = np.zeros((len(input_age), len(input_age)))
for i, sfh_i in enumerate(sfh_voronoi_with_sn):
    n_i = sn_list[sn_mask][i]
    lamb_i = np.sum(sfh_i * 10.0**answer.x) * t_obs * epsilon
    for j, m_j in enumerate(sfh_i):
        for k, m_k in enumerate(sfh_i):
            alpha[j][k] += (n_i / lamb_i - 1)**2. * m_j * m_k


alpha *= (t_obs * epsilon)**2
# compute the covariance matrix here
covariance = np.linalg.pinv(alpha)

plt.ion()
plt.figure(1, figsize=(8, 8))
plt.clf()

plt.errorbar(
    input_age,
    10.0**answer.x,
    yerr=dtd_err,
    label="leakless",
    fmt='.',
)

_y = 10.0**answer_no_sn.x 
_y[_y < 1e-29] = 1e-29
plt.scatter(
    input_age,
    _y,
    label="All SN removed",
    marker='v',
    color='C2'
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
plt.ylim(0.9e-30, 1.1e-8)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Delay Time (Gyr)")
plt.ylabel(r"SN / yr / M$_\odot$")
plt.legend(loc="upper right")
plt.title("Non-parametric fit")
plt.tight_layout()
plt.savefig(f"best_fit_dtd_rate_firefly_multiplier_{int(nudge_factor)}.png")

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
for s in sfh_voronoi[sn_list.astype('bool')]:
    plt.scatter(input_age, s, s=2, color='grey')

plt.plot(input_age, np.sum(sfh_voronoi[sn_list.astype('bool')], axis=0), label='SN host')
plt.plot(input_age, np.sum(sfh_voronoi, axis=0), label='total')
plt.xlabel('Time (yr)')
plt.ylabel('Mass formed at the given age')
plt.xscale("log")
plt.yscale("log")
plt.title('Stars Formed at the given age (where SNe were found)in that voronoi cell')
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(f"sfh_with_sn_{int(nudge_factor)}.png")



decades_val = np.array([1e7, 1e8, 1e9, 1e10])
decades_arg = np.where(np.in1d(input_age, decades_val))[0]


plt.figure(4, figsize=(9.5, 8))
plt.clf()
plt.imshow(np.log10(covariance), aspect='auto', origin='lower')
plt.xticks(decades_arg, ['1e7', '1e8', '1e9', '1e10'])
plt.yticks(decades_arg, ['1e7', '1e8', '1e9', '1e10'])
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
plt.imshow(correlation, aspect='auto', origin='lower')
plt.xticks(decades_arg, ['1e7', '1e8', '1e9', '1e10'])
plt.yticks(decades_arg, ['1e7', '1e8', '1e9', '1e10'])
plt.colorbar()
plt.tight_layout()
plt.savefig(f"dtd_correlation_matrix_{int(nudge_factor)}.png")

