import copy
import os

from astropy import cosmology
from astropy import units
import emcee
from matplotlib import pyplot as plt
from numba import jit
import numpy as np
from scipy.optimize import minimize
from scipy import interpolate as itp


def get_dtd(gap, gradient, normalisation=1.0):
    """
    Return an interpolated function of a delay time distribution
    function based on the input delay time and gradient. The returned
    function takes t which is the time in yr.
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
    t = 10.**np.linspace(1.0, 11.0, 10001)
    dtd = np.zeros_like(t)
    mask = t > gap
    dtd[mask] = (t[mask] * 1e-9) ** gradient
    dtd /= max(dtd)
    dtd *= normalisation
    dtd_itp = itp.interp1d(t, dtd, kind=3, fill_value="extrapolate")
    return dtd_itp


@jit(nopython=False)
def likelihood_spexel(dtd, mass_grid, sn_mask):
    # Eq. 2, the lamb is for each galaxy
    lamb = np.sum(10.0**dtd * mass_grid, axis=2)
    # Eq. 6, currently assuming either 0 or 1 SN per voronoi cell
    ln_like = -np.sum(lamb) + np.sum(np.log(lamb[sn_mask]))
    print(-ln_like)
    return -ln_like


cosmo = cosmology.FlatLambdaCDM(
    H0=70. * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)
age_universe = cosmo.age(0)
log_age_universe = np.log10(age_universe.value * 1e9)


n_rings = 5
n_spexels = np.arange(n_rings) * 6
n_spexels[0] = 1


i = 0

input_sfh_cube = np.load(
    os.path.join("output", "sfh", "galaxy_sfh_{}.npy".format(i))
)
input_age = input_sfh_cube[0]
sfh = input_sfh_cube[1]
log_age_bin_size = input_age[1] - input_age[0]
time_bin_duration = 10.0 ** (input_age + log_age_bin_size / 2.0) - 10.0 ** (
    input_age - log_age_bin_size / 2.0
)

mass_grid = np.zeros((1000, sum(n_spexels), len(input_age)))
n_sn = np.zeros((1000, sum(n_spexels)))

sn_list = np.load(os.path.join("output", "sn_list.npy"))
sn_galaxy_id = sn_list[:, 0]
sn_spexel_id = sn_list[:, 1]

for i in range(1000):
    input_sfh_cube = np.load(
        os.path.join("output", "sfh", "galaxy_sfh_{}.npy".format(i))
    )
    for j, spexels in enumerate(n_spexels):
        sfh = input_sfh_cube[j + 1]
        if j == 0:
            spexels_skip = 0
        else:
            spexels_skip = np.sum(n_spexels[:j])
        for spx in range(spexels):
            spexel_idx = spexels_skip + spx
            # Convert to thr unit of: total solar mass formed in this bin
            mass_grid[i][spexel_idx] = sfh * time_bin_duration
            if ((i == sn_galaxy_id) & (spexel_idx == sn_spexel_id)).any():
                n_sn[i][spexel_idx] = 1


mass_grid_10_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 10, axis=2)]
)
mass_grid_20_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 20, axis=2)]
)
mass_grid_50_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 50, axis=2)]
)

input_age_10_bin = np.array(
    [
        np.log10(np.mean(10.0**i))
        for i in np.array_split(input_age, 10, axis=0)
    ]
)
input_age_20_bin = np.array(
    [
        np.log10(np.mean(10.0**i))
        for i in np.array_split(input_age, 20, axis=0)
    ]
)
input_age_50_bin = np.array(
    [
        np.log10(np.mean(10.0**i))
        for i in np.array_split(input_age, 50, axis=0)
    ]
)

# 140e-14 SNe / Msun / yr at 0.21 Gyr
gap = 50e6
beta = -1.1

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * 1.

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

# time of observation
t_obs = 10.0
# discovery efficiency
epsilon = 1.0
# a flat input DTD function
dtd = np.log10(dtd_itp(10.0**input_age))
dtd[~np.isfinite(dtd)] = -1e30
dtd_10_bin = np.log10(dtd_itp(10.0**input_age_10_bin))
dtd_10_bin[~np.isfinite(dtd_10_bin)] = -1e30
dtd_20_bin = np.log10(dtd_itp(10.0**input_age_20_bin))
dtd_20_bin[~np.isfinite(dtd_20_bin)] = -1e30
dtd_50_bin = np.log10(dtd_itp(10.0**input_age_50_bin))
dtd_50_bin[~np.isfinite(dtd_50_bin)] = -1e30
size = len(mass_grid)

mask = (n_sn>0 & np.isfinite(np.sum(mass_grid, axis=2)))
mask_50_bin = (n_sn>0 & np.isfinite(np.sum(mass_grid_50_bin, axis=2)))
mask_20_bin = (n_sn>0 & np.isfinite(np.sum(mass_grid_20_bin, axis=2)))
mask_10_bin = (n_sn>0 & np.isfinite(np.sum(mass_grid_10_bin, axis=2)))

answer = minimize(
    likelihood_spexel,
    dtd,
    args=(mass_grid, mask),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)
print(answer)
np.save(os.path.join("output", "recovered_dtd_0.02_dex"), answer)

answer_50_bin = minimize(
    likelihood_spexel,
    dtd_50_bin,
    args=(mass_grid_50_bin, mask_50_bin),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)
print(answer_50_bin)
np.save(os.path.join("output", "recovered_dtd_0.08_dex"), answer_50_bin)

answer_20_bin = minimize(
    likelihood_spexel,
    dtd_20_bin,
    args=(mass_grid_20_bin, mask_20_bin),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)
print(answer_20_bin)
np.save(os.path.join("output", "recovered_dtd_0.2_dex"), answer_20_bin)

answer_10_bin = minimize(
    likelihood_spexel,
    dtd_10_bin,
    args=(mass_grid_10_bin, mask_10_bin),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)
print(answer_10_bin)
np.save(os.path.join("output", "recovered_dtd_0.4_dex"), answer_10_bin)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.scatter(10.0**input_age, 10.0**answer.x, label='0.02 dex log(age/Gyr) binning')
plt.scatter(10.0**input_age_10_bin, 10.0**answer_10_bin.x, label='0.4 dex log(age/Gyr) binning')
plt.scatter(10.0**input_age_20_bin, 10.0**answer_20_bin.x, label='0.2 dex log(age/Gyr) binning')
plt.scatter(10.0**input_age_50_bin, 10.0**answer_50_bin.x, label='0.08 dex log(age/Gyr) binning')
plt.plot(10.0**input_age, dtd_itp(10.0**input_age), label='Input DTD')
plt.grid()
# plt.xlim(1e-2, 13)
plt.ylim(1e-15, 1e-10)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Delay Time (Gyr)")
plt.ylabel(r"SN / yr / M$_\odot$")
plt.legend(loc="upper right")
plt.title("Non-parametric fit")
plt.tight_layout()
plt.savefig('best_fit_dtd.png')
