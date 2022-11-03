import os

from astropy import units
from matplotlib import pyplot as plt
from numba import jit
import numpy as np
from scipy.optimize import minimize
from scipy import interpolate as itp
from scipy.integrate import quad


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
    t = 10.0 ** np.linspace(1.0, 11.0, 10001)
    dtd = np.zeros_like(t)
    mask = t > gap
    dtd[mask] = (t[mask] * 1e-9) ** gradient
    dtd /= max(dtd)
    dtd *= normalisation
    dtd_itp = itp.interp1d(t, dtd, kind="linear", fill_value="extrapolate")
    return dtd_itp


@jit(nopython=False)
def likelihood_spexel(dtd, mass_grid, n_sn, sn_mask):
    # Eq. 2, the lamb is for each galaxy
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    lamb = np.sum(10.0**dtd * mass_grid, axis=2) * t_obs * epsilon
    # Eq. 6, currently assuming 0, 1 or 2 SN(e) per voronoi cell
    ln_like = np.sum(lamb) - np.sum(
        np.log(lamb[sn_mask] ** n_sn[sn_mask] / n_sn[sn_mask])
    )
    if np.isfinite(ln_like):
        return ln_like
    else:
        return np.inf


"""
lamb_1 = np.sum(10.0**dtd_20_bin * mass_grid_20_bin, axis=2)
lamb_2 = np.sum(10.0**answer_20_bin.x * mass_grid_20_bin, axis=2)

ln_like_1 = np.sum(lamb_1) - np.sum(np.log(lamb_1[mask_20_bin]))
ln_like_2 = np.sum(lamb_2) - np.sum(np.log(lamb_2[mask_20_bin]))
"""

n_rings = 5
n_spexels = np.arange(n_rings) * 6
n_spexels[0] = 1


i = 0

input_sfh_cube = np.load(
    os.path.join("output", "sfh", "galaxy_sfh_{}.npy".format(i))
)
input_age = input_sfh_cube[0]
# sfh = input_sfh_cube[1]
log_age_bin_size = input_age[1] - input_age[0]
time_bin_duration = 10.0 ** (input_age + log_age_bin_size / 2.0) - 10.0 ** (
    input_age - log_age_bin_size / 2.0
)

mass_grid = np.zeros((1000, sum(n_spexels), len(input_age)))


#
#
# To fix: the sfh * time_bin_duration should be properly integrated instead of simply mutiplying
#
#
input_age_bounds = 10.0 ** (input_age - log_age_bin_size / 2.0)
input_age_bounds = np.append(
    input_age_bounds, 10.0 ** (input_age[-1] + log_age_bin_size / 2.0)
)

if os.path.exists(os.path.join("output", "mass_grid.npy")):
    mass_grid = np.load(os.path.join("output", "mass_grid.npy"))
else:
    for i in range(1000):
        print(i)
        input_sfh_cube = np.load(
            os.path.join("output", "sfh", "galaxy_sfh_{}.npy".format(i))
        )
        for j, spexels in enumerate(n_spexels):
            sfh = input_sfh_cube[j + 1]
            sfh_itp = itp.interp1d(
                10.0**input_age, sfh, kind="cubic", fill_value="extrapolate"
            )
            if j == 0:
                spexels_skip = 0
            else:
                spexels_skip = np.sum(n_spexels[:j])
            _total_mass = np.zeros(len(input_age))
            for k in range(len(input_age)):
                _total_mass[k] = quad(
                    sfh_itp,
                    input_age_bounds[k],
                    input_age_bounds[k + 1],
                    limit=100000000,
                    epsabs=1e-12,
                    epsrel=1e-12,
                )[0]
            for spx in range(spexels):
                spexel_idx = spexels_skip + spx
                # Convert to thr unit of: total solar mass formed in this bin
                for k in range(len(input_age)):
                    mass_grid[i][spexel_idx][k] = _total_mass[k]
    np.save(os.path.join("output", "mass_grid"), mass_grid)


mass_grid[mass_grid < 1e-30] = 0.0

mass_grid_5_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 5, axis=2)]
)
mass_grid_10_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 10, axis=2)]
)
mass_grid_20_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 20, axis=2)]
)
mass_grid_50_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 50, axis=2)]
)

input_age_5_bin = np.array(
    [
        np.log10(np.mean(10.0**i))
        for i in np.array_split(input_age, 5, axis=0)
    ]
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

# in unit of years
input_age_bin_size = input_age_bounds[1:] - input_age_bounds[:-1]
input_age_5_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 5, axis=0)]
)
input_age_10_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 10, axis=0)]
)
input_age_20_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 20, axis=0)]
)
input_age_50_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 50, axis=0)]
)

# 140e-14 SNe / Msun / yr at 0.21 Gyr
gap = 50e6
beta = -1.1

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * 10.0

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

n_sn = np.zeros((1000, sum(n_spexels)))

sn_list = np.load(os.path.join("output", "sn_list.npy"))
sn_galaxy_id = sn_list[:, 0]
sn_spexel_id = sn_list[:, 1]

for i, j in zip(sn_galaxy_id, sn_spexel_id):
    n_sn[i][j - 2] += 1

mask = n_sn > 0

# time of observation
t_obs = 10.0
# discovery efficiency
epsilon = 1.0
# a flat input DTD function
dtd_5_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 5)]
)

dtd_10_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 10)]
)

dtd_20_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 20)]
)

dtd_50_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 50)]
)


dtd_5_bin = np.log10(dtd_5_bin)
dtd_10_bin = np.log10(dtd_10_bin)
dtd_20_bin = np.log10(dtd_20_bin)
dtd_50_bin = np.log10(dtd_50_bin)


dtd_5_bin[~np.isfinite(dtd_5_bin)] = -30
dtd_10_bin[~np.isfinite(dtd_10_bin)] = -30
dtd_20_bin[~np.isfinite(dtd_20_bin)] = -30
dtd_50_bin[~np.isfinite(dtd_50_bin)] = -30


size = len(mass_grid)

"""
answer_50_bin = minimize(
    likelihood_spexel,
    dtd_50_bin,
    args=(mass_grid_50_bin, n_sn, mask),
    method="Powell",
    options={"maxiter": 100000, "xtol": 1e-30, "ftol": 1e-30},
)
print(answer_50_bin)
np.save(os.path.join("output", "recovered_dtd_0.08_dex"), answer_50_bin)
"""
answer_20_bin = minimize(
    likelihood_spexel,
    dtd_20_bin,
    args=(mass_grid_20_bin, n_sn, mask),
    method="Powell",
    options={"maxiter": 100000, "xtol": 1e-30, "ftol": 1e-30},
)
print(answer_20_bin)
np.save(os.path.join("output", "recovered_dtd_0.2_dex"), answer_20_bin)

answer_10_bin = minimize(
    likelihood_spexel,
    dtd_10_bin,
    args=(mass_grid_10_bin, n_sn, mask),
    method="Powell",
    options={"maxiter": 100000, "xtol": 1e-30, "ftol": 1e-30},
)
print(answer_10_bin)
np.save(os.path.join("output", "recovered_dtd_0.4_dex"), answer_10_bin)

answer_5_bin = minimize(
    likelihood_spexel,
    dtd_5_bin,
    args=(mass_grid_5_bin, n_sn, mask),
    method="Powell",
    options={"maxiter": 100000, "xtol": 1e-30, "ftol": 1e-30},
)
print(answer_5_bin)
np.save(os.path.join("output", "recovered_dtd_0.8_dex"), answer_5_bin)

plt.figure(1, figsize=(8, 6))
plt.clf()

plt.scatter(
    10.0**input_age_5_bin,
    10.0**answer_5_bin.x,
    label="0.8 dex log(age/yr) binning",
)

plt.scatter(
    10.0**input_age_10_bin,
    10.0**answer_10_bin.x,
    label="0.4 dex log(age/yr) binning",
)

plt.scatter(
    10.0**input_age_20_bin,
    10.0**answer_20_bin.x,
    label="0.2 dex log(age/yr) binning",
)
"""
plt.scatter(
    10.0**input_age_50_bin,
    10.0**answer_50_bin.x,
    label="0.08 dex log(age/yr) binning",
)
"""
plt.scatter(
    10.0**input_age_20_bin, 10.0**dtd_20_bin, color="black", marker="+", label='Initial Condition'
)
plt.plot(10.0**input_age, dtd_itp(10.0**input_age), label="Input DTD")

plt.grid()
plt.xlim(1e7, 2.5e10)
plt.ylim(1e-15, 1e-10)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Delay Time (Gyr)")
plt.ylabel(r"SN / yr / M$_\odot$")
plt.legend(loc="upper right")
plt.title("Non-parametric fit")
plt.tight_layout()
plt.savefig("best_fit_dtd.png")

plt.ylim(1e-31, 1e7)
plt.savefig("best_fit_dtd_zoomed_out.png")


"""
plt.figure(2)
plt.clf()
for i in range(100):
    plt.plot(
        10.0**input_age, mass_grid[i][0], zorder=0, color="grey", alpha=0.1
    )


plt.xscale("log")
plt.yscale("log")


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
