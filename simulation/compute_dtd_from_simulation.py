import os

from astropy import units
from matplotlib import pyplot as plt
from numba import jit
import numpy as np
from scipy.optimize import minimize
from scipy import interpolate as itp
from scipy.integrate import quad
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
    dtd = np.zeros_like(t)
    mask = t > gap
    dtd[mask] = (t[mask] * 1e-9) ** gradient
    dtd /= max(dtd)
    dtd *= normalisation
    dtd_itp = itp.interp1d(t, dtd, kind="linear", fill_value="extrapolate")
    return dtd_itp


@jit(nopython=False)
def likelihood_spexel(dtd_guess, mass_grid, n_sn_grid, n_sn_factorial, sn_mask):
    # force the solution to go negative when it multiplied with ZERO
    # Eq. 2, the lamb is for each galaxy
    if (dtd_guess < 0).any():
        return np.inf
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    lamb = np.sum(dtd_guess * mass_grid, axis=2) * t_obs * epsilon
    # Eq. 6, currently assuming 0, 1, 2, 3 or 4 SN(e) per voronoi cell
    ln_like = np.sum(lamb) - np.sum(
        np.log(lamb[sn_mask] ** n_sn_grid / n_sn_factorial)
    )
    if np.isfinite(ln_like):
        return ln_like
    else:
        return np.inf



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
#
input_age_bounds = 10.0 ** (input_age - log_age_bin_size / 2.0)
input_age_bounds = np.append(
    input_age_bounds, 10.0 ** (input_age[-1] + log_age_bin_size / 2.0)
)

# mass in unit of solar mass
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


mass_grid[mass_grid < 0.0] = 0.0

mass_grid_4_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 4, axis=2)]
)
mass_grid_5_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 5, axis=2)]
)
mass_grid_10_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 10, axis=2)]
)
mass_grid_20_bin = np.dstack(
    [np.sum(i, axis=2) for i in np.array_split(mass_grid, 20, axis=2)]
)

input_age_4_bin = np.array(
    [
        np.log10(np.mean(10.0**i))
        for i in np.array_split(input_age, 4, axis=0)
    ]
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

# in unit of years
input_age_bin_size = input_age_bounds[1:] - input_age_bounds[:-1]
input_age_4_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 4, axis=0)]
)
input_age_5_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 5, axis=0)]
)
input_age_10_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 10, axis=0)]
)
input_age_20_bin_size = np.array(
    [np.sum(i) for i in np.array_split(input_age_bin_size, 20, axis=0)]
)

# 140e-14 SNe / Msun / yr at 0.21 Gyr
# delay time in 50 Myr
gap = 50e6
beta = -1.1
nudge_factor = 100.0

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * nudge_factor

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

# Return the DTD as a function of time (year)
dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

n_sn = np.zeros((1000, sum(n_spexels)))

sn_list = np.load(os.path.join("output", "sn_list.npy"))
sn_galaxy_id = sn_list[:, 0]
sn_spexel_id = sn_list[:, 1]

for i, j in zip(sn_galaxy_id, sn_spexel_id):
    n_sn[i][j - 2] += 1

sn_mask = n_sn > 0

# time of observation
t_obs = 10.0
# discovery efficiency
epsilon = 1.0
# a flat input DTD function
dtd_4_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 4)]
)
dtd_5_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 5)]
)
dtd_10_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 10)]
)
dtd_20_bin = np.array(
    [np.mean(i) for i in np.array_split(dtd_itp(10.0**input_age), 20)]
)


dtd_4_bin = np.log10(dtd_4_bin)
dtd_5_bin = np.log10(dtd_5_bin)
dtd_10_bin = np.log10(dtd_10_bin)
dtd_20_bin = np.log10(dtd_20_bin)

dtd_4_bin[~np.isfinite(dtd_4_bin)] = -30
dtd_5_bin[~np.isfinite(dtd_5_bin)] = -30
dtd_10_bin[~np.isfinite(dtd_10_bin)] = -30
dtd_20_bin[~np.isfinite(dtd_20_bin)] = -30

size = len(mass_grid)

answer_20_bin = minimize(
    likelihood_spexel,
    10.0**(dtd_20_bin * 1.1),
    args=(
        mass_grid_20_bin,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
        sn_mask,
    ),
    method="SLSQP",
    options={"maxiter": 100000, "ftol": 1e-30},
)
print(answer_20_bin)
np.save(os.path.join("output", "recovered_dtd_0.2_dex"), answer_20_bin)

answer_10_bin = minimize(
    likelihood_spexel,
    10.0**(dtd_10_bin * 1.1),
    args=(
        mass_grid_10_bin,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
        sn_mask,
    ),
    method="SLSQP",
    options={"maxiter": 100000, "ftol": 1e-30},
)
print(answer_10_bin)
np.save(os.path.join("output", "recovered_dtd_0.4_dex"), answer_10_bin)

answer_5_bin = minimize(
    likelihood_spexel,
    10.0**(dtd_5_bin * 1.1),
    args=(
        mass_grid_5_bin,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
        sn_mask,
    ),
    method="SLSQP",
    options={"maxiter": 100000, "ftol": 1e-30},
)
print(answer_5_bin)
np.save(os.path.join("output", "recovered_dtd_0.8_dex"), answer_5_bin)

answer_4_bin = minimize(
    likelihood_spexel,
    10.0**(dtd_4_bin * 1.1),
    args=(
        mass_grid_4_bin,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
        sn_mask,
    ),
    method="powell",
    options={"maxiter": 100000, "ftol": 1e-30},
)
print(answer_4_bin)
np.save(os.path.join("output", "recovered_dtd_1.0_dex"), answer_4_bin)

plt.ion()
plt.figure(1, figsize=(8, 6))
plt.clf()

plt.scatter(
    10.0**input_age_4_bin,
    answer_4_bin.x,
    label="1.2 dex log(age/yr) binning",
)

plt.scatter(
    10.0**input_age_5_bin,
    answer_5_bin.x,
    label="0.8 dex log(age/yr) binning",
)

plt.scatter(
    10.0**input_age_10_bin,
    answer_10_bin.x,
    label="0.4 dex log(age/yr) binning",
)

plt.scatter(
    10.0**input_age_20_bin,
    answer_20_bin.x,
    label="0.2 dex log(age/yr) binning",
)

plt.scatter(
    10.0**input_age_20_bin,
    10.0**dtd_20_bin,
    color="black",
    marker="+",
    label="Initial Condition",
)
plt.plot(10.0**input_age, dtd_itp(10.0**input_age), label="Input DTD")

plt.grid()
plt.xlim(1e7, 2.5e10)
plt.ylim(1e-15, 1e-11)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Delay Time (Gyr)")
plt.ylabel(r"SN / yr / M$_\odot$")
plt.legend(loc="upper right")
plt.title("Non-parametric fit")
plt.tight_layout()
plt.savefig("best_fit_dtd.png")


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
