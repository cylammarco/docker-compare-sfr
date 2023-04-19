import os
import sys

from astropy import units
from matplotlib import pyplot as plt

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
    dtd = np.ones_like(t) * -18
    mask = t > gap
    dtd[mask] = (t[mask] * 1e-9) ** gradient
    dtd /= max(dtd)
    dtd *= normalisation
    dtd_itp = itp.interp1d(t, dtd, kind="linear", fill_value="extrapolate")
    return dtd_itp


def likelihood_spexel(
    dtd_guess,
    mass_grid_flattened,
    mass_grid_with_sn_flattened,
    n_sn_flatten,
    n_sn_factorial_flatten,
):
    # force the solution to go negative when it multiplied with ZERO
    # Eq. 2, the lamb is for each galaxy
    if (dtd_guess > -6.0).any() or (dtd_guess < -20.0).any():
        return np.inf
    _dtd_guess = 10.0**dtd_guess
    # In our adaptation, each lamb is for each fibre
    # mass grid has the integrated mass of star formation in that time bin
    lamb = np.sum(_dtd_guess * mass_grid_flattened) * t_obs * epsilon
    # Eq. 6, currently assuming 0, 1, 2, 3 or 4 SN(e) per voronoi cell
    lamb_with_sn = np.sum(
        np.log(
            (np.sum(_dtd_guess * mass_grid_with_sn_flattened, axis=1) * t_obs * epsilon)
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
nudge_factor = float(sys.argv[1])

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * nudge_factor

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

# Return the DTD as a function of time (year)
dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

n_rings = 5
n_spexels = np.arange(n_rings) * 6
n_spexels[0] = 1

ndraw = 1000

i = 0

input_sfh_cube = np.load(
    os.path.join("output", "sfh", "galaxy_sfh_{}.npz".format(i))
)['arr_0']
input_age = input_sfh_cube[0]
# sfh = input_sfh_cube[1]
log_age_bin_size = input_age[1] - input_age[0]
time_bin_duration = 10.0 ** (input_age + log_age_bin_size / 2.0) - 10.0 ** (
    input_age - log_age_bin_size / 2.0
)

mass_grid = np.zeros((ndraw, sum(n_spexels), len(input_age)))


#
#
#
input_age_bounds = 10.0 ** (input_age - log_age_bin_size / 2.0)
input_age_bounds = np.append(
    input_age_bounds, 10.0 ** (input_age[-1] + log_age_bin_size / 2.0)
)

# mass in unit of solar mass
if os.path.exists(os.path.join("output", f"mass_grid.npz")):
    mass_grid = np.load(os.path.join("output", f"mass_grid.npz"))['arr_0']
else:
    for i in range(ndraw):
        print(i)
        input_sfh_cube = np.load(
            os.path.join("output", "sfh", "galaxy_sfh_{}.npz".format(i))
        )['arr_0']
        for j, spexels in enumerate(n_spexels):
            # input_sfh_cube[0] is the age
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
    np.savez_compressed(os.path.join("output", "mass_grid"), mass_grid)


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


def schechter(logm, logphi, logmstar, alpha, m_lower=None):
    """
    Generate a Schechter function (in dlogm).
    """
    phi = (
        (10**logphi)
        * np.log(10)
        * 10 ** ((logm - logmstar) * (alpha + 1))
        * np.exp(-(10 ** (logm - logmstar)))
    )
    return phi


def parameter_at_z0(y, z0, z1=0.2, z2=1.6, z3=3.0):
    """
    Compute parameter at redshift 'z0' as a function
    of the polynomial parameters 'y' and the
    redshift anchor points 'z1', 'z2', and 'z3'.
    """
    y1, y2, y3 = y
    a = ((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / (
        z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)
    )
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a * z0**2 + b * z0 + c


# Continuity model median parameters + 1-sigma uncertainties.
pars = {
    "logphi1": [-2.44, -3.08, -4.14],
    "logphi1_err": [0.02, 0.03, 0.1],
    "logphi2": [-2.89, -3.29, -3.51],
    "logphi2_err": [0.04, 0.03, 0.03],
    "logmstar": [10.79, 10.88, 10.84],
    "logmstar_err": [0.02, 0.02, 0.04],
    "alpha1": [-0.28],
    "alpha1_err": [0.07],
    "alpha2": [-1.48],
    "alpha2_err": [0.1],
}

# Draw samples from posterior assuming independent Gaussian uncertainties.
# Then convert to mass function at 'z = z0'.
draws = {}
z0 = 0.025
for par in ["logphi1", "logphi2", "logmstar", "alpha1", "alpha2"]:
    samp = np.array(
        [
            np.random.normal(median, scale=err, size=ndraw)
            for median, err in zip(pars[par], pars[par + "_err"])
        ]
    )
    if par in ["logphi1", "logphi2", "logmstar"]:
        draws[par] = parameter_at_z0(samp, z0)
    else:
        draws[par] = samp.squeeze()

# Generate Schechter functions.
logm = np.linspace(8, 12, ndraw)  # log(M) grid
phi1 = schechter(
    logm[:, None],
    draws["logphi1"],  # primary component
    draws["logmstar"],
    draws["alpha1"],
)
phi2 = schechter(
    logm[:, None],
    draws["logphi2"],  # secondary component
    draws["logmstar"],
    draws["alpha2"],
)
phi = phi1 + phi2  # combined mass function

# Compute median as the mass function
phi_16, mf, phi_84 = np.percentile(phi, [16, 50, 84], axis=1)

# Get the lower mass limit
M_i_limit = -17
M_i_solar = 4.50
ml_ratio = 3.0
mass_limit = ml_ratio * 10.0 ** (0.4 * (M_i_solar - M_i_limit))
log_mass_limit = np.log10(mass_limit)

mf_normed = mf / mf[np.argmin(np.abs(logm - log_mass_limit))]


# check the galaxy mass distribution
plt.figure(1000)
plt.clf()
plt.hist(
    np.log10(np.sum(np.sum(mass_grid, axis=1), axis=1)),
    bins=50,
    histtype="step",
    density=True,
)
plt.hist(
    np.log10(np.sum(np.sum(mass_grid_20_bin, axis=1), axis=1)),
    bins=50,
    histtype="step",
    density=True,
)
plt.hist(
    np.log10(np.sum(np.sum(mass_grid_10_bin, axis=1), axis=1)),
    bins=50,
    histtype="step",
    density=True,
)
plt.hist(
    np.log10(np.sum(np.sum(mass_grid_5_bin, axis=1), axis=1)),
    bins=50,
    histtype="step",
    density=True,
)
plt.hist(
    np.log10(np.sum(np.sum(mass_grid_4_bin, axis=1), axis=1)),
    bins=50,
    histtype="step",
    density=True,
)
plt.plot(
    logm,
    mf_normed,
    label="Mass Function at z=0.025",
)
plt.yscale("log")

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

n_sn = np.zeros((ndraw, sum(n_spexels)))

sn_list = np.load(os.path.join("output", f"sn_list_rate_multiplier_{int(nudge_factor)}.npy"))
sn_galaxy_id = sn_list[:, 0]
sn_spexel_id = sn_list[:, 1]

for i, j in zip(sn_galaxy_id, sn_spexel_id):
    n_sn[i][j] += 1

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

dtd_4_bin[~np.isfinite(dtd_4_bin)] = -18
dtd_5_bin[~np.isfinite(dtd_5_bin)] = -18
dtd_10_bin[~np.isfinite(dtd_10_bin)] = -18
dtd_20_bin[~np.isfinite(dtd_20_bin)] = -18

size = len(mass_grid)

mass_grid_20_bin_flattened = np.sum(np.sum(mass_grid_20_bin, axis=0), axis=0)
mass_grid_10_bin_flattened = np.sum(np.sum(mass_grid_10_bin, axis=0), axis=0)
mass_grid_5_bin_flattened = np.sum(np.sum(mass_grid_5_bin, axis=0), axis=0)
mass_grid_4_bin_flattened = np.sum(np.sum(mass_grid_4_bin, axis=0), axis=0)

mass_grid_20_bin_with_sn_flattened = mass_grid_20_bin[sn_mask]
mass_grid_10_bin_with_sn_flattened = mass_grid_10_bin[sn_mask]
mass_grid_5_bin_with_sn_flattened = mass_grid_5_bin[sn_mask]
mass_grid_4_bin_with_sn_flattened = mass_grid_4_bin[sn_mask]


answer_20_bin = minimize(
    likelihood_spexel,
    dtd_20_bin * 1.1,
    args=(
        mass_grid_20_bin_flattened,
        mass_grid_20_bin_with_sn_flattened,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
    ),
    method="Powell",
    tol=1e-30,
    options={"maxiter": 10000, "xtol": 1e-30},
)
print(answer_20_bin)
np.save(os.path.join("output", f"recovered_dtd_0.2_dex_rate_multiplier_{int(nudge_factor)}"), answer_20_bin)

answer_10_bin = minimize(
    likelihood_spexel,
    dtd_10_bin * 1.1,
    args=(
        mass_grid_10_bin_flattened,
        mass_grid_10_bin_with_sn_flattened,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
    ),
    method="Powell",
    tol=1e-30,
    options={"maxiter": 10000, "xtol": 1e-30},
)
print(answer_10_bin)
np.save(os.path.join("output", f"recovered_dtd_0.4_dex_rate_multiplier_{int(nudge_factor)}"), answer_10_bin)

answer_5_bin = minimize(
    likelihood_spexel,
    dtd_5_bin * 1.1,
    args=(
        mass_grid_5_bin_flattened,
        mass_grid_5_bin_with_sn_flattened,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
    ),
    method="Powell",
    tol=1e-30,
    options={"maxiter": 10000, 'xtol': 1e-30},
)
print(answer_5_bin)
np.save(os.path.join("output", f"recovered_dtd_0.8_dex_rate_multiplier_{int(nudge_factor)}"), answer_5_bin)

answer_4_bin = minimize(
    likelihood_spexel,
    dtd_4_bin * 1.1,
    args=(
        mass_grid_4_bin_flattened,
        mass_grid_4_bin_with_sn_flattened,
        n_sn[sn_mask],
        special.factorial(n_sn[sn_mask]),
    ),
    method="Powell",
    tol=1e-30,
    options={"maxiter": 10000, 'xtol': 1e-30},
)
print(answer_4_bin)
np.save(os.path.join("output", f"recovered_dtd_1.0_dex_rate_multiplier_{int(nudge_factor)}"), answer_4_bin)

plt.ion()
plt.figure(1, figsize=(8, 8))
plt.clf()

plt.scatter(
    10.0**input_age_4_bin,
    10.0**answer_4_bin.x,
    label="1.2 dex log(age/yr) binning",
)

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

plt.scatter(
    10.0**input_age_20_bin,
    10.0**(dtd_20_bin),
    color="black",
    marker="+",
    label="Initial Condition",
)
plt.plot(10.0**input_age, dtd_itp(10.0**input_age), label="Input DTD")

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
plt.savefig(f"best_fit_dtd_rate_multiplier_{int(nudge_factor)}.png")


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
