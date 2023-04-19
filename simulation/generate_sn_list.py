import os
import sys

from astropy import units
from astropy import cosmology
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate as itp
from scipy import integrate as itg


cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)
age_universe = cosmo.age(0).to_value() * 1e9
log_age_universe = np.log10(age_universe)


def get_dtd(gap, gradient, normalisation=1.0):
    """
    Return an interpolated function of a delay time distribution
    function based on the input delay time and gradient. The returned
    function takes t which is the lockback time in yr.
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
    dtd[dtd<1e-80] = 1e-300
    dtd_itp = itp.interp1d(t, dtd, kind='linear', fill_value="extrapolate")
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
    nudge_factor = 100.0

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * nudge_factor

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)


t = np.arange(6.5, 10.5, 0.1)

input_sfh_cube = np.load(
    os.path.join("output", "sfh", "galaxy_sfh_1.npz")
)['arr_0']
# this is a lookback time
input_age = input_sfh_cube[0]
# append extra "time" to improve stability of interpolation at the edge
input_age = np.append(input_age, [10.52, 10.54, 10.56, 10.58, 10.60])

input_sfh = input_sfh_cube[1]
input_sfh = np.append(input_sfh, [min(input_sfh)] * 5)

sfr_itp = itp.interp1d(
    10.0**input_age, input_sfh, kind=3, fill_value="extrapolate"
)

fig = plt.figure(1, figsize=(8, 8))
fig.clf()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(10.**t, dtd_itp(10.**t))
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(1e-14, 1e-10)
ax1.set_ylabel('Delayed Time Distribution')

ax2.plot(10.**t, sfr_itp(10.**t))
ax2.set_xscale('log')
ax2.set_xlabel('Lookback Time')
ax2.set_ylabel('Relative Star Formation Rate')

plt.savefig('example_dtd_sfr.png')


n_rings = 5
n_spexels = np.arange(n_rings) * 6
n_spexels[0] = 1

# years of observation
detection_window = 10.0

sn_list_galaxy = []
sn_list_spexel = []

N_sn = 0

for i in range(1000):

    print(i)

    galaxy_data_cube = np.load(
        os.path.join("output", "spectrum", "galaxy_{}.npz".format(i))
    )['arr_0']
    wave = galaxy_data_cube[0]
    input_sfh_cube = np.load(
        os.path.join("output", "sfh", "galaxy_sfh_{}.npz".format(i))
    )['arr_0']

    for j, spexels in enumerate(n_spexels):

        if j == 0:
            spexels_skip = 0
        else:
            spexels_skip = np.sum(n_spexels[:j])

        # j=0 is the age
        input_sfh = input_sfh_cube[j+1]
        input_sfh = np.append(input_sfh, [min(input_sfh)] * 5)
        sfr_itp = itp.interp1d(
            10.0**input_age, input_sfh, kind=3, fill_value="extrapolate"
        )

        # the integral limits are the lookback time
        # sfr_itp uses lookback time
        rate = itg.quad(
            sn_rate,
            0.0,
            age_universe,
            args=(dtd_itp, sfr_itp),
            epsabs=1.49e-30,
            epsrel=1.49e-30,
            limit=1000,
            maxp1=1000,
            limlst=1000,
        )
        lamb = rate[0] * detection_window
        # Get the Possion probability of NOT observing an SN in that spexels
        # within the detection window
        # P(n=0) = e^-lambda
        prob_0 = np.exp(-lamb)
        prob_1 = lamb * prob_0
        prob_2 = lamb ** 2.0 * prob_0 / 2.0
        prob_3 = lamb ** 3.0 * prob_0 / 6.0
        prob_4 = lamb ** 4.0 * prob_0 / 24.0
        prob_5 = lamb ** 5.0 * prob_0 / 120.0
        prob_6 = 1 - prob_0 - prob_1 - prob_2 - prob_3 - prob_4 - prob_5

        for spx in range(spexels):

            spexel_idx = spexels_skip + spx
            rnd = np.random.random()

            if prob_0 > rnd:
                pass
            elif prob_0 + prob_1 > rnd:
                sn_list_galaxy.append(i)
                sn_list_spexel.append(spexel_idx)
                N_sn += 1
                print("BOOM! {}-th SN!".format(N_sn))
            elif prob_0 + prob_1 + prob_2 > rnd:
                for _ in range(2):
                    sn_list_galaxy.append(i)
                    sn_list_spexel.append(spexel_idx)
                    N_sn += 1
                    print("BOOM! {}-th SN!".format(N_sn))
            elif prob_0 + prob_1 + prob_2 + prob_3 > rnd:
                for _ in range(3):
                    sn_list_galaxy.append(i)
                    sn_list_spexel.append(spexel_idx)
                    N_sn += 1
                    print("BOOM! {}-th SN!".format(N_sn))
            elif prob_0 + prob_1 + prob_2 + prob_3 + prob_4 > rnd:
                for _ in range(4):
                    sn_list_galaxy.append(i)
                    sn_list_spexel.append(spexel_idx)
                    N_sn += 1
                    print("BOOM! {}-th SN!".format(N_sn))
            elif prob_0 + prob_1 + prob_2 + prob_3 + prob_4 + prob_5 > rnd:
                for _ in range(5):
                    sn_list_galaxy.append(i)
                    sn_list_spexel.append(spexel_idx)
                    N_sn += 1
                    print("BOOM! {}-th SN!".format(N_sn))
            else:
                for _ in range(6):
                    sn_list_galaxy.append(i)
                    sn_list_spexel.append(spexel_idx)
                    N_sn += 1
                    print("BOOM! {}-th SN!".format(N_sn))

np.save(
    os.path.join("output", f"sn_list_rate_multiplier_{int(nudge_factor)}"),
    np.column_stack((sn_list_galaxy, sn_list_spexel)),
)
