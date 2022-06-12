import os

from astropy import units
from astropy import cosmology
import numpy as np
from scipy import interpolate as itp
from scipy import integrate as itg


cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)
age_universe = cosmo.age(0).value * 1e9
log_age_universe = np.log10(age_universe)


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


def sn_rate(tau, t, dtd, sfr):
    """
    Return the SN rate in the given time.
    Parameters
    ----------
    t : array_like
        The time in yr.
    dtd : callable
        The delay time distribution function (time since the beginning). In
        the unit of number of supernovae per solar mass formed per year.
    sfr : callable
        The star formation rate function (look-back-time). In the unit of
        solar mass formed per year.
    """
    return sfr(t - tau) * dtd(tau)

# 140e-14 SNe / Msun / yr at 0.21 Gyr
gap = 50e6
beta = -1.1

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * 1.

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

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
        os.path.join("output", "spectrum", "galaxy_{}.npy".format(i))
    )
    wave = galaxy_data_cube[0]
    input_sfh_cube = np.load(
        os.path.join("output", "sfh", "galaxy_sfh_{}.npy".format(i))
    )
    # this is a lookback time
    input_age = input_sfh_cube[0]
    input_age = np.append(input_age, [10.52, 10.54, 10.56, 10.58, 10.60])

    for j, spexels in enumerate(n_spexels):

        if j == 0:
            spexels_skip = 0
        else:
            spexels_skip = np.sum(n_spexels[: j - 1])

        input_sfh = input_sfh_cube[j]
        input_sfh = np.append(input_sfh, [min(input_sfh)]*5)

        sfr_itp = itp.interp1d(
            10.**input_age, input_sfh, kind=3, fill_value="extrapolate"
        )
        # the integral limits are the lookback time
        rate = itg.quad(
            sn_rate,
            0.0,
            age_universe,
            args=(age_universe, dtd_itp, sfr_itp),
            epsabs=1.49e-16,
            epsrel=1.49e-16,
            limit=1000,
            maxp1=1000,
            limlst=1000,
        )
        lamb = rate[0] * detection_window
        # Get the Possion probability of NOT observing an SN in that spexels within the detection window
        prob = np.exp(-lamb)

        for spx in range(spexels):

            spexel_idx = 1 + spexels_skip + spx

            if prob < np.random.random():
                sn_list_galaxy.append(i)
                sn_list_spexel.append(spexel_idx)
                N_sn += 1
                print('BOOM! {}-th SN!'.format(N_sn))

np.save(
    os.path.join("output", "sn_list"),
    np.concatenate((sn_list_galaxy, sn_list_spexel)),
)



'''
from matplotlib.pyplot import *
ion()

figure(1)
plot(input_age, sfr_itp(10.0**input_age))


figure(2)
plot(input_age, dtd_itp(10.0**input_age))

'''