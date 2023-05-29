from matplotlib.pyplot import *
import numpy as np
from scipy import fft
from scipy.signal import convolve, deconvolve
from scipy.interpolate import interp1d

ion()


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
    dtd_itp = interp1d(t, dtd, kind="linear", fill_value="extrapolate")
    return dtd_itp


input_age = np.array(
    [
        6.5000010e06,
        7.0000000e06,
        7.5000005e06,
        8.0000005e06,
        8.5000020e06,
        8.9999990e06,
        9.5000000e06,
        1.0000000e07,
        1.5000002e07,
        2.0000002e07,
        2.5000002e07,
        3.0000000e07,
        3.5000004e07,
        3.9999996e07,
        4.5000000e07,
        4.9999992e07,
        5.4999996e07,
        6.0000004e07,
        6.4999992e07,
        6.9999992e07,
        7.5000000e07,
        8.0000008e07,
        8.5000000e07,
        8.9999992e07,
        9.5000000e07,
        1.0000000e08,
        1.9999998e08,
        2.9999997e08,
        4.0000000e08,
        4.9999997e08,
        5.9999994e08,
        7.0000000e08,
        8.0000000e08,
        9.0000000e08,
        1.0000000e09,
        1.5000000e09,
        2.0000000e09,
        3.0000000e09,
        4.0000005e09,
        5.0000000e09,
        6.0000005e09,
        6.9999995e09,
        8.0000000e09,
        8.9999995e09,
        1.0000000e10,
        1.1000001e10,
        1.1999999e10,
        1.2999999e10,
    ],
    dtype="float32",
)
input_age = np.around(input_age, decimals=-5)


# 140e-14 SNe / Msun / yr at 0.21 Gyr
# no SN in the first 50 Myr
gap = 50e6
beta = -1.1
nudge_factor = 1.0

t1 = gap / 1e9
t2 = 0.21
snr_t2 = 140e-14 * nudge_factor

# find the normalsation at the peak SN production
snr_t1 = snr_t2 * t1**beta / t2**beta

dtd_itp = get_dtd(gap, beta, normalisation=snr_t1)

DTD = dtd_itp(input_age)
SFH = input_age.copy()

t_resampled = np.arange(5e6, 1.5e10, 1e6)
DTD_resampled = interp1d(
    input_age, DTD, kind="linear", fill_value="extrapolate"
)(t_resampled)
SFH_resampled = interp1d(
    input_age, SFH, kind="linear", fill_value="extrapolate"
)(t_resampled)
SFH_resampled[SFH_resampled < 1e-50] = 1e-50
DTD_resampled[DTD_resampled < 1e-50] = 1e-50
SNR_resampled = SFH_resampled * DTD_resampled

F_SFH = fft.rfft(SFH_resampled)
F_SNR = fft.rfft(SNR_resampled)

F_DTD = F_SNR / F_SFH

fig, axs = subplots(3, 1, figsize=(8, 12), sharex=True)

axs[0].plot(input_age, SFH, label="input")
axs[0].plot(t_resampled, SFH_resampled, label="resampled")
axs[0].plot(
    t_resampled,
    fft.irfft(F_SFH, len(t_resampled)),
    label="recovered from inverse-FFT",
)
axs[0].set_ylabel("Star Formation Rate")

axs[1].plot(t_resampled, SNR_resampled, label="resampled")
axs[1].plot(
    t_resampled,
    fft.irfft(F_SNR, len(t_resampled)),
    label="recovered from inverse-FFT",
)
axs[1].set_ylabel("Supernova Rate")

axs[2].plot(input_age, DTD, label="input")
axs[2].plot(t_resampled, DTD_resampled, label="resampled")
axs[2].plot(
    t_resampled,
    fft.irfft(F_DTD, len(t_resampled)),
    label="recovered from inverse-FFT",
)
axs[2].set_xlabel("Lookback Time")
axs[2].set_ylabel("Delay-Time Distribution")

axs[0].set_xscale("log")
axs[1].set_xscale("log")
axs[2].set_xscale("log")

axs[1].set_yscale("log")
axs[2].set_yscale("log")

axs[0].legend()
axs[1].legend()
axs[2].legend()

tight_layout()
subplots_adjust(hspace=0)
