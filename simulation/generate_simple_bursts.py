import os

from astropy import units
from astropy import cosmology
import fsps
import numpy as np
from matplotlib import pyplot as plt
from spectres import spectres

if not os.path.exists("output"):
    os.mkdir("output")

if not os.path.exists(os.path.join("output", "simple_bursts")):
    os.mkdir(os.path.join("output", "simple_bursts"))

if not os.path.exists(os.path.join("output", "simple_bursts_with_noise")):
    os.mkdir(os.path.join("output", "simple_bursts_with_noise"))

cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)


# Python-FSPS defines the age as the time since the beginning of the Universe
# in Gyr, always supply in a sorted array.
sp = fsps.StellarPopulation(
    vactoair_flag=True,
    compute_vega_mags=False,
    zcontinuous=1,
    sfh=0,
    add_agb_dust_model=False,
    add_dust_emission=False,
    add_igm_absorption=False,
    add_neb_emission=False,
    add_neb_continuum=False,
    nebemlineinspec=False,
)
sp.params["logzsol"] = 0.0
sn_10 = 10.0
sn_20 = 20.0
sn_50 = 50.0

FWHM_gal = (
    2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
)
wave_new = np.arange(3500.0, 8500.0, FWHM_gal)
age_list = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 11.0]

# age = lookback time

for age in age_list:
    wave, spectrum = sp.get_spectrum(tage=age, peraa=True)
    #
    # resample the spectrum to manga resolution
    spectrum_new = spectres(wave_new, wave, spectrum)
    #
    # add noise
    noise = np.sqrt(spectrum_new)
    noise_scale_factor_10 = np.mean(spectrum_new / noise) / sn_10
    noise_10 = noise * noise_scale_factor_10
    noise_scale_factor_20 = np.mean(spectrum_new / noise) / sn_20
    noise_20 = noise * noise_scale_factor_20
    noise_scale_factor_50 = np.mean(spectrum_new / noise) / sn_50
    noise_50 = noise * noise_scale_factor_50
    spectrum_new_with_noise_10 = np.random.normal(loc=spectrum_new, scale=noise_10)
    spectrum_new_with_noise_20 = np.random.normal(loc=spectrum_new, scale=noise_20)
    spectrum_new_with_noise_50 = np.random.normal(loc=spectrum_new, scale=noise_50)
    data_out_10 = np.column_stack((wave_new, spectrum_new, noise_10))
    data_out_20 = np.column_stack((wave_new, spectrum_new, noise_20))
    data_out_50 = np.column_stack((wave_new, spectrum_new, noise_50))
    data_with_noise_out_10 = np.column_stack(
        (wave_new, spectrum_new_with_noise_10, noise_10)
    )
    data_with_noise_out_20 = np.column_stack(
        (wave_new, spectrum_new_with_noise_20, noise_20)
    )
    data_with_noise_out_50 = np.column_stack(
        (wave_new, spectrum_new_with_noise_50, noise_50)
    )
    #
    #
    data_out = np.column_stack(
        (wave_new, spectrum_new, np.ones_like(spectrum_new) * 1e-6)
    )
    data_with_noise_out_10 = np.column_stack(
        (wave_new, spectrum_new_with_noise_10, noise_10)
    )
    data_with_noise_out_20 = np.column_stack(
        (wave_new, spectrum_new_with_noise_20, noise_20)
    )
    data_with_noise_out_50 = np.column_stack(
        (wave_new, spectrum_new_with_noise_50, noise_50)
    )
    # save noiseless data
    np.save(
        os.path.join(
            "output",
            "simple_bursts",
            f"spectrum_{age}_gyr"
        ),
        data_out,
    )
    plt.figure(1)
    plt.clf()
    plt.plot(wave_new, spectrum_new)
    plt.xlabel("Wavelength / A")
    plt.ylabel("Flux per A")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "output",
            "simple_bursts",
            f"spectrum_{age}_gyr.png"
        )
    )
    # save noisy data
    np.save(
        os.path.join(
            "output",
            "simple_bursts_with_noise",
            f"spectrum_{age}_gyr_with_noise_10"
        ),
        data_with_noise_out_10,
    )
    np.save(
        os.path.join(
            "output",
            "simple_bursts_with_noise",
            f"spectrum_{age}_gyr_with_noise_20"
        ),
        data_with_noise_out_20,
    )
    np.save(
        os.path.join(
            "output",
            "simple_bursts_with_noise",
            f"spectrum_{age}_gyr_with_noise_50"
        ),
        data_with_noise_out_50,
    )
    plt.figure(2)
    plt.clf()
    plt.plot(wave_new, spectrum_new_with_noise_10)
    plt.xlabel("Wavelength / A")
    plt.ylabel("Flux per A")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "output",
            "simple_bursts_with_noise",
            f"spectrum_{age}_gyr_with_noise_10.png"
        )
    )
