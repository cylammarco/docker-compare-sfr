import os
from re import L

from astropy import units
from astropy import cosmology
import fsps
from spectres import spectres
import numpy as np
from matplotlib import pyplot as plt

if not os.path.exists("output"):
    os.mkdir("output")


cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)
age_universe = cosmo.age(0).to_value() * 1e9
log_age_universe = np.log10(age_universe)

# log(age) distribution function
log_age = np.linspace(6.5, 10.5, 2001)
log_age_bin_size = log_age[1] - log_age[0]


# Python-FSPS defines the age as the time since the beginning of the Universe
# in Gyr, always supply in a sorted array.
sp = fsps.StellarPopulation(
    compute_vega_mags=False,
    vactoair_flag=True,
    zcontinuous=3,
    sfh=3,
    imf_type=0,
    masscut=100.0,
    add_agb_dust_model=False,
    add_dust_emission=False,
    add_igm_absorption=False,
    add_neb_emission=False,
    add_neb_continuum=False,
    nebemlineinspec=False,
    dust_type=1,
    uvb=0.0,
    dust_tesc=5.51,
    dust1=0.0,
    dust2=0.0,
    dust_index=0.0,
    dust1_index=0.0,
)

sn_10 = 10.0
sn_20 = 20.0
sn_50 = 50.0
tau = 2.0


FWHM_gal = (
    2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
)
wave_new = np.arange(3500.0, 8500.0, FWHM_gal)
age_list = [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 11.0]
# equivalent to [Z] = -0.3, -0.15, 0.0, 0.15, 0.3
# see Figure 4 of https://academic.oup.com/mnras/article/508/4/4844/6385769
# These are the multiplier of solar metallicity.
metallicity_list = [0.5, 0.7, 1.0, 1.4, 2.0]

for metallicity in metallicity_list:
    for age in age_list:
        # log_age is the axis for the SFH
        # age is the time when maximum star formation occurs (in unit of Gyr)
        # _t is the time since the beginning (in unit of Gyr)
        _t = 10.0**log_age_universe / 1.0e9 - 10.0**log_age / 1.0e9
        mask = _t > 0
        _t = _t[mask][::-1]
        _M = np.exp(-np.abs(_t - (10.0**log_age_universe / 1.0e9 - age)) / tau)
        _Z = 0.019 * np.ones_like(_M) * metallicity
        sp.set_tabular_sfh(_t, _M, _Z)
        #
        # get the spectrum
        # the _t and _M have taken into account of the zero SFR, so the tage is the age of the universe
        wave, spectrum = sp.get_spectrum(
            tage=10.0**log_age_universe / 1.0e9, peraa=True
        )
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
        data_out = np.column_stack((wave_new, spectrum_new, 1e-6 * np.ones_like(spectrum_new)))
        data_with_noise_out_10 = np.column_stack(
            (wave_new, spectrum_new_with_noise_10, noise_10)
        )
        data_with_noise_out_20 = np.column_stack(
            (wave_new, spectrum_new_with_noise_20, noise_20)
        )
        data_with_noise_out_50 = np.column_stack(
            (wave_new, spectrum_new_with_noise_50, noise_50)
        )
        np.save(
            os.path.join(
                "output",
                "exponential_rise_decay",
                f"spectrum_{age:.1f}_gyr_{tau:.1f}_rise_decay_{metallicity:.1f}_zsolar",
            ),
            data_out,
        )
        plt.figure(1)
        plt.clf()
        plt.plot(wave_new, spectrum_new)
        plt.xlabel("Wavelength / A")
        plt.ylabel("Flux per A")
        plt.title(f"{age:.1f} Gyr (noiseless)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "output",
                "exponential_rise_decay",
                f"spectrum_{age:.1f}_gyr_{tau:.1f}_rise_decay_{metallicity:.1f}_zsolar.png",
            )
        )
        np.save(
            os.path.join(
                "output",
                "exponential_rise_decay_with_noise",
                f"spectrum_{age:.1f}_gyr_{tau:.1f}_rise_decay_{metallicity:.1f}_zsolar_with_noise_10",
            ),
            data_with_noise_out_10,
        )
        np.save(
            os.path.join(
                "output",
                "exponential_rise_decay_with_noise",
                f"spectrum_{age:.1f}_gyr_{tau:.1f}_rise_decay_{metallicity:.1f}_zsolar_with_noise_20",
            ),
            data_with_noise_out_20,
        )
        np.save(
            os.path.join(
                "output",
                "exponential_rise_decay_with_noise",
                f"spectrum_{age:.1f}_gyr_{tau:.1f}_rise_decay_{metallicity:.1f}_zsolar_with_noise_50",
            ),
            data_with_noise_out_50,
        )
