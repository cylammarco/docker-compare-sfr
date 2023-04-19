import os

from astropy import units
from astropy import cosmology
from astropy.io import fits
import fsps
import numpy as np
from matplotlib import pyplot as plt
from spectres import spectres

if not os.path.exists("output"):
    os.mkdir("output")

if not os.path.exists(os.path.join("output", "fsps_generated_template")):
    os.mkdir(os.path.join("output", "fsps_generated_template"))


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
sp.params["logzsol"] = 0.0
sn_10 = 10.0
sn_20 = 20.0
sn_50 = 50.0

FWHM_gal = (
    2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
)
wave_new = np.arange(3000.0, 10000.0, FWHM_gal)
age_list = [
    0.0631,
    0.07943,
    0.1,
    0.125893,
    0.15849,
    0.19953,
    0.2512,
    0.31623,
    0.3981,
    0.5012,
    0.6310,
    0.7943,
    1.0,
    1.25893,
    1.58489,
    1.9953,
    2.5119,
    3.1623,
    3.9811,
    5.0119,
    6.3096,
    7.9433,
    10.0,
    12.5893,
    15.8489,
]

# age = lookback time
for age in age_list:
    wave, spectrum = sp.get_spectrum(tage=age, peraa=True)
    #
    # resample the spectrum to manga resolution
    spectrum_new = spectres(wave_new, wave, spectrum)
    data_out = np.column_stack(
        (wave_new, spectrum_new, np.ones_like(spectrum_new) * 1e-6)
    )
    filename = os.path.join(
            "output",
            "fsps_generated_template",
            f"Eun1.30Zp0.00T{age:2.4f}_iPp0.00_baseFe_linear_FWHM_variable",
        )
    # save noiseless data
    np.save(
        filename,
        data_out,
    )
    hdu = fits.PrimaryHDU(spectrum_new)
    hdu.header['CRVAL1'] = min(wave_new)
    hdu.header['CDELT1'] = FWHM_gal
    hdu.header['CRPIX1'] = 1.
    hdu.writeto(filename + ".fits")
