import glob
import os

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import ppxf as ppxf_package
from ppxf.capfit import chi2
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from spectres import spectres
from scipy.optimize import minimize


def find_ref(reg_guess, templates, galaxy, noise, velscale, start, plot,
             moments, degree, mdegree, clean, vsyst, lam, reg_dim, component,
             chi2_desired):

    pp = ppxf(templates=templates,
              galaxy=galaxy,
              noise=noise,
              velscale=velscale,
              start=start,
              plot=plot,
              moments=moments,
              degree=degree,
              mdegree=mdegree,
              clean=clean,
              vsyst=vsyst,
              lam=lam,
              regul=reg_guess,
              reg_dim=reg_dim,
              component=component)
    chi2_current = (pp.chi2 - 1) * galaxy.size

    print('Current regularisation value: %#.8g' % reg_guess)
    print('Desired Delta Chi^2: %#.8g' % chi2_desired)
    print('Current Delta Chi^2: %#.8g' % chi2_current)

    return np.exp(abs(chi2_current - chi2_desired))


ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
miles_pathname = ppxf_dir + '/fsps_miles_models/Mun1.30*.fits'

miles_filelist = glob.glob(miles_pathname)

# speed of light
c = 299792.458
# redshift is zero
z = 0.0
# MILES has an approximate FWHM resolution of 2.51A.
FWHM_gal = 2.51

header = fits.open(miles_filelist[0])[0].header
wave_start = header['CRVAL1']
wave_bin = header['CDELT1']
wave_n = header['NAXIS1']
wave = np.linspace(wave_start, wave_start + (wave_n - 1) * wave_bin, wave_n)

velscale = c * np.median(np.diff(wave)) / wave[-1]
miles = lib.miles(miles_pathname, velscale, FWHM_gal)

reg_dim = miles.templates.shape[1:]
templates = miles.templates.reshape(miles.templates.shape[0], -1)

# eq.(8) of Cappellari (2017)
vel = c * np.log(1 + z)
# (km/s), starting guess for [V, sigma]
start = [vel, 10., 0., 0.]

n_temps = templates.shape[1]

# Assign component=0 to the stellar templates, component=1 to the Balmer
# gas emission lines templates and component=2 to the forbidden lines.
component = [0] * n_temps

# Fit (V, sig, h3, h4) moments=4 for the stars
# and (V, sig) moments=2 for the two gas kinematic components
moments = 4

if not os.path.exists('../synthetic_spectra/sfr'):
    os.mkdir('../synthetic_spectra/sfr')

if not os.path.exists('../synthetic_spectra/fitted_model'):
    os.mkdir('../synthetic_spectra/fitted_model')

if not os.path.exists('../synthetic_spectra/age_metallicity'):
    os.mkdir('../synthetic_spectra/age_metallicity')

degree = 10
mdegree = 1

for miles_spectrum in miles_filelist:

    print(miles_spectrum)
    miles_metallicity = miles_spectrum.split('\\')[-1].split('Z')[1].split('T')[0]
    miles_age = miles_spectrum.split('\\')[-1].split('T')[1].split('_')[0]

    galaxy = fits.open(miles_spectrum)[0].data

    galaxy_rebinned, wave_rebinned, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy_rebinned)
    galaxy_rebinned /= norm_factor
    noise_rebinned = np.ones_like(galaxy_rebinned) * 0.05 * (np.random.random(len(galaxy_rebinned)) - 0.5)
    galaxy_rebinned = galaxy_rebinned + noise_rebinned

    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave_rebinned[0])
    # See the pPXF documentation for the keyword REGUL

    chi2_desired = np.sqrt(2 * galaxy_rebinned.size)
    pp = ppxf(templates,
              galaxy_rebinned,
              np.abs(noise_rebinned),
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=degree,
              mdegree=mdegree,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave_rebinned),
              regul=0,
              reg_dim=reg_dim,
              component=component)
    noise_scaled = np.abs(noise_rebinned) * np.sqrt(pp.chi2)

    results = minimize(find_ref,
                       100.,
                       args=(templates, galaxy_rebinned, noise_scaled, velscale, start,
                             False, moments, degree, mdegree, False, dv,
                             np.exp(wave_rebinned), reg_dim, component, chi2_desired),
                       method='Powell',
                       options={'ftol': 1e-1})
    best_reg = results.x

    pp = ppxf(templates,
              galaxy_rebinned,
              noise_scaled,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=degree,
              mdegree=mdegree,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave_rebinned),
              regul=best_reg,
              reg_dim=reg_dim,
              component=component)

    chi2_current = (pp.chi2 - 1) * galaxy_rebinned.size

    print('Desired Delta Chi^2: %#.6g' % chi2_desired)
    print('Current Delta Chi^2: %#.6g' % chi2_current)

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    age_input = [min(age), float(miles_age), float(miles_age), float(miles_age), max(age)]
    sfr_input = [0, 0, max(sfr), 0, 0]

    plt.figure(1, figsize=(10, 6))
    plt.clf()
    plt.plot(age_input, sfr_input, label='Input', color='black')
    plt.plot(age, sfr, label='Recovered')
    plt.plot()
    plt.xscale('log')
    plt.xlabel('Lookback Time (Gyr)')
    plt.ylabel('Relative Star Formation Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/miles_z{}_t{}.png'.format(
        miles_metallicity, miles_age))

    # Plot fit results for stars and gas.
    plt.figure(2, figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/fitted_model/'
                'miles_z{}_t{}_fitted_model.png'.format(miles_metallicity, miles_age))

    # Plot stellar population mass-fraction distribution
    plt.figure(3, figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/age_metallicity/'
                'miles_z{}_t{}_age_metallicity.png'.format(miles_metallicity, miles_age))
