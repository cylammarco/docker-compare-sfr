import copy
import os

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
              regul=10.**reg_guess,
              reg_dim=reg_dim,
              component=component)
    chi2_current = (pp.chi2 - 1) * galaxy.size

    print('Current regularisation value: %#.8g' % 10.**reg_guess)
    print('Desired Delta Chi^2: %#.8g' % chi2_desired)
    print('Current Delta Chi^2: %#.8g' % chi2_current)

    return np.exp(abs(chi2_current - chi2_desired))


ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
miles_pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

synthetic_spectrum_dir = '../synthetic_spectra'

# speed of light
c = 299792.458
# redshift is zero
z = 0.0
# MILES has an approximate FWHM resolution of 2.51A.
FWHM_gal = 2.51

wave, spec = np.load('../synthetic_spectra/sp_sb00_z0.0_t000063.npy').T
mask = (wave >= 3600.) & (wave <= 7400.)
wave = wave[mask]
spec = spec[mask]
wave_new = np.arange(3620., 7400., 2.)
spec = spectres(wave_new, wave, spec)

velscale = c * np.median(np.diff(wave_new)) / wave_new[-1]
miles = lib.miles(miles_pathname, velscale, FWHM_gal)

reg_dim = miles.templates.shape[1:]
templates = miles.templates.reshape(miles.templates.shape[0], -1)

# eq.(8) of Cappellari (2017)
vel = c * np.log(1 + z)
# (km/s), starting guess for [V, sigma]
start = [vel, 10.]

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
mdegree = 4

for sf_type in ['ed30']:

    for z in [-0.5, -0.25, 0.0, 0.25, 0.5]:

        for t in 10.**np.arange(-2.0, 1.3, 0.1):
            # Star burst
            wave, galaxy = np.load(
                '../synthetic_spectra/sp_{0}_z{1:1.1f}_t{2:06d}.npy'.format(
                    sf_type, z, int(t * 1000))).T
            mask = (wave >= 3600.) & (wave <= 7400.)
            wave = wave[mask]
            galaxy = galaxy[mask]
            galaxy = spectres(wave_new, wave, galaxy)
            galaxy, wave, velscale = util.log_rebin(
                [np.nanmin(wave_new), np.nanmax(wave_new)],
                galaxy,
                velscale=velscale)
            norm_factor = np.nanmedian(galaxy)
            galaxy /= norm_factor
            noise = np.ones_like(galaxy) * 0.05 * (np.random.random() - 0.5)
            galaxy = galaxy + noise

            # eq.(8) of Cappellari (2017)
            dv = c * (miles.log_lam_temp[0] - wave[0])
            # See the pPXF documentation for the keyword REGUL
            regul_err = np.nanmedian(noise)  # Desired regularization error
            chi2_desired = np.sqrt(2 * galaxy.size)
            pp = ppxf(templates,
                      galaxy,
                      np.abs(noise),
                      velscale,
                      start,
                      plot=False,
                      moments=moments,
                      degree=degree,
                      mdegree=mdegree,
                      clean=False,
                      vsyst=dv,
                      lam=np.exp(wave),
                      regul=0,
                      reg_dim=reg_dim,
                      component=component)
            noise_scaled = np.abs(noise) * np.sqrt(pp.chi2)

            results = minimize(find_ref,
                               1.5,
                               args=(templates, galaxy, noise_scaled, velscale,
                                     start, False, moments, degree, mdegree,
                                     False, dv, np.exp(wave), reg_dim,
                                     component, chi2_desired),
                               method='Powell',
                               options={'ftol': 1e-1})
            best_reg = 10.**results.x

            pp = ppxf(templates,
                      galaxy,
                      noise_scaled,
                      velscale,
                      start,
                      plot=False,
                      moments=moments,
                      degree=degree,
                      mdegree=mdegree,
                      clean=False,
                      vsyst=dv,
                      lam=np.exp(wave),
                      regul=best_reg,
                      reg_dim=reg_dim,
                      component=component)

            chi2_current = (pp.chi2 - 1) * galaxy.size

            print('Desired Delta Chi^2: %#.6g' % chi2_desired)
            print('Current Delta Chi^2: %#.6g' % chi2_current)

            weights = pp.weights
            weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
            sfr = np.sum(weights, axis=1)
            age = miles.age_grid[:, 0]

            if (sf_type=='sb00'):
                age_input = [min(age), t, t, t, max(age)]
                sfr_input = [0, 0, max(sfr), 0, 0]
            else:
                age_input = 1.0**np.linspace(np.log10(min(age)), np.log10(max(age)), 1000)
                sfr_input = np.zeros_like(age_input)
                sfr_input[age_input<=t] = np.exp((age_input[age_input<=t]-t)/3.0)

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
            plt.savefig(
                '../synthetic_spectra/sfr/sp_{0}_z{1:1.1f}_t{2:06d}.png'.
                format(sf_type, z, int(t * 1000)))

            # Plot fit results for stars and gas.
            plt.figure(2, figsize=(16, 8))
            plt.clf()
            pp.plot()
            plt.tight_layout()
            plt.savefig('../synthetic_spectra/fitted_model/'
                        'sp_{0}_z{1:1.1f}_t{2:06d}_fitted_model.png'.format(
                            sf_type, z, int(t * 1000)))

            # Plot stellar population mass-fraction distribution
            plt.figure(3, figsize=(16, 8))
            plt.clf()
            miles.plot(weights)
            plt.tight_layout()
            plt.savefig('../synthetic_spectra/age_metallicity/'
                        'sp_{0}_z{1:1.1f}_t{2:06d}_age_metallicity.png'.format(
                            sf_type, z, int(t * 1000)))
