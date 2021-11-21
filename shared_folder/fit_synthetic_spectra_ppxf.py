import copy
from matplotlib import pyplot as plt
import numpy as np
import os

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib

ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
miles_pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

synthetic_spectrum_dir = '../synthetic_spectra'

# speed of light
c = 299792.458
# redshift is zero
z = 0.0
# FSPS has an approximate FWHM resolution of 2.0A.
FWHM_gal = 2.0

wave, spec = np.load('../synthetic_spectra/sp_sb05_z00_t1.npy').T
mask = (wave > 3600.) & (wave < 7400.)
wave = wave[mask]
spec = spec[mask]
velscale = c * np.median(np.diff(wave)) / wave[-1]
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

for t in np.arange(1, 15):
    # Star burst
    wave, galaxy = np.load(
        '../synthetic_spectra/sp_sb05_z00_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr, label='Recovered')
    plt.plot()
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_sb05_z00_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_sb05_z00_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_sb05_z00_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_sb05_zp01_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_sb05_zp01_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_sb05_zp01_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_sb05_zp01_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_sb05_zm01_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_sb05_zm01_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_sb05_zm01_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_sb05_zm01_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_sb05_zp02_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_sb05_zp02_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_sb05_zp02_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_sb05_zp02_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_sb05_zm02_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_sb05_zm02_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_sb05_zm02_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_sb05_zm02_t{}_age_metallicity.png'
        .format(t))

    # Exponential
    wave, galaxy = np.load(
        '../synthetic_spectra/sp_ed1_z00_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_ed1_z00_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_ed1_z00_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_ed1_z00_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_ed1_zp01_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_ed1_zp01_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_ed1_zp01_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_ed1_zp01_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_ed1_zm01_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_ed1_zm01_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_ed1_zm01_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_ed1_zm01_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_ed1_zp02_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_ed1_zp02_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_ed1_zp02_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_ed1_zp02_t{}_age_metallicity.png'
        .format(t))

    wave, galaxy = np.load(
        '../synthetic_spectra/sp_ed1_zm02_t{}.npy'.format(t)).T
    mask = (wave > 3600.) & (wave < 7400.)
    wave = wave[mask]
    galaxy = galaxy[mask]
    galaxy, wave, velscale = util.log_rebin(
        [np.nanmin(wave), np.nanmax(wave)], galaxy, velscale=velscale)
    norm_factor = np.nanmedian(galaxy)
    galaxy /= norm_factor
    noise = galaxy * 0.01 + 0.01
    galaxy += noise * np.array(
        [1 if np.random.random() < 0.5 else -1 for i in range(len(galaxy))])
    # eq.(8) of Cappellari (2017)
    dv = c * (miles.log_lam_temp[0] - wave[0])
    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error
    pp = ppxf(templates,
              galaxy,
              noise,
              velscale,
              start,
              plot=False,
              moments=moments,
              degree=4,
              mdegree=4,
              clean=False,
              vsyst=dv,
              lam=np.exp(wave),
              regul=1. / regul_err,
              reg_dim=reg_dim,
              component=component)
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    weights = pp.weights
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('../synthetic_spectra/sfr/sp_ed1_zm02_t{}.png'.format(t))

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/fitted_model/sp_ed1_zm02_t{}_fitted_model.png'.
        format(t))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '../synthetic_spectra/age_metallicity/sp_ed1_zm02_t{}_age_metallicity.png'
        .format(t))
