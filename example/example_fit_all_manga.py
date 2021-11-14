from astropy.io import fits
import copy
from matplotlib import pyplot as plt
import numpy as np
import os

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib

from manga_reader import manga_reader

# SDSS redshift estimate
z = 0.02894
c = 299792.458  # speed of light in km/s
#filename = 'manga-7495-12704-LINCUBE.fits.gz'
filename = 'manga-7495-12704-LOGCUBE-VOR10-GAU-MILESHC.fits.gz'
foldername = filename.split('.')[0]

if not os.path.exists('/home/sfr/ppxf/{}'.format(foldername)):
    os.mkdir('/home/sfr/ppxf/{}'.format(foldername))

mr = manga_reader(z)
mr.load_lincube('/home/sfr/example/manga_example_data/{}'.format(filename))

# Save the idx to pix function
np.save('/home/sfr/ppxf/{}/manga_7495_12704_ppxf_idx_to_pix'.
        format(foldername), mr._idx_to_pix)


def manga_flux_to_maggies(flux):
    return flux * 1e-17 * 3.33564095E+04 * wave**2. / 3631.


# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
##############################################################################

tie_balmer = True
limit_doublets = True

ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))

for i, (wave, flux, flux_err) in enumerate(mr.iterate_data()):

    print(i)
    if i < 498:
        continue

    # Only use the wavelength range in common between galaxy and stellar library.
    #
    mask = (wave > 3540.) & (wave < 7400.) & (flux > 0)

    wave = wave[mask]
    galaxy = flux[mask]
    noise = flux_err[mask]

    # Ignore blank spectra
    if ((np.sum(galaxy) == 0) | (np.sum(np.isfinite(noise)) <= 1)):

        print('{} is not an usable spectrum.'.format(i))
        continue

    else:

        print('Going forward with {}.'.format(i))

    # Normalize spectrum to avoid numerical issues
    norm_factor = np.nanmedian(galaxy)

    if norm_factor <= 0:

        print('Less than half of spectrum {} is usable, ignore.'.format(i))
        continue

    noise /= norm_factor
    galaxy /= norm_factor

    noise[~np.isfinite(noise)] = max(noise[np.isfinite(noise)])

    # The SDSS wavelengths are in vacuum, while the MILES ones are in air.
    # For a rigorous treatment, the SDSS vacuum wavelengths should be
    # converted into air wavelengths and the spectra should be resampled.
    # To avoid resampling, given that the wavelength dependence of the
    # correction is very weak, I approximate it with a constant factor.
    #
    wave *= np.median(util.vac_to_air(wave) / wave)

    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    _, _, velscale = util.log_rebin(np.array([wave[0], wave[-1]]), galaxy)
    #velscale = c * np.log(wave[1] / wave[0])  # eq.(8) of Cappellari (2017)
    FWHM_gal = mr.rfwhm  # SDSS has an approximate instrumental resolution FWHM of 2.76A.

    #------------------- Setup templates -----------------------

    pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

    # The templates are not normalized.
    # In this way the weights and mean values are mass-weighted quantities.
    # Use the keyword 'norm_range' for light-weighted quantities.
    miles = lib.miles(pathname, velscale, FWHM_gal)

    # The stellar templates are reshaped below into a 2-dim array with each
    # spectrum as a column, however we save the original array dimensions,
    # which are needed to specify the regularization dimensions
    #
    reg_dim = miles.templates.shape[1:]
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

    # See the pPXF documentation for the keyword REGUL
    regul_err = np.nanmedian(noise)  # Desired regularization error

    # Estimate the wavelength fitted range in the rest frame.
    lam_range_gal = np.array([np.min(wave), np.max(wave)]) / (1 + z)

    # Construct a set of Gaussian emission line templates.
    # The `emission_lines` function defines the most common lines, but additional
    # lines can be included by editing the function in the file ppxf_util.py.
    gas_templates, gas_names, line_wave = util.emission_lines(
        miles.log_lam_temp,
        lam_range_gal,
        FWHM_gal,
        tie_balmer=tie_balmer,
        limit_doublets=limit_doublets)

    # Combines the stellar and gaseous templates into a single array.
    # During the PPXF fit they will be assigned a different kinematic
    # COMPONENT value
    #
    templates = np.column_stack([stars_templates, gas_templates])

    #-----------------------------------------------------------

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in ppxf_example_kinematics_sauron.py and Sec.2.4 of Cappellari (2017)
    #
    dv = c * (miles.log_lam_temp[0] - np.log(wave[0])
              )  # eq.(8) of Cappellari (2017)
    vel = c * np.log(1 + z)  # eq.(8) of Cappellari (2017)
    start = [vel, 180.]  # (km/s), starting guess for [V, sigma]

    n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a
                          for a in gas_names])  # forbidden lines contain "[*]"
    n_balmer = len(gas_names) - n_forbidden

    # Assign component=0 to the stellar templates, component=1 to the Balmer
    # gas emission lines templates and component=2 to the forbidden lines.
    component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
    gas_component = np.array(
        component) > 0  # gas_component=True for gas templates

    # Fit (V, sig, h3, h4) moments=4 for the stars
    # and (V, sig) moments=2 for the two gas kinematic components
    moments = [4, 2, 2]

    # Adopt the same starting value for the stars and the two gas components
    start = [start, start, start]

    # If the Balmer lines are tied one should allow for gas reddeining.
    # The gas_reddening can be different from the stellar one, if both are fitted.
    gas_reddening = 0 if tie_balmer else None

    # Here the actual fit starts.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (degree=-1) and only use
    # multiplicative ones (mdegree=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    try:
        pp = ppxf(templates,
                  galaxy,
                  noise,
                  velscale,
                  start,
                  plot=False,
                  moments=moments,
                  degree=-1,
                  mdegree=10,
                  clean=False,
                  vsyst=dv,
                  lam=wave,
                  regul=1. / regul_err,
                  reg_dim=reg_dim,
                  component=component,
                  gas_component=gas_component,
                  gas_names=gas_names,
                  gas_reddening=gas_reddening)
    except Exception:
        plt.figure(i + 10000)
        plt.clf()
        plt.plot(wave, galaxy)
        plt.xlabel('Wavelength')
        plt.ylabel('Maggies')
        plt.tight_layout()
        plt.savefig(
            '/home/sfr/ppxf/{}/manga_7495_12704_ppxf_fitting_failed_{}.png'.
            format(foldername, i))
        continue

    np.save('/home/sfr/ppxf/{}/manga_7495_12704_ppxf_{}'.
        format(foldername, i), pp)
    np.save('/home/sfr/ppxf/{}/manga_7495_12704_ppxf_{}_miles_model'.
        format(foldername, i), miles)

    # When the two Delta Chi^2 below are the same, the solution
    # is the smoothest consistent with the observed spectrum.
    #
    print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
    print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

    # Exclude weights of the gas templates
    weights = pp.weights[~gas_component]
    weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.plot(age, sfr)
    plt.xlabel('Lookback Time / Gyr')
    plt.ylabel('Relative Star Formation Rate')
    plt.tight_layout()
    plt.savefig('/home/sfr/ppxf/{}/manga_7495_12704_ppxf_sfr_{}.png'.format(
        foldername, i))

    miles.mean_age_metal(weights)
    miles.mass_to_light(weights, band="r")

    # Plot fit results for stars and gas.
    plt.figure(figsize=(16, 8))
    plt.clf()
    pp.plot()
    plt.tight_layout()
    plt.savefig(
        '/home/sfr/ppxf/{}/manga_7495_12704_ppxf_fitted_model_{}.png'.format(
            foldername, i))

    # Plot stellar population mass-fraction distribution
    plt.figure(figsize=(16, 8))
    plt.clf()
    miles.plot(weights)
    plt.tight_layout()
    plt.savefig(
        '/home/sfr/ppxf/{}/manga_7495_12704_ppxf_age_metallicity_{}.png'.
        format(foldername, i))
