import os

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

manga_datacube = fits.open('/home/sfr/example/manga_example_data/manga-7495-12704-LINCUBE.fits.gz')
# log10(wavelength [Å])
wave = manga_datacube[1].data['wavelength']

# coadded calibrated flux [10-17 ergs/s/cm2/Å]
# divide by 3631. to turn the unit into maggies
flux = manga_datacube[1].data['flux'] * 1e-17 * 3.33564095E+04 * wave**2. / 3631.
# inverse variance of flux
err = np.sqrt(1. / manga_datacube[1].data['inverse_variance']
              ) * 1e-17 * 3.33564095E+04 * wave**2. / 3631.

z = 0.02894  # SDSS redshift estimate

# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf

from time import perf_counter as clock
from os import path
import os

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib

##############################################################################

tie_balmer = True
limit_doublets = True

ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

# Read SDSS DR8 galaxy spectrum taken from here http://www.sdss3.org/dr8/
# The spectrum is *already* log rebinned by the SDSS DR8
# pipeline and log_rebin should not be used in this case.
#
t = manga_datacube[1].data

# Only use the wavelength range in common between galaxy and stellar library.
#
mask = (t['wavelength'] > 3540.) & (t['wavelength'] < 7400.)
flux = t['flux'][mask]
galaxy = flux / np.median(flux)  # Normalize spectrum to avoid numerical issues
wave = t['wavelength'][mask]

# The SDSS wavelengths are in vacuum, while the MILES ones are in air.
# For a rigorous treatment, the SDSS vacuum wavelengths should be
# converted into air wavelengths and the spectra should be resampled.
# To avoid resampling, given that the wavelength dependence of the
# correction is very weak, I approximate it with a constant factor.
#
wave *= np.median(util.vac_to_air(wave) / wave)

# The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
# A constant noise is not a bad approximation in the fitted wavelength
# range and reduces the noise in the fit.
#
noise = np.full_like(galaxy, 0.01635)  # Assume constant noise per pixel here

# The velocity step was already chosen by the SDSS pipeline
# and we convert it below to km/s
#
c = 299792.458  # speed of light in km/s
velscale = c * np.log(wave[1] / wave[0])  # eq.(8) of Cappellari (2017)
FWHM_gal = 2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.

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
regul_err = 0.013  # Desired regularization error

# Estimate the wavelength fitted range in the rest frame.
lam_range_gal = np.array([np.min(wave), np.max(wave)]) / (1 + z)

# Combines the stellar and gaseous templates into a single array.
# During the PPXF fit they will be assigned a different kinematic
# COMPONENT value
#
templates = stars_templates

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
c = 299792.458
dv = c * (miles.log_lam_temp[0] - np.log(wave[0])
          )  # eq.(8) of Cappellari (2017)
vel = c * np.log(1 + z)  # eq.(8) of Cappellari (2017)
start = [vel, 180.]  # (km/s), starting guess for [V, sigma]

n_temps = stars_templates.shape[1]

# Assign component=0 to the stellar templates, component=1 to the Balmer
# gas emission lines templates and component=2 to the forbidden lines.
component = [0] * n_temps

# Fit (V, sig, h3, h4) moments=4 for the stars
# and (V, sig) moments=2 for the two gas kinematic components
moments = 4

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
pp = ppxf(templates,
          galaxy,
          noise,
          velscale,
          start,
          plot=False,
          moments=moments,
          degree=4,
          mdegree=0,
          clean=True,
          vsyst=dv,
          lam=wave,
          regul=1. / regul_err,
          reg_dim=reg_dim,
          component=component)

# When the two Delta Chi^2 below are the same, the solution
# is the smoothest consistent with the observed spectrum.
#
print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

weights = pp.weights  # Exclude weights of the gas templates
weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
sfr = np.sum(weights, axis=1)
age = miles.age_grid[:, 0]
plt.figure(figsize=(10, 6))
plt.plot(age, sfr)
plt.xlabel('Lookback Time / Gyr')
plt.ylabel('Relative Star Formation Rate')
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/manga_datacube3522_ppxf_sfr.png')

miles.mean_age_metal(weights)
miles.mass_to_light(weights, band="r")

# Plot fit results for stars and gas.
plt.figure(figsize=(16, 8))
pp.plot()
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/manga_datacube3522_ppxf_fitted_model.png')

# Plot stellar population mass-fraction distribution
plt.figure(figsize=(16, 8))
miles.plot(weights)
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/manga_datacube3522_ppxf_age_metallicity.png')

# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
# pyPipe3D
