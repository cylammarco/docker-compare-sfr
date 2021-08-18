import os

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.io import read_results as reader
from prospect.models import priors
from prospect.models.sedmodel import SedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from prospect.utils.obsutils import fix_obs

ngc = fits.open('/home/sfr/example/ppxf_example_data/NGC4636_SDSS_DR12.fits')
# log10(wavelength [Å])
wave = 10.**(ngc[1].data['loglam'])
# coadded calibrated flux [10-17 ergs/s/cm2/Å]
# divide by 3631. to turn the unit into maggies
flux = ngc[1].data['flux'] * 1e-17 * 3.33564095E+04 * wave**2. / 3631.
# inverse variance of flux
err = np.sqrt(
    1. / ngc[1].data['ivar']) * 1e-17 * 3.33564095E+04 * wave**2. / 3631.

# prospector
# prospector
# prospector
# prospector
# prospector
# prospector
# prospector
# prospector
# prospector
# prospector
# prospector
# prospector

# The obs dictionary, empty for now
obs = {}

obs["filters"] = None
obs["maggies"] = None
obs["maggies_unc"] = None
obs["phot_mask"] = None
obs["phot_wave"] = None

# We do not have a spectrum, so we set some required elements of the obs
# dictionary to None.
# (this would be a vector of vacuum wavelengths in angstroms)
obs["wavelength"] = wave
# (this would be the spectrum in units of maggies)
obs["spectrum"] = flux
# (spectral uncertainties are given here)
obs['unc'] = err
# (again, to ignore a particular wavelength set the value of the
#  corresponding elemnt of the mask to *False*)
obs['mask'] = np.ones_like(wave, dtype=bool)

obs = fix_obs(obs)

# establish bounds
plt.figure(figsize=(16, 8))

# plot all the data
plt.plot(obs['wavelength'],
         obs['spectrum'],
         label='NGC 4636',
         marker='o',
         markersize=5,
         alpha=0.8,
         ls='',
         lw=3,
         color='slateblue')

# prettify
plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
plt.savefig('/home/sfr/prospector/ngc4636_prospector_input.png')

mass_param = {
    "name": "mass",
    # The mass parameter here is a scalar, so it has N=1
    "N": 1,
    # We will be fitting for the mass, so it is a free parameter
    "isfree": True,
    # This is the initial value. For fixed parameters this is the
    # value that will always be used.
    "init": 1e8,
    # This sets the prior probability for the parameter
    "prior": priors.LogUniform(mini=1e7, maxi=1e9),
    # this sets the initial dispersion to use when generating
    # clouds of emcee "walkers".  It is not required, but can be very helpful.
    "init_disp": 1e6,
    # this sets the minimum dispersion to use when generating
    # clouds of emcee "walkers".  It is not required, but can be useful if
    # burn-in rounds leave the walker distribution too narrow for some reason.
    "disp_floor": 1e6,
    # This is not required, but can be helpful
    "units": "solar masses formed",
}

# Look at all the prepackaged parameter sets
TemplateLibrary.show_contents()
TemplateLibrary.describe("parametric_sfh")

model_params = TemplateLibrary["parametric_sfh"]

# Now add the lumdist parameter by hand as another entry in the dictionary.
# This will control the distance since we are setting the redshift to zero.
# In `build_obs` above we used a distance of 10Mpc to convert from absolute
# to apparent magnitudes, so we use that here too, since the `maggies` are
# appropriate for that distance.
model_params["lumdist"] = {
    "N": 1,
    "isfree": False,
    "init": 10.,
    "units": "Mpc"
}

# Let's make some changes to initial values appropriate for our objects
# and data
model_params["zred"]["init"] = 0.0
model_params["dust2"]["init"] = 0.05
model_params["logzsol"]["init"] = -0.5
model_params["tage"]["init"] = 13.
model_params["mass"]["init"] = 1e8

# These are dwarf galaxies, so lets also adjust the metallicity prior,
# the tau parameter upward, and the mass prior downward
model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=1e2)
model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

# If we are going to be using emcee, it is useful to provide a
# minimum scale for the cloud of walkers (the default is 0.1)
model_params["mass"]["disp_floor"] = 1e6
model_params["tau"]["disp_floor"] = 1.0
model_params["tage"]["disp_floor"] = 1.0

# Set metallicity as a free parameter
model_params["logzsol"]["isfree"] = True
# And use value supplied by fixed_metallicity keyword
#model_params["logzsol"]['init'] = fixed_metallicity

model_params["zred"]['isfree'] = False
# And set the value to the object_redshift keyword
model_params["zred"]['init'] = 0.003129

# Add dust emission (with fixed dust SED parameters)
# Since `model_params` is a dictionary of parameter specifications, and
# `TemplateLibrary` returns dictionaries of parameter specifications,
# we can just update `model_params` with the parameters described in
# the pre-packaged `dust_emission` parameter set.
model_params.update(TemplateLibrary["dust_emission"])

# Now instantiate the model object using this dictionary of parameter
# specifications
model = SedModel(model_params)

print(model)
print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
print("Initial parameter dictionary:\n{}".format(model.params))

sps = CSPSpecBasis(zcontinuous=1)

# Generate the model SED at the initial value of theta
theta = model.theta.copy()
initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
title_text = ','.join(
    ["{}={}".format(p, model.params[p][0]) for p in model.free_params])

a = 1.0 + model.params.get('zred', 0.0)  # cosmological redshifting

wspec = sps.wavelengths
wspec *= a  # redshift them

print(sps.ssp.libraries)

# --- start minimization ----
run_params = {}
run_params["dynesty"] = False
run_params["emcee"] = False
run_params["optimize"] = True
run_params["min_method"] = 'lm'
# We'll start minimization from "nmin" separate places,
# the first based on the current values of each parameter and the
# rest drawn from the prior.  Starting from these extra draws
# can guard against local minima, or problems caused by
# starting at the edge of a prior (e.g. dust2=0.0)
run_params["nmin"] = 10

os.chdir('/home/sfr/prospector')
output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)

print("Done optmization in {}s".format(output["optimization"][1]))

print(model.theta)
(results, topt) = output["optimization"]
# Find which of the minimizations gave the best result,
# and use the parameter vector for that minimization
ind_best = np.argmin([r.cost for r in results])
print(ind_best)
theta_best = results[ind_best].x.copy()
print(theta_best)

# generate model
prediction = model.mean_model(theta_best, obs=obs, sps=sps)
pspec, pphot, pfrac = prediction

plt.figure(figsize=(16, 8))

# plot Data, best fit model, and old models
plt.loglog(obs['wavelength'],
           obs['spectrum'],
           label='Input spectrum',
           lw=1.0,
           color='gray',
           alpha=0.75)
plt.loglog(obs['wavelength'],
           pspec,
           label='Model spectrum',
           lw=1.0,
           alpha=0.75)

# Prettify
plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
plt.savefig('/home/sfr/prospector/ngc4636_prospector_fitted_model.jpg')

hfile = "/home/sfr/prospector/ngc4636_mcmc.h5"
writer.write_hdf5(hfile,
                  run_params,
                  model,
                  obs,
                  output["sampling"][0],
                  output["optimization"][0],
                  tsample=output["sampling"][1],
                  toptimize=output["optimization"][1],
                  sps=sps)

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
res, obs, _ = reader.results_from("/home/sfr/prospector/ngc4636_mcmc.h5")


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

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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
t = ngc['COADD'].data
z = 0.003129  # SDSS redshift estimate

# Only use the wavelength range in common between galaxy and stellar library.
#
mask = (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409))
flux = t['flux'][mask]
galaxy = flux / np.median(flux)  # Normalize spectrum to avoid numerical issues
wave = 10.**t['loglam'][mask]

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
c = 299792.458
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
gas_component = np.array(component) > 0  # gas_component=True for gas templates

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
pp = ppxf(templates,
          galaxy,
          noise,
          velscale,
          start,
          plot=False,
          moments=moments,
          degree=-1,
          mdegree=10,
          vsyst=dv,
          lam=wave,
          clean=False,
          regul=1. / regul_err,
          reg_dim=reg_dim,
          component=component,
          gas_component=gas_component,
          gas_names=gas_names,
          gas_reddening=gas_reddening)

# When the two Delta Chi^2 below are the same, the solution
# is the smoothest consistent with the observed spectrum.
#
print('Desired Delta Chi^2: %#.4g' % np.sqrt(2 * galaxy.size))
print('Current Delta Chi^2: %#.4g' % ((pp.chi2 - 1) * galaxy.size))

weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
weights = weights.reshape(reg_dim) / weights.sum()  # Normalized

miles.mean_age_metal(weights)
miles.mass_to_light(weights, band="r")

# Plot fit results for stars and gas.
plt.figure(figsize=(16, 8))
pp.plot()
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/ngc4636_ppxf_fitted_model.png')

# Plot stellar population mass-fraction distribution
plt.figure(figsize=(16, 8))
miles.plot(weights)
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/ngc4636_ppxf_age_metallicity.png')













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
