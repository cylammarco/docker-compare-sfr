import os

from astropy.io import fits
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt
import numpy as np

from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.io import read_results as reader
from prospect.models import priors
from prospect.models import transforms
from prospect.models.sedmodel import SedModel
from prospect.models.templates import adjust_dirichlet_agebins
from prospect.models.templates import TemplateLibrary
from prospect.utils.obsutils import fix_obs
from prospect.sources import FastSSPBasis
from prospect.sources import FastStepBasis

ngc = fits.open('/home/sfr/example/ppxf_example_data/NGC3522_SDSS_DR8.fits')
# log10(wavelength [Å])
wave = ngc[1].data['wavelength']

# coadded calibrated flux [10-17 ergs/s/cm2/Å]
# divide by 3631. to turn the unit into maggies
flux = ngc[1].data['flux'] * 1e-17 * 3.33564095E+04 * wave**2. / 3631.
# inverse variance of flux
err = np.sqrt(1. / ngc[1].data['inverse_variance']
              ) * 1e-17 * 3.33564095E+04 * wave**2. / 3631.

z = 0.004153  # SDSS redshift estimate

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


model_params = TemplateLibrary['dirichlet_sfh']
model_params["zred"]['isfree'] = False
model_params["zred"]['init'] = z

dirichlet_agelims = np.array([1e-9, 0.3, 3.0, 13.6])
model_params = adjust_dirichlet_agebins(
    model_params, agelims=(np.log10(dirichlet_agelims) + 9))

sps_dirichlet = FastStepBasis(zcontinuous=1)
sed_model = SedModel(model_params)

print("\nInitial free parameter vector theta:\n  {}\n".format(sed_model.theta))
print("Initial parameter dictionary:\n{}".format(sed_model.params))

# Generate the model SED at the initial value of theta
theta = sed_model.theta.copy()
title_text = ','.join(
    ["{}={}".format(p, sed_model.params[p][0]) for p in sed_model.free_params])

a = 1.0 + sed_model.params.get('zred', 0.0)  # cosmological redshifting

wspec = sps_dirichlet.wavelengths
wspec *= a  # redshift them

print(sps_dirichlet.ssp.libraries)


# We'll start minimization from "nmin" separate places,
# the first based on the current values of each parameter and the
# rest drawn from the prior.  Starting from these extra draws
# can guard against local minima, or problems caused by
# starting at the edge of a prior (e.g. dust2=0.0)
run_params = {}
run_params['verbose'] = True
run_params["nmin"] = 0
# Number of emcee walkers
run_params["nwalkers"] = 32
# Number of iterations of the MCMC sampling
run_params["niter"] = 2048
run_params["nburn"] = [
    int(run_params["niter"] * 0.05),
    int(run_params["niter"] * 0.1)
]

run_params["dynesty"] = False
run_params["emcee"] = True
run_params["optimize"] = False

# --- start minimization ----
os.chdir('/home/sfr/prospector')
output = fit_model(obs,
                   sed_model,
                   sps_dirichlet,
                   lnprobfn=lnprobfn,
                   **run_params)

result = output['sampling'][0]

nwalkers, niter = run_params['nwalkers'], run_params['niter']
theta = []
for i in range(nwalkers):
    theta.append(result.chain[i, -1])

theta_mean = np.mean(sigma_clip(theta, axis=0), axis=0)

# generate model
prediction = sed_model.mean_model(theta_mean, obs, sps=sps_dirichlet)
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

if os.path.exists('/home/sfr/prospector/ngc3522_dirichlet_fitted_model.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_dirichlet_fitted_model.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_dirichlet_fitted_model.jpg')

hfile = "/home/sfr/prospector/ngc3522_dirichlet_mcmc.h5"
if os.path.exists(hfile):
    os.remove(hfile)

# Does not seem to be writing model, output and sps.
writer.write_hdf5(
    hfile,
    model_params,
    sed_model,
    obs,
    sampler=result,
    #                  optimize_result_list=output["optimization"][0],
    tsample=output["sampling"][1],
    toptimize=output["optimization"][1],
    sps=sps_dirichlet)

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
res, o, m = reader.results_from("/home/sfr/prospector/ngc3522_dirichlet_mcmc.h5",
                                model=sed_model)

n_mcmc_grid = len(res['theta_labels'])

cornerfig = reader.subcorner(res,
                             start=0,
                             thin=n_mcmc_grid,
                             truths=theta_mean,
                             fig=plt.subplots(n_mcmc_grid,
                                              n_mcmc_grid,
                                              figsize=(n_mcmc_grid * 2,
                                                       n_mcmc_grid * 2))[0])

if os.path.exists('/home/sfr/prospector/ngc3522_dirichlet_mcmc_corner.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_dirichlet_mcmc_corner.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_dirichlet_mcmc_corner.jpg')


for i, label in enumerate(res['theta_labels']):
    print('{} = {}'.format(label, theta_mean[i]))







dirichlet_agelims = np.array([1e-9, 0.1, 0.3, 1.0, 3.0, 6.0, 13.6])
model_params = adjust_dirichlet_agebins(
    model_params, agelims=(np.log10(dirichlet_agelims) + 9))

# Use the mean parameters as the initial values of the prospector-alpha model
model_params["logzsol"]["init"] = theta_mean[0]
model_params["dust2"]["init"] = theta_mean[1]
model_params["total_mass"]["init"] = theta_mean[4]

run_params["nmin"] = 0
# Number of emcee walkers
run_params["nwalkers"] = 32
# Number of iterations of the MCMC sampling
run_params["niter"] = 4096
run_params["nburn"] = [
    int(run_params["niter"] * 0.05),
    int(run_params["niter"] * 0.1)
]

sed_model = SedModel(model_params)

print("\nInitial free parameter vector theta:\n  {}\n".format(
    sed_model.theta))
print("Initial parameter dictionary:\n{}".format(sed_model.params))

# Generate the model SED at the initial value of theta
theta_dirichlet = sed_model.theta.copy()
title_text = ','.join([
    "{}={}".format(p, sed_model.params[p][0])
    for p in sed_model.free_params
])

a = 1.0 + sed_model.params.get('zred', 0.0)  # cosmological redshifting

wspec = sps_dirichlet.wavelengths
wspec *= a  # redshift them

print(sps_dirichlet.ssp.libraries)


# --- start minimization ----

os.chdir('/home/sfr/prospector')
output_dirichlet = fit_model(obs,
                             sed_model,
                             sps_dirichlet,
                             lnprobfn=lnprobfn,
                             **run_params)

result_dirichlet = output_dirichlet['sampling'][0]

nwalkers, niter = run_params['nwalkers'], run_params[
    'niter']
theta_dirichlet = []
for i in range(nwalkers):
    theta_dirichlet.append(result_dirichlet.chain[i, -1])

theta_dirichlet_mean = np.mean(theta_dirichlet, axis=0)

# generate model
prediction_dirichlet = sed_model.mean_model(theta_dirichlet_mean,
                                                  obs,
                                                  sps=sps_dirichlet)
pspec_dirichlet, pphot_dirichlet, pfrac_dirichlet = prediction_dirichlet

plt.figure(figsize=(16, 8))

# plot Data, best fit model, and old models
plt.loglog(obs['wavelength'],
           obs['spectrum'],
           label='Input spectrum',
           lw=1.0,
           color='gray',
           alpha=0.75)
plt.loglog(obs['wavelength'],
           pspec_dirichlet,
           label='Model spectrum',
           lw=1.0,
           alpha=0.75)

# Prettify
plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.legend(loc='best', fontsize=20)
plt.tight_layout()

if os.path.exists('/home/sfr/prospector/ngc3522_dirichlet_fitted_model_step_2.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_dirichlet_fitted_model_step_2.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_dirichlet_fitted_model_step_2.jpg')

hfile = "/home/sfr/prospector/ngc3522_dirichlet_mcmc_step_2.h5"
if os.path.exists(hfile):
    os.remove(hfile)

# Does not seem to be writing model, output and sps.
writer.write_hdf5(
    hfile,
    model_params,
    sed_model,
    obs,
    sampler=result_dirichlet,
    #                  optimize_result_list=output["optimization"][0],
    tsample=output["sampling"][1],
    toptimize=output["optimization"][1],
    sps=sps_dirichlet)

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
res_dirichlet, o, m = reader.results_from(
    "/home/sfr/prospector/ngc3522_dirichlet_mcmc_step_2.h5", model=model_params)

n_mcmc_grid_dirichlet = len(res_dirichlet['theta_labels'])

cornerfig = reader.subcorner(res_dirichlet,
                             start=0,
                             thin=n_mcmc_grid_dirichlet,
                             truths=theta_dirichlet_mean,
                             fig=plt.subplots(
                                 n_mcmc_grid_dirichlet,
                                 n_mcmc_grid_dirichlet,
                                 figsize=(n_mcmc_grid_dirichlet * 2,
                                          n_mcmc_grid_dirichlet * 2))[0])

if os.path.exists('/home/sfr/prospector/ngc3522_dirichlet_corner_step_2.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_dirichlet_corner_step_2.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_dirichlet_corner_step_2.jpg')

# Plot the SFH here
theta_logzsol = theta_dirichlet_mean[0]
theta_dust2 = theta_dirichlet_mean[1]
theta_fraction_1 = theta_dirichlet_mean[2]
theta_fraction_2 = theta_dirichlet_mean[3]
theta_fraction_3 = theta_dirichlet_mean[4]
theta_fraction_4 = theta_dirichlet_mean[5]
theta_fraction_5 = theta_dirichlet_mean[6]
theta_total_mass = theta_dirichlet_mean[7]
'''
theta_duste_umin = theta_dirichlet_mean[8]
theta_duste_qpah = theta_dirichlet_mean[9]
theta_gamma = theta_dirichlet_mean[10]
theta_fagn = theta_dirichlet_mean[11]
theta_agn_tau = theta_dirichlet_mean[12]
theta_dust_ratio = theta_dirichlet_mean[13]
theta_dust_index = theta_dirichlet_mean[14]
'''
z_fraction = np.array((theta_fraction_1, theta_fraction_2, theta_fraction_3,
                       theta_fraction_4, theta_fraction_5))
agebins = sed_model.__dict__['params']['agebins']
sfr = transforms.zfrac_to_sfr(total_mass=theta_total_mass,
                              z_fraction=z_fraction,
                              agebins=agebins)

plt.figure(figsize=(8, 8))
plt.clf()

agebins_edges = np.append(agebins[:, 0], agebins[:, 1][-1])

# plot Data, best fit model, and old models
#for i, age in enumerate(agebins):
#    plt.plot(age, [sfr[i], sfr[i]], lw=1.0, alpha=0.75, color='C0')

#plt.bar(agebins_edges[:-1], height=sfr, width=np.diff(agebins_edges), align='edge')
plt.plot(np.mean(agebins, axis=1), sfr, drawstyle='steps-mid', color='C0')
plt.plot(agebins[-1], [sfr[-1], sfr[-1]], color='C0')
# Prettify
plt.xlabel('log(Age) / yr')
plt.ylabel('Star Formation Rate')
plt.ylim(ymin=0)
plt.xlim(4., np.log10(14e9))
plt.tight_layout()

if os.path.exists('/home/sfr/prospector/ngc3522_dirichlet_sfh_step_2.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_dirichlet_sfh_step_2.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_dirichlet_sfh_step_2.jpg')







model_params = TemplateLibrary['alpha']
model_params["zred"]['isfree'] = False
model_params["zred"]['init'] = z

# Use the mean parameters as the initial values of the prospector-alpha model
model_params["logzsol"]["init"] = theta_dirichlet_mean[0]
model_params["dust2"]["init"] = theta_dirichlet_mean[1]
model_params["z_fraction"]["init"] = theta_dirichlet_mean[2:7]
model_params["total_mass"]["init"] = theta_dirichlet_mean[7]

run_params["nmin"] = 0
# Number of emcee walkers
run_params["nwalkers"] = 32
# Number of iterations of the MCMC sampling
run_params["niter"] = 4096
run_params["nburn"] = [
    int(run_params["niter"] * 0.05),
    int(run_params["niter"] * 0.1)
]

sed_model = SedModel(model_params)

print("\nInitial free parameter vector theta:\n  {}\n".format(
    sed_model.theta))
print("Initial parameter dictionary:\n{}".format(sed_model.params))

# Generate the model SED at the initial value of theta
theta_alpha = sed_model.theta.copy()
title_text = ','.join([
    "{}={}".format(p, sed_model.params[p][0])
    for p in sed_model.free_params
])

a = 1.0 + sed_model.params.get('zred', 0.0)  # cosmological redshifting

wspec = sps_dirichlet.wavelengths
wspec *= a  # redshift them

print(sps_dirichlet.ssp.libraries)


# --- start minimization ----

os.chdir('/home/sfr/prospector')
output_alpha = fit_model(obs,
                             sed_model,
                             sps_dirichlet,
                             lnprobfn=lnprobfn,
                             **run_params)

result_alpha = output_alpha['sampling'][0]

nwalkers, niter = run_params['nwalkers'], run_params[
    'niter']
theta_alpha = []
for i in range(nwalkers):
    theta_alpha.append(result_alpha.chain[i, -1])

theta_alpha_mean = np.mean(theta_alpha, axis=0)

# generate model
prediction_alpha = sed_model.mean_model(theta_alpha_mean,
                                                  obs,
                                                  sps=sps_dirichlet)
pspec_alpha, pphot_alpha, pfrac_alpha = prediction_alpha

plt.figure(figsize=(16, 8))

# plot Data, best fit model, and old models
plt.loglog(obs['wavelength'],
           obs['spectrum'],
           label='Input spectrum',
           lw=1.0,
           color='gray',
           alpha=0.75)
plt.loglog(obs['wavelength'],
           pspec_alpha,
           label='Model spectrum',
           lw=1.0,
           alpha=0.75)

# Prettify
plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.legend(loc='best', fontsize=20)
plt.tight_layout()

if os.path.exists('/home/sfr/prospector/ngc3522_alpha_fitted_model.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_alpha_fitted_model.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_alpha_fitted_model.jpg')

hfile = "/home/sfr/prospector/ngc3522_alpha_mcmc.h5"
if os.path.exists(hfile):
    os.remove(hfile)

# Does not seem to be writing model, output and sps.
writer.write_hdf5(
    hfile,
    model_params,
    sed_model,
    obs,
    sampler=result_alpha,
    #                  optimize_result_list=output["optimization"][0],
    tsample=output["sampling"][1],
    toptimize=output["optimization"][1],
    sps=sps_dirichlet)

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
res_alpha, o, m = reader.results_from(
    "/home/sfr/prospector/ngc3522_alpha_mcmc.h5", model=model_params)

n_mcmc_grid_alpha = len(res_alpha['theta_labels'])

cornerfig = reader.subcorner(res_alpha,
                             start=0,
                             thin=n_mcmc_grid_alpha,
                             truths=theta_alpha_mean,
                             fig=plt.subplots(
                                 n_mcmc_grid_alpha,
                                 n_mcmc_grid_alpha,
                                 figsize=(n_mcmc_grid_alpha * 2,
                                          n_mcmc_grid_alpha * 2))[0])

if os.path.exists('/home/sfr/prospector/ngc3522_alpha_corner.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_alpha_corner.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_alpha_corner.jpg')

# Plot the SFH here
theta_logzsol = theta_alpha_mean[0]
theta_dust2 = theta_alpha_mean[1]
theta_fraction_1 = theta_alpha_mean[2]
theta_fraction_2 = theta_alpha_mean[3]
theta_fraction_3 = theta_alpha_mean[4]
theta_fraction_4 = theta_alpha_mean[5]
theta_fraction_5 = theta_alpha_mean[6]
theta_total_mass = theta_alpha_mean[7]
theta_duste_umin = theta_alpha_mean[8]
theta_duste_qpah = theta_alpha_mean[9]
theta_gamma = theta_alpha_mean[10]
theta_fagn = theta_alpha_mean[11]
theta_agn_tau = theta_alpha_mean[12]
theta_dust_ratio = theta_alpha_mean[13]
theta_dust_index = theta_alpha_mean[14]

z_fraction = np.array((theta_fraction_1, theta_fraction_2, theta_fraction_3,
                       theta_fraction_4, theta_fraction_5))
agebins = sed_model.__dict__['params']['agebins']
sfr = transforms.zfrac_to_sfr(total_mass=theta_total_mass,
                              z_fraction=z_fraction,
                              agebins=agebins)

plt.figure(figsize=(8, 8))
plt.clf()

agebins_edges = np.append(agebins[:, 0], agebins[:, 1][-1])

# plot Data, best fit model, and old models
#for i, age in enumerate(agebins):
#    plt.plot(age, [sfr[i], sfr[i]], lw=1.0, alpha=0.75, color='C0')

#plt.bar(agebins_edges[:-1], height=sfr, width=np.diff(agebins_edges), align='edge')
plt.plot(np.mean(agebins, axis=1), sfr, drawstyle='steps-mid', color='C0')
plt.plot(agebins[-1], [sfr[-1], sfr[-1]], color='C0')
# Prettify
plt.xlabel('log(Age) / yr')
plt.ylabel('Star Formation Rate')
plt.ylim(ymin=0)
plt.xlim(4., np.log10(14e9))
plt.tight_layout()

if os.path.exists('/home/sfr/prospector/ngc3522_alpha_sfh.jpg'):
    os.remove('/home/sfr/prospector/ngc3522_alpha_sfh.jpg')

plt.savefig('/home/sfr/prospector/ngc3522_alpha_sfh.jpg')








# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf
# ppxf

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

# coadded calibrated flux [10-17 ergs/s/cm2/Å]
# divide by 3631. to turn the unit into maggies
flux = ngc[1].data['flux'] * 1e-17
# inverse variance of flux
err = np.sqrt(1. / ngc[1].data['inverse_variance']) * 1e-17
wave = ngc[1].data['wavelength']

# Only use the wavelength range in common between galaxy and stellar library.
#
mask = (wave > 3540.) & (wave < 7400.)
norm_factor = np.median(flux)

galaxy = flux[mask]  # Normalize spectrum to avoid numerical issues
noise = err[mask]
wave = wave[mask]

noise[~np.isfinite(noise)] = max(noise[np.isfinite(noise)])

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
#noise = np.full_like(galaxy, err)  # Assume constant noise per pixel here

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
regul_err = np.nanmedian(noise)  # Desired regularization error

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
          reddening=True,
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
plt.clf()
plt.scatter(age, sfr, s=5)
plt.plot(age, sfr, drawstyle='steps-mid')
plt.xlabel('Lookback Time / Gyr')
plt.ylabel('Relative Star Formation Rate')
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/ngc3522_ppxf_sfr.png')

miles.mean_age_metal(weights)
miles.mass_to_light(weights, band="r")

# Plot fit results for stars and gas.
plt.figure(figsize=(16, 8))
pp.plot()
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/ngc3522_ppxf_fitted_model.png')

# Plot stellar population mass-fraction distribution
plt.figure(figsize=(16, 8))
miles.plot(weights)
plt.tight_layout()
plt.savefig('/home/sfr/ppxf/ngc3522_ppxf_age_metallicity.png')

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
