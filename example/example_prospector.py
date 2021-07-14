import os

import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from prospect.models import priors
from prospect.models.templates import TemplateLibrary
from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.io import read_results as reader
from prospect.utils.obsutils import fix_obs
import sedpy

rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'xtick.color': 'k'})
rcParams.update({'ytick.color': 'k'})
rcParams.update({'font.size': 20})

os.chdir('example/prospector_example_data')

# This is almost entirely copy-and-pasted from
# https://github.com/bd-j/prospector/blob/main/demo/InteractiveDemo.ipynb


def build_obs(snr=10, ldist=10.0, **extras):
    """Build a dictionary of observational data.  In this example
    the data consist of photometry for a single nearby dwarf galaxy
    from Johnson et al. 2013.

    :param snr:
        The S/N to assign to the photometry, since none are reported
        in Johnson et al. 2013

    :param ldist:
        The luminosity distance to assume for translating absolute magnitudes
        into apparent magnitudes.

    :returns obs:
        A dictionary of observational data to use in the fit.
    """

    # The obs dictionary, empty for now
    obs = {}

    # These are the names of the relevant filters,
    # in the same order as the photometric data (see below)
    galex = ['galex_FUV', 'galex_NUV']
    spitzer = ['spitzer_irac_ch' + n for n in ['1', '2', '3', '4']]
    sdss = ['sdss_{0}0'.format(b) for b in ['u', 'g', 'r', 'i', 'z']]
    filternames = galex + sdss + spitzer
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the
    # `obs` dictionary
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Now we store the measured fluxes for a single object, **in the same order
    # as "filters"**
    # In this example we use a row of absolute AB magnitudes from
    # Johnson et al. 2013 (NGC4163)
    # We then turn them into apparent magnitudes based on the supplied `ldist`
    # meta-parameter. You could also, e.g. read from a catalog. The units of
    # the fluxes need to be maggies (Jy/3631) so we will do the conversion
    # here too.
    M_AB = np.array([
        -11.93, -12.37, -13.37, -14.22, -14.61, -14.86, -14.94, -14.09, -13.62,
        -13.23, -12.78
    ])
    dm = 25 + 5.0 * np.log10(ldist)
    mags = M_AB + dm
    obs["maggies"] = 10**(-0.4 * mags)

    # And now we store the uncertainties (again in units of maggies)
    # In this example we are going to fudge the uncertainties based on the
    # supplied `snr` meta-parameter.
    obs["maggies_unc"] = (1. / snr) * obs["maggies"]

    # Now we need a mask, which says which flux values to consider in the
    # likelihood.
    # IMPORTANT: the mask is *True* for values that you *want* to fit,
    # and *False* for values you want to ignore.  Here we ignore the
    # spitzer bands.
    obs["phot_mask"] = np.array(
        ['spitzer' not in f.name for f in obs["filters"]])

    # This is an array of effective wavelengths for each of the filters.
    # It is not necessary, but it can be useful for plotting so we store it
    # here as a convenience
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # We do not have a spectrum, so we set some required elements of the obs
    # dictionary to None.
    # (this would be a vector of vacuum wavelengths in angstroms)
    obs["wavelength"] = None
    # (this would be the spectrum in units of maggies)
    obs["spectrum"] = None
    # (spectral uncertainties are given here)
    obs['unc'] = None
    # (again, to ignore a particular wavelength set the value of the
    #  corresponding elemnt of the mask to *False*)
    obs['mask'] = None

    # This function ensures all required keys are present in the obs
    # dictionary, adding default values if necessary
    obs = fix_obs(obs)

    return obs


run_params = {}
run_params["snr"] = 10.0
run_params["ldist"] = 10.0

# Build the obs dictionary using the meta-parameters
obs = build_obs(**run_params)

# Look at the contents of the obs dictionary
print("Obs Dictionary Keys:\n\n{}\n".format(obs.keys()))
print("--------\nFilter objects:\n")
print(obs["filters"])

# --- Plot the Data ----
# This is why we stored these...
wphot = obs["phot_wave"]

# establish bounds
xmin, xmax = np.min(wphot) * 0.8, np.max(wphot) / 0.8
ymin, ymax = obs["maggies"].min() * 0.8, obs["maggies"].max() / 0.4
plt.figure(figsize=(16, 8))

# plot all the data
plt.plot(wphot,
         obs['maggies'],
         label='All observed photometry',
         marker='o',
         markersize=12,
         alpha=0.8,
         ls='',
         lw=3,
         color='slateblue')

# overplot only the data we intend to fit
mask = obs["phot_mask"]
plt.errorbar(wphot[mask],
             obs['maggies'][mask],
             yerr=obs['maggies_unc'][mask],
             label='Photometry to fit',
             marker='o',
             markersize=8,
             alpha=0.8,
             ls='',
             lw=3,
             ecolor='tomato',
             markerfacecolor='none',
             markeredgecolor='tomato',
             markeredgewidth=3)

# plot Filters
for f in obs['filters']:
    w, t = f.wavelength.copy(), f.transmission.copy()
    t = t / t.max()
    t = 10**(0.2 * (np.log10(ymax / ymin))) * t * ymin
    plt.loglog(w, t, lw=3, color='gray', alpha=0.7)

# prettify
plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xscale("log")
plt.yscale("log")
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
plt.savefig('../../prospector/fig_1_photometric_points_and_filters.jpg')

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
    "prior": priors.LogUniform(mini=1e6, maxi=1e12),
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


def build_model(object_redshift=None,
                ldist=10.0,
                fixed_metallicity=None,
                add_duste=False,
                **extras):
    """Build a prospect.models.SedModel object

    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate
        for this redshift. Otherwise, the redshift will be zero.

    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed
        frame (apparent) photometry will be appropriate for this luminosity
        distance.

    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the
        given value.

    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to
        the model.

    :returns model:
        An instance of prospect.models.SedModel
    """
    from prospect.models.sedmodel import SedModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by
    # parameter name
    model_params = TemplateLibrary["parametric_sfh"]

    # Now add the lumdist parameter by hand as another entry in the dictionary.
    # This will control the distance since we are setting the redshift to zero.
    # In `build_obs` above we used a distance of 10Mpc to convert from absolute
    # to apparent magnitudes, so we use that here too, since the `maggies` are
    # appropriate for that distance.
    model_params["lumdist"] = {
        "N": 1,
        "isfree": False,
        "init": ldist,
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

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        # And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        # Since `model_params` is a dictionary of parameter specifications, and
        # `TemplateLibrary` returns dictionaries of parameter specifications,
        # we can just update `model_params` with the parameters described in
        # the pre-packaged `dust_emission` parameter set.
        model_params.update(TemplateLibrary["dust_emission"])

    # Now instantiate the model object using this dictionary of parameter
    # specifications
    model = SedModel(model_params)

    return model


run_params["object_redshift"] = 0.0
run_params["fixed_metallicity"] = None
run_params["add_duste"] = True

model = build_model(**run_params)
print(model)
print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
print("Initial parameter dictionary:\n{}".format(model.params))


def build_sps(zcontinuous=1, **extras):
    """
    :param zcontinuous:
        A vlue of 1 insures that we use interpolation between SSPs to
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps


run_params["zcontinuous"] = 1

sps = build_sps(**run_params)

# Generate the model SED at the initial value of theta
theta = model.theta.copy()
initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
title_text = ','.join(
    ["{}={}".format(p, model.params[p][0]) for p in model.free_params])

a = 1.0 + model.params.get('zred', 0.0)  # cosmological redshifting
# photometric effective wavelengths
wphot = obs["phot_wave"]
# spectroscopic wavelengths
if obs["wavelength"] is None:
    # *restframe* spectral wavelengths, since obs["wavelength"] is None
    wspec = sps.wavelengths
    wspec *= a  # redshift them
else:
    wspec = obs["wavelength"]

# establish bounds
xmin, xmax = np.min(wphot) * 0.8, np.max(wphot) / 0.8
temp = np.interp(np.linspace(xmin, xmax, 10000), wspec, initial_spec)
ymin, ymax = temp.min() * 0.8, temp.max() / 0.4
plt.figure(figsize=(16, 8))

# plot model + data
plt.loglog(wspec,
           initial_spec,
           label='Model spectrum',
           lw=0.7,
           color='navy',
           alpha=0.7)
plt.errorbar(wphot,
             initial_phot,
             label='Model photometry',
             marker='s',
             markersize=10,
             alpha=0.8,
             ls='',
             lw=3,
             markerfacecolor='none',
             markeredgecolor='blue',
             markeredgewidth=3)
plt.errorbar(wphot,
             obs['maggies'],
             yerr=obs['maggies_unc'],
             label='Observed photometry',
             marker='o',
             markersize=10,
             alpha=0.8,
             ls='',
             lw=3,
             ecolor='red',
             markerfacecolor='none',
             markeredgecolor='red',
             markeredgewidth=3)
plt.title(title_text)

# plot Filters
for f in obs['filters']:
    w, t = f.wavelength.copy(), f.transmission.copy()
    t = t / t.max()
    t = 10**(0.2 * (np.log10(ymax / ymin))) * t * ymin
    plt.loglog(w, t, lw=3, color='gray', alpha=0.7)

# prettify
plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
plt.savefig('../../prospector/fig_2_plotting_a_random_model.jpg')

verbose = False
run_params["verbose"] = verbose

# Here we will run all our building functions
obs = build_obs(**run_params)
sps = build_sps(**run_params)
model = build_model(**run_params)

# For fsps based sources it is useful to
# know which stellar isochrone and spectral library
# we are using
print(sps.ssp.libraries)

# --- start minimization ----
run_params["dynesty"] = False
run_params["emcee"] = False
run_params["optimize"] = True
run_params["min_method"] = 'lm'
# We'll start minimization from "nmin" separate places,
# the first based on the current values of each parameter and the
# rest drawn from the prior.  Starting from these extra draws
# can guard against local minima, or problems caused by
# starting at the edge of a prior (e.g. dust2=0.0)
run_params["nmin"] = 2

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
plt.loglog(wspec,
           initial_spec,
           label='Old model spectrum',
           lw=0.7,
           color='gray',
           alpha=0.5)
plt.errorbar(wphot,
             initial_phot,
             label='Old model Photometry',
             marker='s',
             markersize=10,
             alpha=0.6,
             ls='',
             lw=3,
             markerfacecolor='none',
             markeredgecolor='gray',
             markeredgewidth=3)
plt.loglog(wspec,
           pspec,
           label='Model spectrum',
           lw=0.7,
           color='slateblue',
           alpha=0.7)
plt.errorbar(wphot,
             pphot,
             label='Model photometry',
             marker='s',
             markersize=10,
             alpha=0.8,
             ls='',
             lw=3,
             markerfacecolor='none',
             markeredgecolor='slateblue',
             markeredgewidth=3)
plt.errorbar(wphot,
             obs['maggies'],
             yerr=obs['maggies_unc'],
             label='Observed photometry',
             marker='o',
             markersize=10,
             alpha=0.8,
             ls='',
             lw=3,
             ecolor='tomato',
             markerfacecolor='none',
             markeredgecolor='tomato',
             markeredgewidth=3)

# plot filter transmission curves
for f in obs['filters']:
    w, t = f.wavelength.copy(), f.transmission.copy()
    t = t / t.max()
    t = 10**(0.2 * (np.log10(ymax / ymin))) * t * ymin
    plt.loglog(w, t, lw=3, color='gray', alpha=0.7)

# Prettify
plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
plt.savefig('../../prospector/fig_3_plotting_a_fitted_model.jpg')

# Set this to False if you don't want to do another optimization
# before emcee sampling (but note that the "optimization" entry
# in the output dictionary will be (None, 0.) in this case)
# If set to true then another round of optmization will be performed
# before sampling begins and the "optmization" entry of the output
# will be populated.
run_params["optimize"] = False
run_params["emcee"] = True
run_params["dynesty"] = False
# Number of emcee walkers
run_params["nwalkers"] = 128
# Number of iterations of the MCMC sampling
run_params["niter"] = 512
# Number of iterations in each round of burn-in
# After each round, the walkers are reinitialized based on the
# locations of the highest probablity half of the walkers.
run_params["nburn"] = [16, 32, 64]

output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
print('done emcee in {0}s'.format(output["sampling"][1]))

hfile = "../../prospector/demo_emcee_mcmc.h5"
writer.write_hdf5(hfile,
                  run_params,
                  model,
                  obs,
                  output["sampling"][0],
                  output["optimization"][0],
                  tsample=output["sampling"][1],
                  toptimize=output["optimization"][1])

print('Finished')

run_params["dynesty"] = True
run_params["optmization"] = False
run_params["emcee"] = False
run_params["nested_method"] = "rwalk"
run_params["nlive_init"] = 400
run_params["nlive_batch"] = 200
run_params["nested_dlogz_init"] = 0.05
run_params["nested_posterior_thresh"] = 0.05
run_params["nested_maxcall"] = int(1e7)

output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
print('done dynesty in {0}s'.format(output["sampling"][1]))

hfile = "../../prospector/demo_dynesty_mcmc.h5"
writer.write_hdf5(hfile,
                  run_params,
                  model,
                  obs,
                  output["sampling"][0],
                  output["optimization"][0],
                  tsample=output["sampling"][1],
                  toptimize=output["optimization"][1])

print('Finished')

results_type = "emcee"  # | "dynesty"
# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
result, obs, _ = reader.results_from("../../prospector/demo_{}_mcmc.h5".format(results_type),
                                     dangerous=False)

# let's look at what's stored in the `result` dictionary
print(result.keys())

if results_type == "emcee":
    chosen = np.random.choice(result["run_params"]["nwalkers"],
                              size=10,
                              replace=False)
    tracefig = reader.traceplot(result, figsize=(20, 10), chains=chosen)
else:
    tracefig = reader.traceplot(result, figsize=(20, 10))

plt.savefig('../../prospector/fig_4_mcmc_trace.jpg')

# maximum a posteriori (of the locations visited by the MCMC sampler)
imax = np.argmax(result['lnprobability'])
if results_type == "emcee":
    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()
    thin = 5
else:
    theta_max = result["chain"][imax, :]
    thin = 1

print('Optimization value: {}'.format(theta_best))
print('MAP value: {}'.format(theta_max))
cornerfig = reader.subcorner(result,
                             start=0,
                             thin=thin,
                             truths=theta_best,
                             fig=plt.subplots(5, 5, figsize=(27, 27))[0])
plt.savefig('../../prospector/fig_5_mcmc_corner.jpg')

# randomly chosen parameters from chain
randint = np.random.randint
if results_type == "emcee":
    nwalkers, niter = run_params['nwalkers'], run_params['niter']
    theta = result['chain'][randint(nwalkers), randint(niter)]
else:
    theta = result["chain"][randint(len(result["chain"]))]

# generate models
# sps = reader.get_sps(result)  # this works if using parameter files
mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)
mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)

# Make plot of data and model
plt.figure(figsize=(16, 8))

plt.loglog(wspec,
           mspec,
           label='Model spectrum (random draw)',
           lw=0.7,
           color='navy',
           alpha=0.7)
plt.loglog(wspec,
           mspec_map,
           label='Model spectrum (MAP)',
           lw=0.7,
           color='green',
           alpha=0.7)
plt.errorbar(wphot,
             mphot,
             label='Model photometry (random draw)',
             marker='s',
             markersize=10,
             alpha=0.8,
             ls='',
             lw=3,
             markerfacecolor='none',
             markeredgecolor='blue',
             markeredgewidth=3)
plt.errorbar(wphot,
             mphot_map,
             label='Model photometry (MAP)',
             marker='s',
             markersize=10,
             alpha=0.8,
             ls='',
             lw=3,
             markerfacecolor='none',
             markeredgecolor='green',
             markeredgewidth=3)
plt.errorbar(wphot,
             obs['maggies'],
             yerr=obs['maggies_unc'],
             label='Observed photometry',
             ecolor='red',
             marker='o',
             markersize=10,
             ls='',
             lw=3,
             alpha=0.8,
             markerfacecolor='none',
             markeredgecolor='red',
             markeredgewidth=3)

# plot transmission curves
for f in obs['filters']:
    w, t = f.wavelength.copy(), f.transmission.copy()
    t = t / t.max()
    t = 10**(0.2 * (np.log10(ymax / ymin))) * t * ymin
    plt.loglog(w, t, lw=3, color='gray', alpha=0.7)

plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
plt.savefig('../../prospector/fig_6_regenerated_sed_from_stored_mcmc_chains.jpg')
