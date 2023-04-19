import os

from astropy import constants
from astropy.io import fits
import extinction
import numpy as np
from numpy.polynomial import legendre
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from spectres import spectres
from matplotlib.image import NonUniformImage
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate as itp

plt.ion()


def find_reg(
    reg_guess,
    templates,
    galaxy,
    noise,
    velscale,
    start,
    goodpixels,
    plot,
    moments,
    degree,
    mdegree,
    fixed,
    clean,
    ftol,
    lam,
    lam_temp,
    linear_method,
    reg_ord,
    bias,
    reg_dim,
    component,
    gas_component,
    gas_names,
    desired_chi2,
):
    reg_guess = 10.0 ** reg_guess[0]
    pp = ppxf(
        templates=templates,
        galaxy=galaxy,
        noise=noise,
        velscale=velscale,
        start=start,
        goodpixels=goodpixels,
        plot=plot,
        moments=moments,
        degree=degree,
        mdegree=mdegree,
        fixed=fixed,
        clean=clean,
        ftol=ftol,
        lam=lam,
        lam_temp=lam_temp,
        linear_method=linear_method,
        regul=reg_guess,
        reg_ord=reg_ord,
        bias=bias,
        reg_dim=reg_dim,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
        quiet=True,
    )
    chi2_diff = abs(pp.chi2 - desired_chi2)
    """
    print("")
    print(
        "================================================================================"
    )
    print("Current regularisation value is: {}.".format(reg_guess))
    print("Current Chi^2: {}".format(pp.chi2))
    print("Desired Chi^2: {}".format(desired_chi2))
    print("Current distance from the desired Chi^2 is: {}.".format(chi2_diff))
    print(
        "================================================================================"
    )
    print("")
    """
    return chi2_diff


def get_uncertainty(
    templates_corrected,
    galaxy_best_fit,
    noise_rescaled,
    velscale_rebinned,
    start,
    goodpixels,
    plot,
    moments,
    degree,
    mdegree,
    fixed,
    clean,
    ftol,
    lam,
    lam_temp,
    linear_method,
    regul,
    reg_ord,
    bias,
    reg_dim,
    component,
    gas_component,
    gas_names,
    residual,
):
    galaxy_output = []
    light_weight_output = []
    mass_weight_output = []
    sfr_light_output = []
    sfr_weight_output = []
    for _ in range(100):
        galaxy_noise_added = np.random.normal(
            loc=galaxy_best_fit, scale=residual
        )
        pp = ppxf(
            templates_corrected,
            galaxy_noise_added,
            noise_rescaled,
            velscale_rebinned,
            start,
            goodpixels=goodpixels,
            plot=plot,
            moments=moments,
            degree=degree,
            mdegree=mdegree,
            fixed=fixed,
            clean=clean,
            ftol=ftol,
            lam=lam,
            lam_temp=lam_temp,
            linear_method=linear_method,
            regul=regul,
            reg_ord=reg_ord,
            bias=bias,
            reg_dim=reg_dim,
            component=component,
            gas_component=gas_component,
            gas_names=gas_names,
        )
        light_weights = pp.weights[
            ~gas_component
        ]  # Exclude gas templates weights
        light_weights = light_weights.reshape(
            reg_dim
        )  # Reshape to a 2D matrix

        # convert from light to mass, hence 1./Mass-to-light
        mass_weights = light_weights / miles.flux
        mass_weights[np.isnan(mass_weights)] = 0.0

        sfr_light = np.median(light_weights, axis=1)
        sfr_light[np.isnan(sfr_light)] = 0.0

        sfr_mass = np.median(mass_weights, axis=1)
        sfr_mass[np.isnan(sfr_mass)] = 0.0

        galaxy_output.append(pp.bestfit)
        light_weight_output.append(light_weights)
        mass_weight_output.append(mass_weights)
        sfr_light_output.append(sfr_light)
        sfr_weight_output.append(sfr_mass)

    return (
        galaxy_output,
        light_weight_output,
        mass_weight_output,
        sfr_light_output,
        sfr_weight_output,
    )


data_firefly = fits.open("manga-firefly-v3_1_1-miles.fits.gz")
firefly_mask = np.where(
    data_firefly["GALAXY_INFO"].data["PLATEIFU"] == "9881-9102"
)[0][0]

ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
#miles_pathname = os.path.join(
#    ppxf_dir, "fsps_generated_template", "Eun1.30Z*.fits"
#)
miles_pathname = os.path.join(
    ppxf_dir, "miles_models", "Eun1.30Z*.fits"
)

data = fits.open(
    "manga-9881\manga-9881-9102-LOGCUBE-HYB10-MILESHC-MASTARSSP.fits.gz"
)

# speed of light
c = constants.c.to("km/s").value
# Data are at z=0.025
firefly_z = data_firefly[1].data["REDSHIFT"]

z = firefly_z[firefly_mask]
firefly_bin_id_list = data_firefly["SPATIAL_BINID"].data
firefly_sfh_list = data_firefly["STAR_FORMATION_HISTORY"].data
firefly_spatial_info_list = data_firefly["SPATIAL_INFO"].data

# Fit (V, sig, h3, h4) moments=4 for the stars
# and (V, sig) moments=2 for the two gas kinematic components
moments = [4, 2, 2]

factor = 10  # Oversampling integer factor for an accurate convolution
h3 = 0.1  # Adopted G-H parameters of the LOSVD
h4 = 0.1

FWHM_gal = (
    2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
)

wave = data["WAVE"].data / (1 + z)
wave_new = np.arange(wave[0], 9000.0, FWHM_gal)
velscale = c * np.log(wave_new[1] / wave_new[0])

# Estimate the wavelength fitted range in the rest frame.
lam_range_gal = np.array([np.min(wave), 9000.0])

### for setting up the MILES
_galaxy = data["FLUX"].data[:, 0, 0]
_galaxy = spectres(wave_new, wave, _galaxy, fill=0.0)

# natural log
(
    _,
    _,
    velscale_rebinned,
) = util.log_rebin(
    wave_new,
    _galaxy,
    velscale=velscale,
)

miles = lib.miles(
    miles_pathname,
    velscale_rebinned,
    FWHM_gal,
    age_range=None,
    norm_range=[5070, 5950],
    metal_range=[-0.05, 0.05],
    wave_range=(3000, 9500),
)
reg_dim = miles.templates.shape[1:]
star_templates = miles.templates.reshape(miles.templates.shape[0], -1)
n_temps = star_templates.shape[1]

tie_balmer = True
limit_doublets = True

# Construct a set of Gaussian emission line templates.
# The `emission_lines` function defines the most common lines, but additional
# lines can be included by editing the function in the file ppxf_util.py.
gas_templates, gas_names, line_wave = util.emission_lines(
    miles.ln_lam_temp,
    lam_range_gal,
    FWHM_gal,
    tie_balmer=tie_balmer,
    limit_doublets=limit_doublets,
    pixel=True,
)

# Combines the stellar and gaseous templates into a single array.
# During the PPXF fit they will be assigned a different kinematic
# COMPONENT value
#
templates = np.column_stack([star_templates, gas_templates])

# If the Balmer lines are tied one should allow for gas reddeining.
# The gas_reddening can be different from the stellar one, if both are fitted.
gas_reddening = 0 if tie_balmer else None

n_forbidden = np.sum(
    ["[" in a for a in gas_names]
)  # forbidden lines contain "[*]"
n_balmer = len(gas_names) - n_forbidden

# Assign component=0 to the stellar templates, component=1 to
# the Balmer gas emission lines templates and component=2 to
# the forbidden lines.
component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
gas_component = np.array(component) > 0  # gas_component=True for gas templates

vorid = data["BINID"].data[0].astype("int")
vorid_list = np.sort(vorid[vorid >= 0])

for _id in vorid_list:
    # Get the vornoi pixel coordinate
    # Since the data are all repeating when they share the same vorid, we
    # are only getting the position from the first
    _pos = np.argwhere(vorid == _id)[0]
    _x, _y = _pos[0], _pos[1]
    if not isinstance(_x, (int, np.int64)):
        x = _x[0]
        y = _y[0]
    x = _x
    y = _y

    galaxy = data["FLUX"].data[:, x, y]
    galaxy_err = 1.0 / data["IVAR"].data[:, x, y]

    galaxy, galaxy_err = spectres(wave_new, wave, galaxy, np.sqrt(galaxy_err))
    non_nan_mask = (~np.isnan(galaxy) & ~np.isnan(galaxy_err))
    wave_new = wave_new[non_nan_mask]
    galaxy = galaxy[non_nan_mask]
    galaxy_err = galaxy_err[non_nan_mask]

    # natural log
    (
        galaxy_log_rebinned,
        wave_rebinned,
        velscale_rebinned,
    ) = util.log_rebin(wave_new, galaxy, velscale=velscale, flux=False)
    (
        noise,
        wave_rebinned,
        velscale_rebinned,
    ) = util.log_rebin(wave_new, galaxy_err, velscale=velscale, flux=False)

    goodpixels = np.arange(len(galaxy_log_rebinned))[
        np.isfinite(galaxy_log_rebinned) & ~np.isnan(galaxy_log_rebinned)
    ]

    weights = []

    vel = 0.0
    # (km/s), starting guess for [V, sigma]
    start = [[vel, 1.0, h3, h4], [vel, 1.0], [vel, 1.0]]

    # fixed = None
    fixed = [[0, 0, 0, 0], [0, 0], [0, 0]]

    # step (i) of Kacharov et al. 2018
    # Get the kinematics
    _pp1 = ppxf(
        templates,
        galaxy_log_rebinned,
        noise,
        velscale_rebinned,
        start,
        goodpixels=goodpixels,
        plot=False,
        moments=moments,
        degree=10,
        mdegree=10,
        fixed=fixed,
        clean=False,
        ftol=1e-8,
        lam=np.exp(wave_rebinned),
        lam_temp=miles.lam_temp,
        linear_method="lsq_box",
        bias=None,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
    )

    start = _pp1.sol
    fixed = [[1, 1, 1, 1], [1, 1], [1, 1]]

    # Step (ii) of Kacharov et al. 2018
    # Fit for the reddening value
    dust_gas = {"start": [0.1], "bounds": [[0, 8]], "component": gas_component}
    dust_stars = {
        "start": [0.1, -0.1],
        "bounds": [[0, 4], [-1, 10.0]],
        "component": ~gas_component,
    }
    dust = [dust_stars, dust_gas]
    _pp2 = ppxf(
        templates,
        galaxy_log_rebinned,
        noise,
        velscale_rebinned,
        start,
        goodpixels=goodpixels,
        plot=False,
        moments=moments,
        degree=-1,
        dust=dust,
        mdegree=0,
        fixed=fixed,
        clean=False,
        ftol=1e-8,
        lam=np.exp(wave_rebinned),
        lam_temp=miles.lam_temp,
        linear_method="lsq_box",
        regul=0,
        bias=None,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
    )
    print(_pp2.dust)

    Av, delta = _pp2.dust[0]["sol"]
    star_templates_reddened = np.vstack(
        [
            extinction.apply(extinction.calzetti00(miles.lam_temp, Av, 3.1), i)
            for i in templates.T[~gas_component]
        ]
    ).T
    templates_reddened = np.column_stack(
        [star_templates_reddened, gas_templates]
    )

    dust_gas = {"start": [0.0], "bounds": [[0, 8]], "fixed": [1], "component": gas_component}
    dust_stars = {"start": [0.0, 0.0], "bounds": [[0, 4], [-1, 0.4]], "fixed": [1, 1], "component": ~gas_component}
    dust = [dust_stars, dust_gas]
    # Step (iii) of Kacharov et al. 2018
    # Perform an unregularised fit, with a 10th order multiplicative polynomial
    _pp3 = ppxf(
        templates_reddened,
        galaxy_log_rebinned,
        noise,
        velscale_rebinned,
        start,
        goodpixels=goodpixels,
        plot=False,
        moments=moments,
        degree=-1,
        mdegree=10,
        fixed=fixed,
        clean=False,
        ftol=1e-8,
        lam=np.exp(wave_rebinned),
        lam_temp=miles.lam_temp,
        linear_method="lsq_box",
        regul=0,
        bias=None,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
    )

    # Rescale the noise so that reduced chi2 becomes 1
    noise_rescaled = noise * np.sqrt(_pp3.chi2)
    delta_chi2 = np.sqrt(2 * goodpixels.size)

    desired_chi2 = (goodpixels.size + delta_chi2) / goodpixels.size

    mpoly = _pp3.mpoly

    # get the mpoly at the template wavelength (linear)
    mpoly_linear_space = 10.0 ** itp.interp1d(
        np.exp(wave_rebinned), np.log10(mpoly), fill_value="extrapolate"
    )(miles.lam_temp)

    # Correct the templates with the multiplicative polynomial
    star_templates_corrected = (
        templates_reddened.T[~gas_component] * mpoly_linear_space
    )
    templates_corrected = np.column_stack(
        [star_templates_corrected.T, gas_templates]
    )

    _pp4 = ppxf(
        templates_corrected,
        galaxy_log_rebinned,
        noise_rescaled,
        velscale_rebinned,
        start,
        goodpixels=goodpixels,
        plot=False,
        moments=moments,
        degree=-1,
        mdegree=0,
        fixed=fixed,
        clean=False,
        ftol=1e-8,
        lam=np.exp(wave_rebinned),
        lam_temp=miles.lam_temp,
        linear_method="lsq_box",
        regul=0,
        bias=None,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
    )

    noise_rescaled_2 = noise_rescaled * np.sqrt(_pp4.chi2)

    # Step (iv) of Kacharov et al. 2018
    results = minimize(
        find_reg,
        np.log10(100.0),
        args=(
            templates_corrected,
            galaxy_log_rebinned,
            noise_rescaled_2,
            velscale_rebinned,
            start,
            goodpixels,
            False,
            moments,
            -1,
            0,
            fixed,
            False,
            1e-8,
            np.exp(wave_rebinned),
            miles.lam_temp,
            "lsq_box",
            1,
            None,
            reg_dim,
            component,
            gas_component,
            gas_names,
            desired_chi2,
        ),
        tol=1e-8,
        method="Powell",
        options={"ftol": 1e-8, "xtol": 1e-8},
    )
    best_reg = 10.0 ** results.x[0]

    # Get the residuals for resampling for noise estimation
    pp = ppxf(
        templates_corrected,
        galaxy_log_rebinned,
        noise_rescaled_2,
        velscale_rebinned,
        start,
        goodpixels=goodpixels,
        plot=False,
        moments=moments,
        degree=-1,
        mdegree=0,
        fixed=fixed,
        clean=False,
        ftol=1e-8,
        lam=np.exp(wave_rebinned),
        lam_temp=miles.lam_temp,
        linear_method="lsq_box",
        regul=best_reg,
        reg_ord=1,
        bias=None,
        reg_dim=reg_dim,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
    )
    print(f"Best fit reduced-chi2: {pp.chi2}")

    # Resampling 100 times using the residuals
    (
        galaxy_bootstrap,
        light_weight_bootstrap,
        mass_weight_bootstrap,
        sfr_light_bootstrap,
        sfr_mass_bootstrap,
    ) = get_uncertainty(
        templates_corrected,
        galaxy_log_rebinned,
        noise_rescaled_2,
        velscale_rebinned,
        start,
        goodpixels=goodpixels,
        plot=False,
        moments=moments,
        degree=-1,
        mdegree=0,
        fixed=fixed,
        clean=False,
        ftol=1e-8,
        lam=np.exp(wave_rebinned),
        lam_temp=miles.lam_temp,
        linear_method="lsq_box",
        regul=best_reg,
        reg_ord=1,
        bias=None,
        reg_dim=reg_dim,
        component=component,
        gas_component=gas_component,
        gas_names=gas_names,
        residual=np.abs(galaxy_log_rebinned - _pp1.bestfit),
    )

    light_weights = pp.weights[~gas_component]  # Exclude gas templates weights
    light_weights = light_weights.reshape(reg_dim)  # Reshape to a 2D matrix
    light_weights /= np.nanmedian(light_weights)

    # convert from light to mass, hence 1./Mass-to-light
    mass_weights = light_weights / miles.flux
    mass_weights[np.isnan(mass_weights)] = 0.0

    sfr_light = np.sum(light_weights, axis=1)
    sfr_light[np.isnan(sfr_light)] = 0.0

    sfr_mass = np.sum(mass_weights, axis=1)
    sfr_mass[np.isnan(sfr_mass)] = 0.0

    age_grid = miles.age_grid[:, 0]
    metal_grid = miles.metal_grid.T[:, 0]

    # Create plot here
    gs = gridspec.GridSpec(4, 1, height_ratios=[4, 1, 4, 4])

    fig = plt.figure(1, figsize=(10, 12))
    plt.clf()
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ax1.plot(np.exp(wave_rebinned), galaxy_log_rebinned, label="Input")
    ax1.fill_between(
        np.exp(wave_rebinned),
        galaxy_log_rebinned - noise,
        galaxy_log_rebinned + noise,
        alpha=0.5,
        color="grey",
        zorder=2,
        label="Error Range",
    )
    ax1.plot(
        np.exp(wave_rebinned),
        pp.bestfit,
        color="black",
        label="Fitted",
    )
    ax1.scatter(
        np.exp(wave_rebinned),
        galaxy_log_rebinned - pp.bestfit,
        color="green",
        s=2,
        label="Residual",
    )
    ax1.grid()
    ax1.set_xlim(min(np.exp(wave_rebinned)), max(np.exp(wave_rebinned)))
    ax1.set_xlabel("Wavelength / A")
    ax1.set_ylabel("Relative Flux")
    ax1.legend()

    im = NonUniformImage(
        ax3,
        interpolation="nearest",
        extent=(
            (1.5 * age_grid[0] - 0.5 * age_grid[1]),
            (1.5 * age_grid[-1] - 0.5 * age_grid[-2]),
            metal_grid[0] - 0.5,
            metal_grid[0] + 0.5,
        ),
    )
    x_ax3 = age_grid
    y_ax3 = metal_grid
    im.set_data(x_ax3, y_ax3, light_weights.T)
    ax3.images.append(im)
    ax3.set_xticklabels([""])
    ax3.set_ylabel("Metallicity")
    ax3.set_xlim(
        (1.5 * age_grid[0] - 0.5 * age_grid[1]),
        (1.5 * age_grid[-1] - 0.5 * age_grid[-2]),
    )
    ax3.set_yticks(metal_grid)
    ax3.set_yticklabels(metal_grid)
    ax3.set_ylim(
        metal_grid[0] - 0.5,
        metal_grid[0] + 0.5,
    )

    # Plot the firefly solution here
    #
    #
    #
    #
    #

    ax4.plot(
        age_grid,
        sfr_light / np.nanmax(sfr_light),
        label="Recovered (light-weighted)",
    )
    for _sfr in sfr_mass_bootstrap:
        ax4.plot(
            age_grid,
            _sfr / np.nanmax(_sfr),
            color="grey",
            alpha=0.1,
            zorder=15,
        )
    sfr_median = np.nanmedian(sfr_mass_bootstrap, axis=0)
    sfr_mass_err = (
        np.nanmedian(np.abs(sfr_mass_bootstrap - sfr_median), axis=0) * 1.4826
    )
    ax4.errorbar(
        age_grid,
        sfr_mass / np.nanmax(sfr_mass),
        yerr=sfr_mass_err / np.nanmax(sfr_mass),
        label="Recovered (mass-weighted)",
        capsize=5,
        elinewidth=2,
    )
    ax4.set_xlim(np.nanmin(age_grid), np.nanmax(age_grid))
    ax4.grid()
    ax4.set_ylabel("Relative Star Formation Rate")
    ax4.set_xlabel("Age / Gyr")
    ax4.legend()

    plt.subplots_adjust(
        top=0.975, bottom=0.05, left=0.08, right=0.95, hspace=0
    )
    plt.ion()
    plt.show()

    plt.savefig(
        "manga-9881/manga-9881-9102-{}-optimal-regul-{}.png".format(
            _id, float(best_reg)
        )
    )


""" # get firefly SFH here
HDU 1: FLUX
HDU 12: BINID
>>> np.shape(data[1].data)
(4563, 54, 54)
# 4563 is the length of the spectrum

>>> np.shape(data_firefly['SPATIAL_INFO'].data) 
(4675, 2800, 4)

>>> np.shape(data_firefly['SPATIAL_BINID'].data)
(4675, 76, 76)

>>> np.shape(firefly_sfh) 
(2800, 8, 3)


# These 2 plots SHOULD show the identical voronoi zones

figure(1)
imshow(np.log10(data[1].data[300]))

_w, _h = np.shape(data[1].data[300])
figure(2)
imshow(data_firefly['SPATIAL_BINID'].data[firefly_mask][:_w, :_h], vmin=0)

"""
