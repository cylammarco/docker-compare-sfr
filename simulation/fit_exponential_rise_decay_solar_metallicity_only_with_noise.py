from astropy import constants
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage
import numpy as np
import os
import ppxf as ppxf_package
import ppxf.miles_util as lib
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from scipy.optimize import minimize
from spectres import spectres

from numpy.polynomial import legendre


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
        quiet=True,
    )
    chi2_diff = abs(pp.chi2 - desired_chi2)
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
            residual,
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
        )
        light_weights = pp.weights  # Exclude gas templates weights
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


ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
miles_pathname = os.path.join(ppxf_dir, "miles_models", "Eun1.30Z*.fits")


# speed of light
c = constants.c.to("km/s").value

FWHM_gal = (
    2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
)
wave_new = np.arange(3600.0, 7500.0, FWHM_gal)
velscale = c * np.median(np.diff(wave_new)) / wave_new[-1]

filename = os.path.join(
    "output",
    "exponential_rise_decay_with_noise",
    "spectrum_1.0_gyr_2.0_rise_decay_with_noise_10.npy",
)
wave, _galaxy, error = np.load(filename, allow_pickle=True).T

velscale = c * np.log(wave_new[1] / wave_new[0])

# Estimate the wavelength fitted range in the rest frame.
lam_range_gal = np.array([np.min(wave), np.max(wave)])

### for setting up the MILES
_galaxy = spectres(wave_new, wave, _galaxy)

# natural log
(_, _, velscale_rebinned,) = util.log_rebin(
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
    wave_range=(3000, 8500),
)
reg_dim = miles.templates.shape[1:]
stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
n_temps = stars_templates.shape[1]

# Combines the stellar and gaseous templates into a single array.
# During the PPXF fit they will be assigned a different kinematic
# COMPONENT value
#
templates = stars_templates


# Fit (V, sig, h3, h4) moments=4 for the stars
# and (V, sig) moments=2 for the two gas kinematic components
moments = [4]

h3 = 0.0  # Adopted G-H parameters of the LOSVD
h4 = 0.0

# Assign component=0 to the stellar templates, component=1 to
# the Balmer gas emission lines templates and component=2 to
# the forbidden lines.
component = [0] * n_temps

tau = 2.0

for sn in ["10", "20", "50"]:
    for age in [0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 11.0]:
        filename = os.path.join(
            "output",
            "exponential_rise_decay_with_noise",
            f"spectrum_{age:.1f}_gyr_{tau:.1f}_rise_decay_with_noise_{sn}.npy",
        )

        wave, galaxy, galaxy_err = np.load(filename, allow_pickle=True).T
        galaxy, galaxy_err = spectres(wave_new, wave, galaxy, galaxy_err)

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
        goodpixels = np.arange(len(galaxy_log_rebinned))[1:]

        start = [100.0, 10.0, h3, h4]
        fixed = [0, 0, 1, 1]

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
            lam_temp=np.exp(miles.ln_lam_temp),
            linear_method="lsq_box",
            bias=None,
            component=component,
        )

        start = _pp1.sol
        fixed = [1, 1, 1, 1]

        # Step (iii) of Kacharov et al. 2018
        # Perform an unregularised fit, with a 10th order multiplicative polynomial
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
            mdegree=10,
            fixed=fixed,
            clean=False,
            ftol=1e-8,
            lam=np.exp(wave_rebinned),
            lam_temp=np.exp(miles.ln_lam_temp),
            linear_method="lsq_box",
            regul=0,
            bias=None,
            component=component,
        )

        # Rescale the noise so that reduced chi2 becomes 1
        noise_rescaled = noise * np.sqrt(_pp2.chi2)
        delta_chi2 = np.sqrt(2 * goodpixels.size)

        desired_chi2 = (goodpixels.size + delta_chi2) / goodpixels.size

        x = np.linspace(-1, 1, len(miles.ln_lam_temp))
        mpoly = legendre.legval(x, np.append(1, _pp2.mpolyweights))

        # Correct the templates with the multiplicative polynomial
        stars_templates_corrected = stars_templates.T * mpoly
        templates_corrected = stars_templates_corrected.T

        _pp3 = ppxf(
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
            lam_temp=np.exp(miles.ln_lam_temp),
            linear_method="lsq_box",
            regul=0,
            bias=None,
            component=component,
        )

        noise_rescaled_2 = noise_rescaled * np.sqrt(_pp3.chi2)

        # Step (iv) of Kacharov et al. 2018
        results = minimize(
            find_reg,
            np.log10(1e-2),
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
                np.exp(miles.ln_lam_temp),
                "lsq_box",
                1,
                None,
                reg_dim,
                component,
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
            ftol=1e-7,
            lam=np.exp(wave_rebinned),
            lam_temp=np.exp(miles.ln_lam_temp),
            linear_method="lsq_box",
            regul=best_reg,
            reg_ord=1,
            bias=None,
            reg_dim=reg_dim,
            component=component,
        )

        # Resampling 100 times using the residuals
        (
            galaxy_bootstrap,
            light_weight_bootstrap,
            mass_weight_bootstrap,
            sfr_light_bootstrap,
            sfr_mass_bootstrap,
        ) = get_uncertainty(
            templates_corrected,
            pp.bestfit,
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
            lam_temp=np.exp(miles.ln_lam_temp),
            linear_method="lsq_box",
            regul=best_reg,
            reg_ord=1,
            bias=None,
            reg_dim=reg_dim,
            component=component,
            residual=np.abs(galaxy_log_rebinned - pp.bestfit),
        )

        light_weights = pp.weights  # Exclude gas templates weights
        light_weights = light_weights.reshape(
            reg_dim
        )  # Reshape to a 2D matrix

        # convert from light to mass, hence 1./Mass-to-light
        mass_weights = light_weights / miles.flux
        mass_weights[np.isnan(mass_weights)] = 0.0

        sfr_light = np.sum(light_weights, axis=1)
        sfr_light[np.isnan(sfr_light)] = 0.0

        sfr_mass = np.sum(mass_weights, axis=1)
        sfr_mass[np.isnan(sfr_mass)] = 0.0

        age_grid = miles.age_grid[:, 0]
        metal_grid = miles.metal_grid.T[:, 0]

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

        t_lookback = 10.0 ** np.arange(5.0, 10.2, 0.001)
        sfh = np.exp(-np.abs(age * 1.0e9 - t_lookback) / (tau * 1.0e9))
        sfh /= max(sfh)

        ax4.plot(t_lookback / 1e9, sfh, label="Input")

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
            np.nanmedian(np.abs(sfr_mass_bootstrap - sfr_median), axis=0)
            * 1.4826
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
            os.path.join(
                "output",
                "exponential_rise_decay_with_noise",
                f"spectrum_{age:.1f}_gyr_{tau:.1f}_rise_decay_with_noise_{sn}_fitted.png",
            )
        )
