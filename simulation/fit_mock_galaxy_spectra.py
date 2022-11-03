import os
import itertools

import sphinx

from astropy import constants
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import ppxf as ppxf_package
from ppxf.ppxf import ppxf, rebin
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from spectres import spectres
from scipy import ndimage, signal
from scipy.optimize import minimize

ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
miles_pathname = ppxf_dir + os.sep + "miles_models" + os.sep + "Eun1.30Z*.fits"

synthetic_spectrum_dir = os.path.join("output", "spectrum")

file_vega = (
    ppxf_dir + "/miles_models/Vazdekis2012_ssp_phot_Padova00_UN_v10.0.txt"
)
file_sdss = (
    ppxf_dir + "/miles_models/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
)
file1 = (
    ppxf_dir
    + "/miles_models/Vazdekis2012_ssp_mass_Padova00_UN_baseFe_v10.0.txt"
)

vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
sdss_bands = ["u", "g", "r", "i"]
vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
sdss_sun_mag = [
    6.55,
    5.12,
    4.68,
    4.57,
]  # values provided by Elena Ricciardelli

slope1, MH1, Age1, m_no_gas = np.loadtxt(file1, usecols=[1, 2, 3, 5]).T
(
    slope2_vega,
    MH2_vega,
    Age2_vega,
    m_U,
    m_B,
    m_V,
    m_R,
    m_I,
    m_J,
    m_H,
    m_K,
) = np.loadtxt(file_vega, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).T
slope2_sdss, MH2_sdss, Age2_sdss, m_u, m_g, m_r, m_i = np.loadtxt(
    file_sdss, usecols=[1, 2, 3, 4, 5, 6, 7]
).T

slope2 = {"vega": slope2_vega, "sdss": slope2_sdss}
MH2 = {"vega": MH2_vega, "sdss": MH2_sdss}
Age2 = {"vega": Age2_vega, "sdss": Age2_sdss}

m_vega = {
    "U": m_U,
    "B": m_B,
    "V": m_V,
    "R": m_R,
    "I": m_I,
    "J": m_J,
    "H": m_H,
    "K": m_K,
}
m_sdss = {"u": m_u, "g": m_g, "r": m_r, "i": m_i}


def mass_to_light(weights, band="r", quiet=False):
    """
    Computes the M/L in a chosen band, given the weights produced
    in output by pPXF. A Salpeter IMF is assumed (slope=1.3).
    The returned M/L includes living stars and stellar remnants,
    but excludes the gas lost during stellar evolution.
    Parameters
    ----------
    weights:
        pPXF output with dimensions weights[miles.n_ages, miles.n_metal]
    band:
        possible choices are "U", "B", "V", "R", "I", "J", "H", "K" for
        the Vega photometric system and "u", "g", "r", "i" for the SDSS
        AB system.
    quiet:
        set to True to suppress the printed output.
    Returns
    -------
    mass_to_light (float) in the given band
    """
    assert (
        miles.age_grid.shape == miles.metal_grid.shape == weights.shape
    ), "Input weight dimensions do not match"
    if band in vega_bands:
        k = vega_bands.index(band)
        sun_mag = vega_sun_mag[k]
        mag = m_vega[band]
        _Age2 = Age2["vega"]
        _slope2 = slope2["vega"]
        _MH2 = MH2["vega"]
    elif band in sdss_bands:
        k = sdss_bands.index(band)
        sun_mag = sdss_sun_mag[k]
        mag = m_sdss[band]
        _Age2 = Age2["sdss"]
        _slope2 = slope2["sdss"]
        _MH2 = MH2["sdss"]
    else:
        raise ValueError("Unsupported photometric band")
    # The following loop is a brute force, but very safe and general,
    # way of matching the photometric quantities to the SSP spectra.
    # It makes no assumption on the sorting and dimensions of the files
    mass_no_gas_grid = np.empty_like(weights)
    lum_grid = np.empty_like(weights)
    for j in range(miles.n_ages):
        for k in range(miles.n_metal):
            p1 = (
                (np.abs(miles.age_grid[j, k] - Age1) < 0.001)
                & (np.abs(miles.metal_grid[j, k] - MH1) < 0.01)
                & (np.abs(1.30 - slope1) < 0.01)
            )
            mass_no_gas_grid[j, k] = m_no_gas[p1]
            p2 = (
                (np.abs(miles.age_grid[j, k] - _Age2) < 0.001)
                & (np.abs(miles.metal_grid[j, k] - _MH2) < 0.01)
                & (np.abs(1.30 - _slope2) < 0.01)
            )
            lum_grid[j, k] = 10 ** (-0.4 * (mag[p2] - sun_mag))
    # This is eq.(2) in Cappellari+13
    # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
    mlpop = np.sum(weights * mass_no_gas_grid, axis=1) / np.sum(
        weights * lum_grid, axis=1
    )
    if not quiet:
        print("Summed M/L in passband " + band + ": %#.4g" % np.nansum(mlpop))
    return mlpop


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
    clean,
    ftol,
    fixed,
    lam,
    lam_temp,
    linear_method,
    bias,
    reg_dim,
    component,
    desired_chi2,
):
    reg_guess = reg_guess[0]
    if reg_guess < 0.0:
        return np.inf
    '''
    print("Current regularisation value is: {}.".format(reg_guess))
    '''
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
        clean=clean,
        ftol=ftol,
        fixed=fixed,
        lam=lam,
        lam_temp=lam_temp,
        linear_method=linear_method,
        regul=reg_guess,
        bias=bias,
        reg_dim=reg_dim,
        component=component,
        quiet=True,
    )
    chi2_diff = pp.chi2 * pp.dof - desired_chi2
    '''
    print("")
    print(
        "================================================================================"
    )
    print("Current Chi^2: {}".format(pp.chi2 * pp.dof))
    print("Desired Chi^2: {}".format(desired_chi2))
    print("Current distance from the desired Chi^2 is: {}.".format(chi2_diff))
    print(
        "================================================================================"
    )
    print("")
    '''
    return chi2_diff**2.0


# speed of light
c = constants.c.to("km/s").value
# redshift is zero
z = 0.0


# Fit (V, sig, h3, h4) moments=4 for the stars
# and (V, sig) moments=2 for the two gas kinematic components
moments = 4

factor = 10  # Oversampling integer factor for an accurate convolution
h3 = 0.01  # Adopted G-H parameters of the LOSVD
h4 = 0.01
sn = 30.0  # Adopted S/N of the Monte Carlo simulation
m = 100  # Number of realizations of the simulation

# Data are at z=0.025, but the spectra are not redshifted in the simulation
z = 0

FWHM_gal = (
    2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
)
wave_new = np.arange(3650.0, 10000.0, FWHM_gal)
velscale = c * np.median(np.diff(wave_new)) / wave_new[-1]


n_rings = 5
n_spexels = np.arange(n_rings) * 6
n_spexels[0] = 1

for i in range(100):

    galaxy_data_cube = np.load(
        os.path.join("output", "spectrum", "galaxy_{}.npy".format(i))
    )
    wave = galaxy_data_cube[0]
    input_sfh_cube = np.load(
        os.path.join("output", "sfh", "galaxy_sfh_{}.npy".format(i))
    )
    input_age = input_sfh_cube[0]

    for j, spexels in enumerate(n_spexels):

        if j == 0:
            spexels_skip = 0
        else:
            spexels_skip = np.sum(n_spexels[:j])

        input_sfh = input_sfh_cube[j + 1]

        for spx in range(spexels):

            galaxy = galaxy_data_cube[1 + spexels_skip + spx]
            galaxy = spectres(wave_new, wave, galaxy)
            velscale = c * np.log(wave_new[1] / wave_new[0])

            # natural log
            (
                galaxy_log_rebinned,
                wave_rebinned,
                velscale_rebinned,
            ) = util.log_rebin(
                [np.nanmin(wave_new), np.nanmax(wave_new)],
                galaxy,
                velscale=velscale,
            )

            noise = galaxy_log_rebinned / sn

            weights = []

            miles = lib.miles(
                miles_pathname,
                velscale_rebinned,
                FWHM_gal,
                age_range=None,
                metal_range=[-0.05, 0.05],
                wave_range=(3500, 10500),
            )
            reg_dim = miles.templates.shape[1:]
            templates = miles.templates.reshape(miles.templates.shape[0], -1)
            n_temps = templates.shape[1]

            # Assign component=0 to the stellar templates, component=1 to
            # the Balmer gas emission lines templates and component=2 to
            # the forbidden lines.
            component = [0] * n_temps

            goodpixels = np.arange(len(galaxy_log_rebinned))

            vel = c * np.log(1 + z)
            # (km/s), starting guess for [V, sigma]
            start = [vel, 1.0, h3, h4]
            degree = 8
            mdegree = 0
            # fixed = None
            fixed = [0, 0, 0, 0]

            # Get the UNREGULARISED fit here
            _pp = ppxf(
                templates,
                galaxy_log_rebinned,
                noise,
                velscale_rebinned,
                start,
                goodpixels=goodpixels,
                plot=False,
                moments=moments,
                degree=degree,
                mdegree=mdegree,
                clean=False,
                ftol=1e-4,
                fixed=fixed,
                lam=np.exp(wave_rebinned),
                lam_temp=np.exp(miles.ln_lam_temp),
                linear_method="lsq_box",
                regul=0.0,
                bias=0.0,
                reg_dim=reg_dim,
                component=component,
                quiet=True,
            )

            scaling_factor = np.sqrt(_pp.chi2)

            unregularised_chi2 = _pp.goodpixels.size
            delta_chi2 = np.sqrt(2 * _pp.goodpixels.size)
            desired_chi2 = unregularised_chi2 + delta_chi2

            print("")
            print(
                "================================================================================"
            )
            print("Galaxy {} Spexel {}".format(i, 1 + spexels_skip + spx))
            print("Unregularised Chi^2: {}".format(unregularised_chi2))
            print("Desired delta(Chi^2) = {}".format(delta_chi2))
            print("Desired Chi^2: {}".format(desired_chi2))
            print(
                "================================================================================"
            )
            print("")

            results = minimize(
                find_reg,
                3e-3,
                args=(
                    templates,
                    galaxy_log_rebinned,
                    noise * scaling_factor,
                    velscale_rebinned,
                    start,
                    goodpixels,
                    False,
                    moments,
                    degree,
                    mdegree,
                    False,
                    1e-4,
                    fixed,
                    np.exp(wave_rebinned),
                    np.exp(miles.ln_lam_temp),
                    "lsq_box",
                    None,
                    reg_dim,
                    component,
                    desired_chi2,
                ),
                tol=5e-1,
                method="Powell",
                options={"ftol": 5e-1, "xtol": 5e-1},
            )
            best_reg = results.x

            pp = ppxf(
                templates,
                galaxy_log_rebinned,
                noise * scaling_factor,
                velscale_rebinned,
                start,
                plot=False,
                moments=moments,
                degree=degree,
                mdegree=mdegree,
                clean=False,
                ftol=1e-4,
                fixed=fixed,
                lam=np.exp(wave_rebinned),
                lam_temp=np.exp(miles.ln_lam_temp),
                linear_method="lsq_box",
                regul=best_reg[0],
                bias=0.0,
                reg_dim=reg_dim,
                component=component,
            )
            print("Current Chi^2: %#.6g" % (pp.chi2 * pp.dof))

            weights = pp.weights
            weights = weights.reshape(reg_dim) / weights.sum()  # Normalized
            sfr = np.sum(weights, axis=1)
            age = miles.age_grid[:, 0]

            ml_r = mass_to_light(weights, band="r")
            ml_V = mass_to_light(weights, band="V")

            sfr_r = sfr / ml_r
            sfr_V = sfr / ml_V

            sfr_r[np.isnan(sfr_r)] = 0.0
            sfr_V[np.isnan(sfr_V)] = 0.0

            gs = gridspec.GridSpec(4, 1, height_ratios=[4, 1, 4, 4])

            fig = plt.figure(1, figsize=(10, 12))
            plt.clf()
            ax1 = plt.subplot(gs[0])
            ax3 = plt.subplot(gs[2])
            ax4 = plt.subplot(gs[3])

            ax1.plot(np.exp(wave_rebinned), galaxy_log_rebinned, label="Input")
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
            ax1.set_xlim(
                min(np.exp(wave_rebinned)), max(np.exp(wave_rebinned))
            )
            ax1.set_xlabel("Wavelength / A")
            ax1.set_ylabel("Relative Flux")
            ax1.set_title(
                "Galaxy {} Spexel {}: Regularisation = {}".format(
                    i, 1 + spexels_skip + spx, best_reg[0]
                )
            )
            ax1.legend()

            ax3.imshow(
                weights.T,
                origin="lower",
                extent=[min(age), max(age), -0.2, 0.11],
                aspect="auto",
            )
            ax3.set_xticklabels([""])
            ax3.set_ylabel("Metallicity")
            ax3b = ax4.twinx()
            ax3b.set_ylabel("Recovered")
            ax3b.set_yticklabels([""])

            ax4.plot(
                age, sfr / np.nanmax(sfr), label="Recovered (light-weighted)"
            )
            ax4.plot(
                age,
                sfr_r / np.nanmax(sfr_r),
                label="Recovered (mass-weighted, r)",
            )
            ax4.plot(
                age,
                sfr_V / np.nanmax(sfr_V),
                label="Recovered (mass-weighted, V)",
            )
            ax4.plot(
                (10.0**input_age) / 1e9,
                input_sfh / np.nanmax(input_sfh),
                label="Input (mass-weighted)",
                color="black",
            )
            ax4.set_xlim(np.nanmin(age), np.nanmax(age))
            ax4.grid()
            ax4.set_xscale("log")
            ax4.set_ylabel("Relative Star Formation Rate")
            ax4.set_xlabel("Age / Gyr")
            ax4.legend()

            plt.subplots_adjust(
                top=0.975, bottom=0.05, left=0.08, right=0.95, hspace=0
            )
            plt.savefig(
                os.path.join(
                    "output",
                    "fit",
                    "galaxy_{}_spexel_{}.png".format(
                        i, 1 + spexels_skip + spx
                    ),
                )
            )
