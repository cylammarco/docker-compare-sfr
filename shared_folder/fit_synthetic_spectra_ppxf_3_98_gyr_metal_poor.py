import os
import itertools

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
miles_pathname = ppxf_dir + os.sep + "miles_models" + os.sep + "Mun1.30*.fits"

synthetic_spectrum_dir = ".." + os.sep + "synthetic_spectra"

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


def find_ref(
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
    vsyst,
    ftol,
    fixed,
    lam,
    linear_method,
    reg_dim,
    component,
):

    print("Current regularisation value is: {}.".format(reg_guess))

    if reg_guess <= 0.0:

        return np.inf

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
        vsyst=vsyst,
        ftol=ftol,
        fixed=fixed,
        lam=lam,
        linear_method=linear_method,
        regul=reg_guess,
        reg_dim=reg_dim,
        component=component,
    )

    return 10.0 ** abs(pp.chi2 - 1.0)


gs = gridspec.GridSpec(5, 1, height_ratios=[4, 1, 4, 4, 4])

# speed of light
c = constants.c.to("km/s").value
# redshift is zero
z = 0.0
# MILES has an approximate FWHM resolution of 2.51A.
FWHM_gal = 2.51

wave_new = np.arange(3620.0, 7400.0, 2.0)

velscale = c * np.median(np.diff(wave_new)) / wave_new[-1]

# Fit (V, sig, h3, h4) moments=4 for the stars
# and (V, sig) moments=2 for the two gas kinematic components
moments = 4

if not os.path.exists(".." + os.sep + "synthetic_spectra" + os.sep + "sfr"):
    os.mkdir(".." + os.sep + "synthetic_spectra" + os.sep + "sfr")

if not os.path.exists(
    ".." + os.sep + "synthetic_spectra" + os.sep + "fitted_model"
):
    os.mkdir(".." + os.sep + "synthetic_spectra" + os.sep + "fitted_model")

if not os.path.exists(
    ".." + os.sep + "synthetic_spectra" + os.sep + "age_metallicity"
):
    os.mkdir(".." + os.sep + "synthetic_spectra" + os.sep + "age_metallicity")

if not os.path.exists(".." + os.sep + "synthetic_spectra" + os.sep + "figure"):
    os.mkdir(".." + os.sep + "synthetic_spectra" + os.sep + "figure")

factor = 10  # Oversampling integer factor for an accurate convolution
h3 = 0.1  # Adopted G-H parameters of the LOSVD
h4 = 0.1
sn = 30.0  # Adopted S/N of the Monte Carlo simulation
m = 100  # Number of realizations of the simulation

velV = np.random.rand(m)  # velocity in *pixels* [=V(km/s)/velScale]
sigmaV = np.linspace(
    0.5, 4, m
)  # Range of sigma in *pixels* [=sigma(km/s)/velScale]

fixed = [0, 0, 0, 0]

sf_type = "ed05"
z = -0.71
t = 10.0**0.6

for degree, mdegree in list(itertools.product(np.arange(10), np.arange(10))):

    # Star burst
    wave, galaxy = np.load(
        ".."
        + os.sep
        + "synthetic_spectra"
        + os.sep
        + "sp_{0}_z{1:1.2f}_t{2:2.2f}.npy".format(sf_type, z, t)
    ).T
    galaxy = spectres(wave_new, wave, galaxy)

    # natural log
    galaxy_log_rebinned, wave_rebinned, velscale_rebinned = util.log_rebin(
        [np.nanmin(wave_new), np.nanmax(wave_new)], galaxy, velscale=velscale
    )

    galaxy_log_rebinned = ndimage.interpolation.zoom(
        galaxy_log_rebinned, factor, order=3
    )
    weights = []
    for j, (vel, sigma) in enumerate(zip(velV, sigmaV)):

        # (km/s), starting guess for [V, sigma]
        start = (
            np.array(
                [
                    vel + np.random.uniform(-1, 1),
                    sigma * np.random.uniform(0.8, 1.2),
                ]
            )
            * velscale
        )

        dx = int(abs(vel) + 5 * sigma)
        x = np.linspace(
            -dx, dx, 2 * dx * factor + 1
        )  # Evaluate the Gaussian using steps of 1/factor pixels.
        w = (x - vel) / sigma
        w2 = w**2
        gauss = np.exp(-0.5 * w2)
        gauss /= np.sum(gauss)  # Normalized total(gauss)=1
        h3poly = w * (2.0 * w2 - 3.0) / np.sqrt(3.0)  # H3(y)
        h4poly = (w2 * (4.0 * w2 - 12.0) + 3.0) / np.sqrt(24.0)  # H4(y)
        losvd = gauss * (1.0 + h3 * h3poly + h4 * h4poly)

        galaxy_rebinned = signal.fftconvolve(
            galaxy_log_rebinned, losvd, mode="same"
        )  # Convolve the oversampled spectrum
        galaxy_rebinned = rebin(
            galaxy_rebinned, factor
        )  # Integrate spectrum into original spectral pixels

        goodpixels = np.arange(dx, galaxy_rebinned.size - dx)

        print("Loading file from " + miles_pathname)
        miles = lib.miles(miles_pathname, velscale_rebinned, FWHM_gal)

        reg_dim = miles.templates.shape[1:]
        templates = miles.templates.reshape(miles.templates.shape[0], -1)

        noise = galaxy_rebinned / sn  # 1sigma error spectrum
        galaxy_rebinned = np.random.normal(
            galaxy_rebinned, noise
        )  # Add noise to the galaxy spectrum

        n_temps = templates.shape[1]

        # Assign component=0 to the stellar templates, component=1 to
        # the Balmer gas emission lines templates and component=2 to
        # the forbidden lines.
        component = [0] * n_temps

        # eq.(8) of Cappellari (2017)
        dv = c * (miles.log_lam_temp[0] - wave_rebinned[0])

        pp = ppxf(
            templates,
            galaxy_rebinned,
            noise,
            velscale_rebinned,
            start,
            goodpixels=goodpixels,
            plot=False,
            moments=moments,
            degree=degree,
            mdegree=mdegree,
            clean=False,
            vsyst=dv,
            ftol=1e-4,
            fixed=fixed,
            lam=np.exp(wave_rebinned),
            linear_method="lsq_box",
            regul=1 / np.nanmedian(noise),
            bias=0.2,
            reg_dim=reg_dim,
            component=component,
        )

        print("Current Chi^2: %#.6g" % pp.chi2)

        _weights = pp.weights
        weights.append(
            _weights.reshape(reg_dim) / _weights.sum()
        )  # Normalized

    weights = np.mean(weights, axis=0)
    sfr = np.sum(weights, axis=1)
    age = miles.age_grid[:, 0]
    metallicity = miles.metal_grid[0]

    if sf_type == "sb00":
        sfr_input = np.zeros_like(age)
        sfr_input[np.argmin(np.abs(age - t))] = 1.0
    else:
        age_input = 10.0 ** np.linspace(
            np.log10(min(age)), np.log10(max(age)), 1000
        )
        sfr_input = np.zeros_like(age_input)
        sfr_input[age_input <= t] = np.exp(
            (age_input[age_input <= t] - t) / 0.5
        )
        sfr_input = sfr_input / np.nanmax(sfr_input) * np.nanmax(sfr)
        sfr_input = spectres(age, age_input, sfr_input, fill=0.0)

    ml_r = mass_to_light(weights, band="r")
    ml_V = mass_to_light(weights, band="V")

    sfr_r = sfr / ml_r
    sfr_V = sfr / ml_V

    input_weights = np.zeros_like(weights.T)
    input_weights[np.argmin(np.abs(metallicity - z)), :] = sfr_input

    fig = plt.figure(1, figsize=(10, 12))
    plt.clf()
    ax1 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    ax5 = plt.subplot(gs[4])

    ax1.plot(np.exp(wave_rebinned), galaxy_rebinned, label="Input")
    ax1.plot(np.exp(wave_rebinned), pp.bestfit, color="black", label="Fitted")
    ax1.scatter(
        np.exp(wave_rebinned),
        galaxy_rebinned - pp.bestfit,
        color="green",
        s=2,
        label="Residual",
    )
    ax1.grid()
    ax1.set_xlim(min(np.exp(wave_rebinned)), max(np.exp(wave_rebinned)))
    ax1.set_xlabel("Wavelength / A")
    ax1.set_ylabel("Relative Flux")
    ax1.legend()

    ax3.imshow(
        input_weights,
        origin="lower",
        extent=[min(age), max(age), min(metallicity), max(metallicity)],
        aspect="auto",
    )
    ax3.set_xticklabels([""])
    ax3.set_ylabel("Metallicity")
    ax3b = ax3.twinx()
    ax3b.set_ylabel("Input")
    ax3b.set_yticklabels([""])

    ax4.imshow(
        weights.T,
        origin="lower",
        extent=[min(age), max(age), min(metallicity), max(metallicity)],
        aspect="auto",
    )
    ax4.set_xticklabels([""])
    ax4.set_ylabel("Metallicity")
    ax4b = ax4.twinx()
    ax4b.set_ylabel("Recovered")
    ax4b.set_yticklabels([""])

    ax5.plot(
        age, sfr_input / np.nanmax(sfr_input), label="Input", color="black"
    )
    ax5.plot(age, sfr / np.nanmax(sfr), label="Recovered (light-weighted)")
    ax5.plot(
        age, sfr_r / np.nanmax(sfr_r), label="Recovered (mass-weighted, r)"
    )
    ax5.plot(
        age, sfr_V / np.nanmax(sfr_V), label="Recovered (mass-weighted, V)"
    )
    ax5.set_xlim(np.nanmin(age), np.nanmax(age))
    ax5.grid()
    ax5.set_xscale("log")
    ax5.set_ylabel("Relative Star Formation Rate")
    ax5.set_xlabel("Age / Gyr")
    ax5.legend()

    plt.subplots_adjust(
        top=0.975, bottom=0.05, left=0.08, right=0.95, hspace=0
    )

    plt.savefig(
        os.path.join(
            "..",
            "synthetic_spectra",
            "figure",
            "sp_{0}_z{1:1.2f}_t{2:2.2f}_degree_{3}_mdegree_{4}.png".format(
                sf_type, z, t, degree, mdegree
            ),
        )
    )
    np.save(
        os.path.join(
            "..",
            "synthetic_spectra",
            "sfr",
            "sp_{0}_z{1:1.2f}_t{2:2.2f}_degree_{3}_mdegree_{4}_sfr".format(
                sf_type, z, t, degree, mdegree
            ),
        ),
        np.column_stack((age, sfr)),
    )
    np.save(
        os.path.join(
            "..",
            "synthetic_spectra",
            "fitted_model",
            "sp_"
            "{0}_z{1:1.2f}_t{2:2.2f}_degree_{3}_mdegree_{4}_fitted_model_pp".format(
                sf_type, z, t, degree, mdegree
            ),
        ),
        pp,
    )
    np.save(
        os.path.join(
            "..",
            "synthetic_spectra",
            "fitted_model",
            "sp_"
            "{0}_z{1:1.2f}_t{2:2.2f}_degree_{3}_mdegree_{4}_fitted_model_weight".format(
                sf_type, z, t, degree, mdegree
            ),
        ),
        weights,
    )
