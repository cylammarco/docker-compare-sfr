from astropy import constants
from astropy.io import fits
import numpy as np
import os
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from spectres import spectres
from matplotlib import gridspec
from matplotlib import pyplot as plt

plt.ion()

data_firefly = fits.open("manga-firefly-v3_1_1-miles.fits.gz")
firefly_mask = np.where(
    data_firefly["GALAXY_INFO"].data["PLATEIFU"] == "10001-6104"
)[0][0]

ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
miles_pathname = os.path.join(ppxf_dir, "miles_models", "Eun1.30Z*.fits")


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


data = fits.open(
    "manga-10001/manga-10001-6104-LOGCUBE-VOR10-MILESHC-MASTARSSP.fits.gz"
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
wave_new = np.arange(3650.0, 7500.0, FWHM_gal)
velscale = c * np.median(np.diff(wave_new)) / wave_new[-1]

wave = data["WAVE"].data
velscale = c * np.log(wave_new[1] / wave_new[0])

# Estimate the wavelength fitted range in the rest frame.
lam_range_gal = np.array([np.min(wave), np.max(wave)]) / (1 + z)

### for setting up the MILES
_galaxy = data["FLUX"].data[:, x, y]
_galaxy = spectres(wave_new, wave, _galaxy)

# natural log
(_, _, velscale_rebinned,) = util.log_rebin(
    [np.nanmin(wave_new), np.nanmax(wave_new)],
    _galaxy,
    velscale=velscale,
)

miles = lib.miles(
    miles_pathname,
    velscale_rebinned,
    FWHM_gal,
    age_range=None,
    metal_range=[-0.05, 0.05],
    wave_range=(3500, 10500),
)
reg_dim = miles.templates.shape[1:]
stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
n_temps = stars_templates.shape[1]

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
)

# Combines the stellar and gaseous templates into a single array.
# During the PPXF fit they will be assigned a different kinematic
# COMPONENT value
#
templates = np.column_stack([stars_templates, gas_templates])

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

    galaxy, galaxy_err = spectres(wave_new, wave, galaxy, galaxy_err)

    # natural log
    (galaxy_log_rebinned, wave_rebinned, velscale_rebinned,) = util.log_rebin(
        [np.nanmin(wave_new), np.nanmax(wave_new)],
        galaxy,
        velscale=velscale,
    )

    (noise, wave_rebinned, velscale_rebinned,) = util.log_rebin(
        [np.nanmin(wave_new), np.nanmax(wave_new)],
        galaxy_err,
        velscale=velscale,
    )

    noise = np.sqrt(noise)

    weights = []

    goodpixels = np.arange(len(galaxy_log_rebinned))

    vel = c * np.log(1 + z)
    # (km/s), starting guess for [V, sigma]
    start = [[vel, 1.0, h3, h4], [vel, 1.0], [vel, 1.0]]

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
        degree=8,
        mdegree=4,
        clean=False,
        ftol=1e-4,
        lam=np.exp(wave_rebinned),
        lam_temp=np.exp(miles.ln_lam_temp),
        linear_method="lsq_box",
        regul=100.0,
        bias=0.0,
        reg_dim=reg_dim,
        component=component,
        quiet=False,
        gas_component=gas_component,
        gas_names=gas_names,
        gas_reddening=gas_reddening,
    )

    weights = _pp.weights[~gas_component]
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
        _pp.bestfit,
        color="black",
        label="Fitted",
    )
    ax1.scatter(
        np.exp(wave_rebinned),
        galaxy_log_rebinned - _pp.bestfit,
        color="green",
        s=2,
        label="Residual",
    )
    ax1.grid()
    ax1.set_xlim(min(np.exp(wave_rebinned)), max(np.exp(wave_rebinned)))
    ax1.set_xlabel("Wavelength / A")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title("Plate {} Fibre {} VorID {}".format(10001, 6104, _id))
    ax1.legend()

    ax3.imshow(
        weights.T,
        origin="lower",
        extent=[min(age), max(age), -0.2, 0.11],
        aspect="auto",
    )
    ax3.set_xticklabels([""])
    ax3.set_ylabel("Metallicity")
    ax3.set_yticks([0.0])
    ax3.set_yticklabels(["0.0"])
    ax3b = ax4.twinx()
    ax3b.set_ylabel("Recovered")
    ax3b.set_yticklabels([""])

    ax4.plot(age, sfr / np.nanmax(sfr), label="Recovered (light-weighted)")
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

    firefly_id = firefly_bin_id_list[firefly_mask]
    firefly_sfh = firefly_sfh_list[firefly_mask]

    bin_id = firefly_id[x, y]
    sfh_data_grid_pos = np.where(
        firefly_spatial_info_list[firefly_mask][:, 0] == bin_id
    )
    firefly_sfh_x_y = firefly_sfh[sfh_data_grid_pos][0]
    firefly_sfh_x_y = firefly_sfh_x_y[firefly_sfh_x_y[:, 2] > 0]

    ax4.scatter(
        10.0 ** firefly_sfh_x_y[:, 0],
        firefly_sfh_x_y[:, 2] / max(firefly_sfh_x_y[:, 2]),
        label="Firefly",
        marker="+",
        color="black",
        s=50,
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
    plt.savefig("manga-10001/manga-10001-6104-{}-regul-5.png".format(_id))


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
