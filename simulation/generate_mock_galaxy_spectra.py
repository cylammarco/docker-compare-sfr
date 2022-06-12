import os
from re import L

from sklearn.metrics import max_error

from astropy import units
from astropy import cosmology
import fsps
import numpy as np
from matplotlib import pyplot as plt

if not os.path.exists('output'):
    os.mkdir('output')

def schechter(logm, logphi, logmstar, alpha, m_lower=None):
    """
    Generate a Schechter function (in dlogm).
    """
    phi = (
        (10**logphi)
        * np.log(10)
        * 10 ** ((logm - logmstar) * (alpha + 1))
        * np.exp(-(10 ** (logm - logmstar)))
    )
    return phi


def parameter_at_z0(y, z0, z1=0.2, z2=1.6, z3=3.0):
    """
    Compute parameter at redshift 'z0' as a function
    of the polynomial parameters 'y' and the
    redshift anchor points 'z1', 'z2', and 'z3'.
    """
    y1, y2, y3 = y
    a = ((y3 - y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / (
        z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3)
    )
    b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
    c = y1 - a * z1**2 - b * z1
    return a * z0**2 + b * z0 + c


# Continuity model median parameters + 1-sigma uncertainties.
pars = {
    "logphi1": [-2.44, -3.08, -4.14],
    "logphi1_err": [0.02, 0.03, 0.1],
    "logphi2": [-2.89, -3.29, -3.51],
    "logphi2_err": [0.04, 0.03, 0.03],
    "logmstar": [10.79, 10.88, 10.84],
    "logmstar_err": [0.02, 0.02, 0.04],
    "alpha1": [-0.28],
    "alpha1_err": [0.07],
    "alpha2": [-1.48],
    "alpha2_err": [0.1],
}

# Draw samples from posterior assuming independent Gaussian uncertainties.
# Then convert to mass function at 'z = z0'.
draws = {}
ndraw = 1000
z0 = 0.025
for par in ["logphi1", "logphi2", "logmstar", "alpha1", "alpha2"]:
    samp = np.array(
        [
            np.random.normal(median, scale=err, size=ndraw)
            for median, err in zip(pars[par], pars[par + "_err"])
        ]
    )
    if par in ["logphi1", "logphi2", "logmstar"]:
        draws[par] = parameter_at_z0(samp, z0)
    else:
        draws[par] = samp.squeeze()

# Generate Schechter functions.
logm = np.linspace(8, 12, 10000)  # log(M) grid
phi1 = schechter(
    logm[:, None],
    draws["logphi1"],  # primary component
    draws["logmstar"],
    draws["alpha1"],
)
phi2 = schechter(
    logm[:, None],
    draws["logphi2"],  # secondary component
    draws["logmstar"],
    draws["alpha2"],
)
phi = phi1 + phi2  # combined mass function

# Compute median as the mass function
phi_16, mf, phi_84 = np.percentile(phi, [16, 50, 84], axis=1)

# Get the lower mass limit
M_i_limit = -17
M_i_solar = 4.50
ml_ratio = 3.0
mass_limit = ml_ratio * 10.0 ** (0.4 * (M_i_solar - M_i_limit))
log_mass_limit = np.log10(mass_limit)


# function to draw from the mass function
def draw_random_mass(logm, mf, log_mass_limit):
    mass_mask = logm > log_mass_limit
    mf_normed = mf[mass_mask] / np.nanmax(mf[mass_mask])
    while 1:
        arg = np.random.randint(0, mf_normed.size)
        picked = mf_normed[arg]
        if np.random.random() < picked:
            return logm[mass_mask][arg]


_mass = []
for i in range(10000):
    _mass.append(draw_random_mass(logm, mf, log_mass_limit))

mf_normed = mf / mf[np.argmin(np.abs(logm - log_mass_limit))]
plt.figure(1)
plt.clf()
plt.plot(
    logm,
    mf_normed,
    label="Mass Function at z=0.025",
)
plt.hist(_mass, bins=100, histtype="step", density=True, label="Drawn sample")
plt.yscale("log")
plt.vlines(
    log_mass_limit,
    ymin=min(mf_normed),
    ymax=max(mf_normed),
    color="black",
    label="Lower Mass Limit",
    ls="--"
)
plt.ylim(min(mf_normed), max(mf_normed))
plt.xlim(min(logm), max(logm))
plt.xlabel(r"$\log{(M/M_{\odot})}$")
plt.ylabel(r"$\log{(\phi / \mathrm{Mpc}^{-3} / \mathrm{dex})}$")
plt.title("Mass Function")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('output/input_mass_function.png')



def draw_random_log_age(minimum, maximum, power=2.0, size=None):
    return np.random.power(a=power, size=size) * (maximum - minimum) + minimum


def get_sfh(log_age, peak_age):
    """
    Parameters
    ----------
    log_age: array
        the age to return the SFH.
    peak_age: float
        the time of the maximum star formation.
    Returns
    -------
    The relative SFH at the given log_age location.
    """
    mean = peak_age
    stdv = 0.1
    variance = stdv**2.0
    f = (
        np.exp(-((log_age - mean) ** 2.0) / 2 / variance)
        / np.sqrt(2 * np.pi)
        / stdv
    )
    return f


# log(age) distribution function
log_age = np.linspace(6.5, 10.5, 201)
log_age_bin_size = log_age[1] - log_age[0]

sfh_1 = get_sfh(log_age, 7.5)
sfh_2 = get_sfh(log_age, 8.0)
sfh_3 = get_sfh(log_age, 8.5)
sfh_4 = get_sfh(log_age, 9.0)
sfh_5 = get_sfh(log_age, 9.5)

plt.figure(2)
plt.clf()
plt.plot(log_age, sfh_1)
plt.plot(log_age, sfh_2)
plt.plot(log_age, sfh_3)
plt.plot(log_age, sfh_4)
plt.plot(log_age, sfh_5)
plt.xlabel("log(Age / Gyr)")
plt.ylabel("Arbitrary Density")
plt.title("Example Star Fromation History")
plt.tight_layout()
plt.savefig('output/input_base_sfh.png')



cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)
log_age_universe = np.log10(cosmo.age(0).value * 1e9)


# Python-FSPS defines the age as the time since the beginning of the Universe
# in Gyr, always supply in a sorted array.
sp = fsps.StellarPopulation(
    compute_vega_mags=False,
    zcontinuous=3,
    sfh=3
)

"""
plt.figure(3)
plt.clf()
plt.plot(wave, spectrum)
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.savefig('output/example_spectrum.png')
"""



def sersic_profile(r, alpha=1, n=1):
    '''
    Parameters
    ----------
    r: float
        distance from the centre.
    alpha: float
        scale length (same unit as r)
    n: float
        sersic index
    Returns
    -------
    The fraction of luminosity of the centre (0).
    '''
    return np.exp(-(r/alpha)**(1/n))

# age = lookback time
# time = time since the beginning of the universe
def build_a_galaxy():
    #
    min_age = 8.0
    max_age = log_age_universe * 0.995
    sn = 30.0
    alpha = 1.0
    alpha_halo = 2.0
    n = 1.0
    n_rings = 5
    n_spexels = np.arange(n_rings) * 6
    n_spexels[0] = 1
    mass_ratio = sersic_profile(np.arange(n_rings), alpha=alpha)
    mass_ratio_halo = sersic_profile(np.arange(n_rings), alpha=alpha_halo)
    #
    # Get a random age from the distribution
    random_ages = np.sort(10.0 ** draw_random_log_age(min_age, max_age, power=1.0, size=(n_rings+1)) / 1e9)
    #
    # Convert from lookback time to time since the beginning (Gyr)
    time = 10.0**log_age_universe / 1.0e9 - random_ages
    # Bin width of each time bin (in yr)
    time_bin_duration = 10.**(log_age + log_age_bin_size / 2.) - 10.**(log_age - log_age_bin_size / 2.)
    #
    # setup the SFH, the +1 is to get the oldest component to use as a halo
    sfh = np.array([get_sfh(log_age, np.log10(random_ages[i]*1e9)) for i in range(n_rings + 1)])
    # Normalise the SFH with time
    sfh_normalisation = np.sum(time_bin_duration * sfh, axis=1)
    # Get a random mass from the mass function, in solar mass
    total_mass = 10.0 ** draw_random_mass(logm, mf, log_mass_limit)
    mass_normalisation = np.sum(mass_ratio * n_spexels)
    mass_normalisation_halo = np.sum(mass_ratio_halo * n_spexels)
    mass_per_spexel = total_mass * 0.9 * mass_ratio / mass_normalisation
    mass_per_spexel_halo = total_mass * 0.1 * mass_ratio_halo / mass_normalisation_halo
    data_cube = []
    sfh_cube = []
    fig1 = plt.figure(1)
    plt.clf()
    ax1 = plt.gca()
    fig2 = plt.figure(2)
    plt.clf()
    ax2 = plt.gca()
    # Work in mass 
    _min_temp = 1e10
    _max_temp = 0
    for i, spx in enumerate(n_spexels):
        #
        # the sfh[-1] is the homogeneous halo SFH across the entire galaxy
        sfh_mass = sfh[i] * mass_per_spexel[i] + sfh[-1] * mass_per_spexel_halo[i]
        # change the unit to per year
        sfh_mass /= sfh_normalisation[i]
        sfh_cube.append(sfh_mass)
        #
        # inform the python-fsps the SFH
        _t = np.concatenate(((10.0**log_age_universe / 1.0e9 - 10.0**log_age / 1.0e9), [0]))
        _t_arg_sort = np.argsort(_t)
        _M = np.concatenate((sfh_mass, [0]))
        _Z = 0.019 * np.ones_like(_M)
        sp.set_tabular_sfh(_t[_t_arg_sort], _M[_t_arg_sort], _Z[_t_arg_sort])
        #
        # get the spectrum 
        wave, spectrum = sp.get_spectrum(tage=10.0**log_age_universe / 1.0e9, peraa=True)
        mask = (wave>1000.0) & (wave<100000.0)
        wave = wave[mask]
        spectrum = spectrum[mask]
        #
        # add noise
        noise = spectrum / sn
        for j in range(spx):
            data_cube.append(np.random.normal(spectrum, noise))
        #
        ax1.plot(log_age, sfh_mass, label=str(i))
        ax2.plot(wave, data_cube[-1], label=str(i))
        #
        spec_view_window = data_cube[-1][(wave>3000.0) & (wave<8000.0)]
        if min(spec_view_window) < _min_temp:
            _min_temp = min(spec_view_window)
        if max(spec_view_window) > _max_temp:
            _max_temp = max(spec_view_window)
    data_cube.insert(0, wave)
    sfh_cube.insert(0, log_age)
    ax1.set_xlabel('log(Age / yr)')
    ax1.set_ylabel(r'SFH / (M$_{\odot}$ / yr)')
    ax2.set_xlabel('Wavelength / A')
    ax2.set_ylabel(r'L$_{\odot}$ / A')
    ax2.set_xlim(3000.0, 8000.0)
    ax2.set_ylim(_min_temp, _max_temp)
    ax1.legend()
    ax1.grid()
    fig1.tight_layout()
    ax2.legend()
    ax2.grid()
    fig2.tight_layout()
    #
    return data_cube, sfh_cube, fig1, fig2



if not os.path.exists(os.path.join('output', 'spectrum')):
    os.mkdir(os.path.join('output', 'spectrum'))

if not os.path.exists(os.path.join('output', 'sfh')):
    os.mkdir(os.path.join('output', 'sfh'))

for i in range(1000):
    #
    print(i)
    galaxy, sfh, f1, f2 = build_a_galaxy()
    np.save(os.path.join('output', 'spectrum', 'galaxy_{}'.format(i)), galaxy)
    np.save(os.path.join('output', 'sfh', 'galaxy_sfh_{}'.format(i)), sfh)
    f1.savefig(os.path.join('output', 'sfh', 'galaxy_{}.png'.format(i)))
    f2.savefig(os.path.join('output', 'spectrum', 'galaxy_{}.png'.format(i)))

