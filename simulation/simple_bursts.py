import os

from astropy import units
from astropy import cosmology
import fsps
import numpy as np
from matplotlib import pyplot as plt

if not os.path.exists("output"):
    os.mkdir("output")

if not os.path.exists(os.path.join("output", "simple_bursts")):
    os.mkdir(os.path.join("output", "simple_bursts"))

cosmo = cosmology.FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc,
    Tcmb0=2.725 * units.K,
    Om0=0.3,
)


# Python-FSPS defines the age as the time since the beginning of the Universe
# in Gyr, always supply in a sorted array.
sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, sfh=0)
sn = 30.0

# age = lookback time
for metallicity in np.arange(-1.0, 1.1, 0.25):
    sp.params["logzsol"] = metallicity
    for age in np.arange(0.5, 13.5, 0.5):
        wave, spectrum = sp.get_spectrum(tage=age, peraa=True)
        #
        # get the spectrum
        mask = (wave > 3000.0) & (wave < 10000.0)
        wave = wave[mask]
        spectrum = spectrum[mask]
        #
        # add noise
        noise = spectrum / sn
        #
        data_out = np.column_stack((wave, spectrum, noise))
        np.save(
            os.path.join(
                "output",
                "simple_bursts",
                "spectrum_{}_gyr_{}_logzsol".format(age, metallicity),
            ),
            data_out,
        )
        plt.figure(1)
        plt.clf()
        plt.plot(wave, spectrum)
        plt.xlabel("Wavelength / A")
        plt.ylabel("Flux per A")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                "output",
                "simple_bursts",
                "spectrum_{}_gyr_{}_logzsol.png".format(age, metallicity),
            )
        )
