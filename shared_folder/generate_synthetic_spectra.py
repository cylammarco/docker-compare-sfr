from matplotlib import pyplot as plt
import fsps
import numpy as np
import os

# Star burst
sp = fsps.StellarPopulation(compute_vega_mags=False,
                                 vactoair_flag=True,
                                 zcontinuous=1,
                                 sfh=0)

if not os.path.exists('synthetic_spectra'):
    os.mkdir('synthetic_spectra')

for z in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    sp.params['logzsol'] = z
    sp.params['sfh'] = 0
    # Star burst
    plt.figure(1, figsize=(12, 12))
    plt.clf()
    for age in 10.**np.arange(-1.3, 1.3, 0.1):
        wave, spec = sp.get_spectrum(tage=age, peraa=True)
        plt.plot(
            wave,
            spec,
            label=r'Star Burst, log(Z/Z$_\odot$) = {0:1.1f}, age = {1:2.2f} Gyr'
            .format(z, age))
        np.save(
            'synthetic_spectra/sp_z{0:1.2f}_t{1:2.2f}'.format(
                z, age), np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / Hz)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('synthetic_spectra/sb00_z{0:1.2f}.png'.format(z))
    # Exponential
    plt.figure(2, figsize=(12, 12))
    plt.clf()

    sp.params['sfh'] = 1
    sp.params['tau'] = 0.5

    for age in 10.**np.arange(-1.3, 1.3, 0.1):
        wave, spec = sp.get_spectrum(tage=age, peraa=True)
        plt.plot(
            wave,
            spec,
            label=
            r'$\tau$ = {0:.1f} Gyr, log(Z/Z$_\odot$) = {1:1.2f}, age = {2:2.2f} Gyr'.
            format(sp.params['tau'], z, age))
        np.save(
            'synthetic_spectra/sp_z{0:1.2f}_t{1:2.2f}'.format(
                z, age), np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / Hz)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('synthetic_spectra/ed05_z{0:1.2f}.png'.format(z))
