from matplotlib import pyplot as plt
import fsps
import numpy as np
import os

# Star burst
sp_sb00 = fsps.StellarPopulation(compute_vega_mags=False,
                                 vactoair_flag=True,
                                 zcontinuous=1,
                                 sfh=0)

# Exponentially decaying SFH with tau=3 Gyr
sp_ed30 = fsps.StellarPopulation(compute_vega_mags=False,
                                 vactoair_flag=True,
                                 zcontinuous=1,
                                 sfh=1,
                                 tau=3.0)

if not os.path.exists('synthetic_spectra'):
    os.mkdir('synthetic_spectra')


for z in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    sp_sb00.params['logzsol'] = z
    sp_ed30.params['logzsol'] = z
    # Star burst
    plt.figure(1, figsize=(12, 12))
    plt.clf()
    for age in 10.**np.arange(-2.0, 1.3, 0.1):
        wave, spec = sp_sb00.get_spectrum(tage=age)
        plt.plot(
            wave,
            spec,
            label=r'Star Burst, log(Z/Z$_\odot$) = {0:1.1f}, age = {1:2.2f} Gyr'
            .format(z, age))
        np.save(
            'synthetic_spectra/sp_sb00_z{0:1.1f}_t{1:06d}'.format(
                z, int(age * 1000)), np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / Hz)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('synthetic_spectra/sb00_z{0:1.1f}.png'.format(z))
    # Exponential
    plt.figure(2, figsize=(12, 12))
    plt.clf()
    for age in 10.**np.arange(-2.0, 1.3, 0.1):
        wave, spec = sp_ed30.get_spectrum(tage=age)
        plt.plot(
            wave,
            spec,
            label=
            r'$\tau$ = 3 Gyr, log(Z/Z$_\odot$) ={0:1.1f}, age = {1:2.2f} Gyr'.
            format(z, age))
        np.save(
            'synthetic_spectra/sp_ed30_z{0:1.1f}_t{1:06d}'.format(
                z, int(age * 1000)), np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / Hz)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('synthetic_spectra/ed30_z{0:1.1f}.png'.format(z))
