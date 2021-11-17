from matplotlib import pyplot as plt
import fsps
import numpy as np
import os

# Star burst of 0.5 Gyr, log(Z/Z_sol) = 0.0
sp_sb05_z00 = fsps.StellarPopulation(compute_vega_mags=False,
                                     zcontinuous=1,
                                     sfh=1,
                                     tau=1.0,
                                     const=1.0,
                                     sf_start=0.0,
                                     sf_trunc=0.5,
                                     logzsol=0.0,
                                     dust_type=0,
                                     dust2=0.)

# Star burst of 0.5 Gyr, log(Z/Z_sol) = 0.1
sp_sb05_zp01 = fsps.StellarPopulation(compute_vega_mags=False,
                                      zcontinuous=1,
                                      sfh=1,
                                      tau=1.0,
                                      const=1.0,
                                      sf_start=0.0,
                                      sf_trunc=0.5,
                                      logzsol=0.1,
                                      dust_type=0,
                                      dust2=0.)

# Star burst of 0.5 Gyr, log(Z/Z_sol) = -0.1
sp_sb05_zm01 = fsps.StellarPopulation(compute_vega_mags=False,
                                      zcontinuous=1,
                                      sfh=1,
                                      tau=1.0,
                                      const=1.0,
                                      sf_start=0.0,
                                      sf_trunc=0.5,
                                      logzsol=-0.1,
                                      dust_type=0,
                                      dust2=0.)

# Star burst of 0.5 Gyr, log(Z/Z_sol) = 0.2
sp_sb05_zp02 = fsps.StellarPopulation(compute_vega_mags=False,
                                      zcontinuous=1,
                                      sfh=1,
                                      tau=1.0,
                                      const=1.0,
                                      sf_start=0.0,
                                      sf_trunc=0.5,
                                      logzsol=0.2,
                                      dust_type=0,
                                      dust2=0.)

# Star burst of 0.5 Gyr, log(Z/Z_sol) = -0.2
sp_sb05_zm02 = fsps.StellarPopulation(compute_vega_mags=False,
                                      zcontinuous=1,
                                      sfh=1,
                                      tau=1.0,
                                      const=1.0,
                                      sf_start=0.0,
                                      sf_trunc=0.5,
                                      logzsol=-0.2,
                                      dust_type=0,
                                      dust2=0.)

#
# Exponentially decaying SFH with tau=1 Gyr, log(Z/Z_sol) = 0.0
sp_ed1_z00 = fsps.StellarPopulation(compute_vega_mags=False,
                                    zcontinuous=1,
                                    sfh=1,
                                    tau=1.0,
                                    const=0.0,
                                    logzsol=0.0,
                                    dust_type=0,
                                    dust2=0.)

# Exponentially decaying SFH with tau=1 Gyr, log(Z/Z_sol) = 0.1
sp_ed1_zp01 = fsps.StellarPopulation(compute_vega_mags=False,
                                     zcontinuous=1,
                                     sfh=1,
                                     tau=1.0,
                                     const=0.0,
                                     logzsol=0.1,
                                     dust_type=0,
                                     dust2=0.)

# Exponentially decaying SFH with tau=1 Gyr, log(Z/Z_sol) = -0.1
sp_ed1_zm01 = fsps.StellarPopulation(compute_vega_mags=False,
                                     zcontinuous=1,
                                     sfh=1,
                                     tau=1.0,
                                     const=0.0,
                                     logzsol=-0.1,
                                     dust_type=0,
                                     dust2=0.)

# Exponentially decaying SFH with tau=1 Gyr, log(Z/Z_sol) = 0.2
sp_ed1_zp02 = fsps.StellarPopulation(compute_vega_mags=False,
                                     zcontinuous=1,
                                     sfh=1,
                                     tau=1.0,
                                     const=0.0,
                                     logzsol=0.2,
                                     dust_type=0,
                                     dust2=0.)

# Exponentially decaying SFH with tau=1 Gyr, log(Z/Z_sol) = -0.2
sp_ed1_zm02 = fsps.StellarPopulation(compute_vega_mags=False,
                                     zcontinuous=1,
                                     sfh=1,
                                     tau=1.0,
                                     const=0.0,
                                     logzsol=-0.2,
                                     dust_type=0,
                                     dust2=0.)

if not os.path.exist('mock_spectrum'):
    os.path.mkdir('mock_spectrum')

for age in np.arange(1, 15):
    # Star burst
    plt.figure(1)
    plt.clf()
    wave, spec = sp_sb05_z00.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = 0.0, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_z00_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_sb05_zp01.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = 0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zp01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_sb05_zm01.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = +0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zm01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_sb05_zp02.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = 0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zp02_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_sb05_zm02.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = -0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zm02_t{}'.format(age),
            np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / A)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('mock_spectrum/sb05_t{}'.format(age))
    # Exponential
    plt.figure(2)
    plt.clf()
    wave, spec = sp_ed1_z00.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = 0.0, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_z00_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_ed1_zp01.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = 0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zp01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_ed1_zm01.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = -0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zm01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_ed1_zp02.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = 0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zp02_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = sp_ed1_zm02.get_spectrum(tage=age)
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = -0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zm02_t{}'.format(age),
            np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / A)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('mock_spectrum/ed1_t{}'.format(age))




'''
# If want to replot with minor changes to the plots.

for age in np.arange(1, 15):
    # Star burst
    plt.figure(1)
    plt.clf()
    wave, spec = np.load('mock_spectrum/sp_sb05_z00_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = 0.0, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_z00_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_sb05_zp01_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = 0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zp01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_sb05_zm01_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = +0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zm01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_sb05_zp02_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = 0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zp02_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_sb05_zm02_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'Star Burst 0.5 Gyr, log(Z/Z$_\odot$) = -0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_sb05_zm02_t{}'.format(age),
            np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / A)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('mock_spectrum/sb05_t{}'.format(age))
    # Exponential
    plt.figure(2)
    plt.clf()
    wave, spec = np.load('mock_spectrum/sp_ed1_z00_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = 0.0, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_z00_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_ed1_zp01_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = 0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zp01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_ed1_zm01_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = -0.1, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zm01_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_ed1_zp02_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = 0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zp02_t{}'.format(age),
            np.column_stack((wave, spec)))
    wave, spec = np.load('mock_spectrum/sp_ed1_zm02_t{}.npy'.format(age)).T
    plt.plot(wave, spec, label=r'$\tau$ = 1 Gyr, log(Z/Z$_\odot$) = -0.2, age = {} Gyr'.format(age))
    np.save('mock_spectrum/sp_ed1_zm02_t{}'.format(age),
            np.column_stack((wave, spec)))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (erg / s / cm / cm / A)')
    plt.xlim(3500, 8500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('mock_spectrum/ed1_t{}'.format(age))

'''