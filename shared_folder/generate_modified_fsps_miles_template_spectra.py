from matplotlib.pyplot import *
from astropy.io import fits
import glob
import os
import numpy as np
import ppxf as ppxf_package

import fsps
from spectres import spectres

ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
miles_pathname = ppxf_dir + os.sep + 'miles_models/Mun1.30*.fits'
output_folder_pathname = ppxf_dir + os.sep + 'fsps_miles_models'

if not os.path.exists(output_folder_pathname):
    os.mkdir(output_folder_pathname)

miles_filelist = glob.glob(miles_pathname)

sp = fsps.StellarPopulation(compute_vega_mags=False,
                            vactoair_flag=True,
                            zcontinuous=1,
                            sfh=0)

for i in miles_filelist:
    filename = i.split(os.sep)[-1]
    metallicity = filename.split('Z')[-1].split('T')[0]
    if metallicity[0] == 'p':
        metallicity = float(metallicity[1:])
    elif metallicity[0] == 'm':
        metallicity = -float(metallicity[1:])
    else:
        print(
            'This should not happen. Check the filename: {}.'.format(filename))
    age = float(filename.split('T')[-1].split('_iPp')[0])
    miles_data = fits.open(i)[0]
    w_start = float(miles_data.header['CRVAL1'])
    w_bin = float(miles_data.header['CDELT1'])
    w_length = int(miles_data.header['NAXIS1'])
    w_end = w_start + (w_length - 1) * w_bin
    wave_model = np.linspace(w_start, w_end, w_length)
    '''
    flux_model = miles_data.data
    '''
    sp.params['logzsol'] = metallicity
    wave_fsps, flux_fsps = sp.get_spectrum(tage=age, peraa=True)
    flux_fsps = spectres(wave_model, wave_fsps, flux_fsps)
    '''
    figure(1)
    clf()
    plot(wave_fsps, flux_fsps * 0.75, label='FSPS')
    plot(wave_model, flux_model, label='MILES')
    legend()
    xlim(3500, 8000)
    '''
    miles_data.header['COMMENT'] = "The modified spectra generated from FSPS."
    miles_data.data = flux_fsps
    miles_data.writeto(os.path.join(output_folder_pathname, filename))
