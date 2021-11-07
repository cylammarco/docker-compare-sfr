from astropy.io import fits
import itertools
import numpy as np
from matplotlib import pyplot as plt


class manga_reader:
    def __init__(self, redshift=None):

        if redshift is None:

            print('Please provide a redshift of the observation. '
                  'It is currently set to 0.0.')
            redshift = 0.0

        self.redshift = redshift

    def _get_metadata(self, header):

        self.x_center = int(header['CRPIX1']) - 1
        self.y_center = int(header['CRPIX2']) - 1

        self.dx = header['CD1_1'] * 3600.  # deg to arcsec
        self.dy = header['CD2_2'] * 3600.  # deg to arcsec

        self.ebvgal = float(header['EBVGAL'])
        self.gfwhm = float(header['GFWHM'])
        self.rfwhm = float(header['RFWHM'])
        self.ifwhm = float(header['IFWHM'])
        self.zfwhm = float(header['ZFWHM'])

    def load_lincube(self, filename):

        self.lincube_data = fits.open(filename)
        self.lincube_flux_header = self.lincube_data['flux'].header

        self.flux = self.lincube_data['flux'].data
        self.flux_err = 1. / self.lincube_data['ivar'].data
        self.wave = self.lincube_data['wave'].data

        self._get_metadata(self.lincube_flux_header)

    def load_logcube(self, filename):

        self.logcube_data = fits.open(filename)
        self.logcube_flux_header = self.logcube_data['flux'].header

        self.flux = 10.**self.logcube_data['flux'].data
        self.flux_err = 1. / 10.**self.logcube_data['ivar'].data
        self.wave = self.logcube_data['wave'].data

        self._get_metadata(self.logcube_flux_header)

    def iterate_data(self, log=False):

        xdim, ydim = np.shape(self.flux[0])

        if log:

            return [
                np.vstack((self.wave, self.flux[:, i, j], self.flux_err[:, i,
                                                                        j]))
                for (i, j) in list(itertools.product(range(xdim), range(ydim)))
            ]

        else:

            return [
                np.vstack((self.wave, self.flux[:, i, j], self.flux_err[:, i,
                                                                        j]))
                for (i, j) in list(itertools.product(range(xdim), range(ydim)))
            ]

    def _imshow(self, log, **kwarg):

        if log:

            im = np.log10(self.flux[self.ind_wave, :, :].sum(axis=0)).T
            try:
                header = self.logcube_flux_header
            except Exception:
                header = self.lincube_flux_header

        else:

            im = self.flux[self.ind_wave, :, :].sum(axis=0).T
            try:
                header = self.lincube_flux_header
            except Exception:
                header = self.logcube_flux_header

        x_extent = (np.array([0., im.shape[0]]) -
                    (im.shape[0] - self.x_center)) * self.dx * (-1)
        y_extent = (np.array([0., im.shape[1]]) -
                    (im.shape[1] - self.y_center)) * self.dy
        extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

        fig = plt.figure(**kwarg)

        plt.imshow(im,
                   extent=extent,
                   vmin=0.1 * np.nanmax(im),
                   vmax=0.9 * np.nanmax(im),
                   origin='lower',
                   interpolation='none',
                   aspect='auto')
        plt.colorbar(label=header['BUNIT'])
        plt.xlabel('arcsec')
        plt.ylabel('arcsec')

        return fig

    def imshow(self, wave=[6550, 6680], log=False, show=True, **kwarg):
        '''
        Parameters
        ----------
        wave: list of size 2 (default: [6550, 6680])
            The minimum and maximum wavelengths at which data are summed to
            generate the 2D reconstructed image. Default shows the H-alpha.
        log: bool or 'both' (default: False)
            Set the z-scale to log.
        show: bool (default: True)
            Set to display the figure.
        savefig: str or None (default: None)
            Provide a filepath+name with extension for saving the figure.
        kwarg: dict
            Keyword argument for configuring the figure preset.

        '''

        self.ind_wave = np.where((self.wave / (1 + self.redshift) > wave[0])
                                 & (self.wave /
                                    (1 + self.redshift) < wave[1]))[0]

        if log or (log == 'both'):

            self.logcube_fig = self._imshow(log=True, **kwarg)

        elif not log or (log == 'both'):

            self.lincube_fig = self._imshow(log=False, **kwarg)

        else:

            print('Unknown log parameter: {}.'.format(log))

        if show:

            plt.show()
