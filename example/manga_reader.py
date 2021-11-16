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

    def _get_metadata(self, header, header2):

        self.x_center = int(header2['CRPIX1']) - 1
        self.y_center = int(header2['CRPIX2']) - 1

        try:
            self.dx = header2['CD1_1'] * 3600.  # deg to arcsec
            self.dy = header2['CD2_2'] * 3600.  # deg to arcsec
        except Exception:
            self.dx = header2['PC1_1'] * 3600
            self.dy = header2['PC2_2'] * 3600

        self.ebvgal = float(header['EBVGAL'])
        self.gfwhm = float(header['GFWHM'])
        self.rfwhm = float(header['RFWHM'])
        self.ifwhm = float(header['IFWHM'])
        self.zfwhm = float(header['ZFWHM'])

        self.xdim, self.ydim = np.shape(self.flux[0])
        self._idx_to_pix = {}

        for id in self.unique_id:

            self._idx_to_pix[id] = np.column_stack(np.where(self.binid == id))

    def load_lincube(self, filename):

        self.lincube_data = fits.open(filename)
        self.lincube_header = self.lincube_data[0].header
        self.lincube_flux_header = self.lincube_data['flux'].header

        self.binid = self.lincube_data['binid'].data[0]
        self.unique_id = np.unique(self.binid)
        self.unique_id = self.unique_id[self.unique_id >= 0]
        self.flux = self.lincube_data['flux'].data * 1e-17
        self.flux_err = np.sqrt(1. / self.lincube_data['ivar'].data) * 1e-17
        self.wave = self.lincube_data['wave'].data

        self._get_metadata(self.lincube_header, self.lincube_flux_header)

    def load_logcube(self, filename):

        self.logcube_data = fits.open(filename)
        self.logcube_header = self.logcube_data[0].header
        self.logcube_flux_header = self.logcube_data['flux'].header

        self.binid = self.logcube_data['binid'].data[0]
        self.unique_id = np.unique(self.binid)
        self.unique_id = self.unique_id[self.unique_id >= 0]
        self.flux = 10.**self.logcube_data['flux'].data * 1e-17
        self.flux_err = np.sqrt(1. / 10.**self.logcube_data['ivar'].data) * 1e-17
        self.wave = self.logcube_data['wave'].data

        self._get_metadata(self.logcube_header, self.logcube_flux_header)

    def iterate_data(self, mode='vor', log=False):

        if mode == 'pix':

            if log:

                return [
                    np.vstack((self.wave, np.log10(self.flux[:, i, j]),
                               np.log10(self.flux_err[:, i, j])))
                    for (i, j) in list(
                        itertools.product(range(self.xdim), range(self.ydim)))
                ]

            else:

                return [
                    np.vstack((self.wave, self.flux[:, i,
                                                    j], self.flux_err[:, i,
                                                                      j]))
                    for (i, j) in list(
                        itertools.product(range(self.xdim), range(self.ydim)))
                ]

        elif mode == 'vor':

            pix = [self._idx_to_pix[k] for k in self.unique_id]
            for i, p in enumerate(pix):
                pix[i] = p[0]

            if log:

                return [
                    np.vstack((self.wave, np.log10(self.flux[:, i, j]),
                               np.log10(self.flux_err[:, i, j])))
                    for (i, j) in pix
                ]

            else:

                return [
                    np.vstack((self.wave, self.flux[:, i,
                                                    j], self.flux_err[:, i,
                                                                      j]))
                    for (i, j) in pix
                ]

        else:

            print('Unknown mode: {}.'.format(mode))

    def idx_to_pix(self, idx):
        '''
        idx starts from 1.
        '''
        return self._idx_to_pix[idx]

    def pix_to_idx(self, x, y):
        '''
        pixel position starts from 0.
        '''
        return self.binid[x, y]

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
