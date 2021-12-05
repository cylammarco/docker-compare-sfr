import os
import glob

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from matplotlib import pyplot as plt
import numpy as np
from plotbin.plot_velfield import plot_velfield
import ppxf as ppxf_package
import ppxf.miles_util as lib
from speclite import filters as sfilters


class ppxf_reader:
    def __init__(self):

        ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
        pathname = ppxf_dir +\
            '/miles_models/Mun1.30*.fits'

        file_vega = ppxf_dir +\
            "/miles_models/Vazdekis2012_ssp_phot_Padova00_UN_v10.0.txt"
        file_sdss = ppxf_dir +\
            "/miles_models/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
        file1 = ppxf_dir +\
            "/miles_models/Vazdekis2012_ssp_mass_Padova00_UN_baseFe_v10.0.txt"

        self.vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
        self.sdss_bands = ["u", "g", "r", "i"]
        self.vega_sun_mag = [
            5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334
        ]
        self.sdss_sun_mag = [6.55, 5.12, 4.68,
                             4.57]  # values provided by Elena Ricciardelli

        self.slope1, self.MH1, self.Age1, self.m_no_gas = np.loadtxt(
            file1, usecols=[1, 2, 3, 5]).T
        slope2_vega, MH2_vega, Age2_vega, m_U, m_B, m_V, m_R, m_I,\
            m_J, m_H, m_K = np.loadtxt(
                file_vega, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).T
        slope2_sdss, MH2_sdss, Age2_sdss, m_u, m_g, m_r, m_i = np.loadtxt(
            file_sdss, usecols=[1, 2, 3, 4, 5, 6, 7]).T

        self.slope2 = {"vega": slope2_vega, "sdss": slope2_sdss}
        self.MH2 = {"vega": MH2_vega, "sdss": MH2_sdss}
        self.Age2 = {"vega": Age2_vega, "sdss": Age2_sdss}

        self.m_vega = {
            "U": m_U,
            "B": m_B,
            "V": m_V,
            "R": m_R,
            "I": m_I,
            "J": m_J,
            "H": m_H,
            "K": m_K
        }
        self.m_sdss = {"u": m_u, "g": m_g, "r": m_r, "i": m_i}

        # Only getting the age from the miles library, the values in the
        # second and third arguments are random.
        self.miles_backup = lib.miles(pathname, 100, 2.5)
        self.age_backup = self.miles_backup.age_grid[:, 0]

        self.miles = self.miles_backup
        self.age = self.age_backup
        self.sfh = None
        self.luminosity = None
        self.ml_ratio = None

        self.idx_to_pix = None

    def mass_to_light(self, weights, band="r", quiet=False):
        """
        Computes the M/L in a chosen band, given the weights produced
        in output by pPXF. A Salpeter IMF is assumed (slope=1.3).
        The returned M/L includes living stars and stellar remnants,
        but excludes the gas lost during stellar evolution.

        This procedure uses the photometric predictions
        from Vazdekis+12 and Ricciardelli+12

        http://adsabs.harvard.edu/abs/2012MNRAS.424..157V
        http://adsabs.harvard.edu/abs/2012MNRAS.424..172R

        they were downloaded in December 2016 below and are included in pPXF
        with permission

        http://www.iac.es/proyecto/miles/pages/photometric-predictions/
            based-on-miuscat-seds.php

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

        assert self.miles.age_grid.shape == self.miles.metal_grid.shape ==\
            weights.shape, "Input weight dimensions do not match"

        if band in self.vega_bands:
            k = self.vega_bands.index(band)
            sun_mag = self.vega_sun_mag[k]
            mag = self.m_vega[band]
            Age2 = self.Age2['vega']
            slope2 = self.slope2['vega']
            MH2 = self.MH2['vega']
        elif band in self.sdss_bands:
            k = self.sdss_bands.index(band)
            sun_mag = self.sdss_sun_mag[k]
            mag = self.m_sdss[band]
            Age2 = self.Age2['sdss']
            slope2 = self.slope2['sdss']
            MH2 = self.MH2['sdss']
        else:
            raise ValueError("Unsupported photometric band")

        # The following loop is a brute force, but very safe and general,
        # way of matching the photometric quantities to the SSP spectra.
        # It makes no assumption on the sorting and dimensions of the files
        mass_no_gas_grid = np.empty_like(weights)
        lum_grid = np.empty_like(weights)
        for j in range(self.miles.n_ages):
            for k in range(self.miles.n_metal):
                p1 = (np.abs(self.miles.age_grid[j, k] - self.Age1) <
                      0.001) & (np.abs(self.miles.metal_grid[j, k] - self.MH1)
                                < 0.01) & (np.abs(1.30 - self.slope1) < 0.01)
                mass_no_gas_grid[j, k] = self.m_no_gas[p1]

                p2 = (np.abs(self.miles.age_grid[j, k] - Age2) <
                      0.001) & (np.abs(self.miles.metal_grid[j, k] - MH2) <
                                0.01) & (np.abs(1.30 - slope2) < 0.01)
                lum_grid[j, k] = 10**(-0.4 * (mag[p2] - sun_mag))

        # This is eq.(2) in Cappellari+13
        # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
        mlpop = np.sum(weights * mass_no_gas_grid) / np.sum(weights * lum_grid)

        if not quiet:
            print('M/L_' + band + ': %#.4g' % mlpop)

        return mlpop

    def load_file(self, ppxf_output_filename):

        idx = int(ppxf_output_filename.split('.')[-2].split('_')[-1])
        ppxf_output = np.load(ppxf_output_filename, allow_pickle=True).item()
        if self.idx_to_pix is not None:
            pix = self.idx_to_pix[idx]
        else:
            pix = None

        return idx, pix, ppxf_output

    def load(self, ppxf_output_folder, verbose=True):
        '''
        Parameters
        ----------
        ppxf_output_folder: str
            The path to the folder relative to the current working directory,
            or an absolute path.

        '''

        # Sort out the input paths (ppxf_output refers to the output from
        # ppxf, not the output from this script)
        self.ppxf_output_folder = os.path.abspath(ppxf_output_folder)
        self.folder_name = self.ppxf_output_folder.split(os.sep)[-1]

        # Sort out the output paths
        self.ppxf_npy_folder = os.path.join(self.ppxf_output_folder, 'npy')
        self.ppxf_model_folder = os.path.join(self.ppxf_output_folder, 'model')
        self.plots_folder = os.path.join(self.ppxf_output_folder,
                                         'diagnostic_plots')

        if not os.path.exists(self.plots_folder):
            os.mkdir(self.plots_folder)

        # Get the voronoi idx to pixel position
        self.idx_to_pix_path = os.path.join(
            self.ppxf_output_folder,
            'manga_' + '_'.join(self.folder_name.split('-')[1:3]) +
            '_ppxf_idx_to_pix.npy')
        self.pix_to_wcs_path = os.path.join(
            self.ppxf_output_folder,
            'manga_' + '_'.join(self.folder_name.split('-')[1:3]) +
            '_ppxf_pix_to_wcs.npy')
        self.miles_model_path = os.path.join(
            self.ppxf_output_folder,
            'manga_' + '_'.join(self.folder_name.split('-')[1:3]) +
            '_ppxf_miles_model.npy')

        self.idx_to_pix = np.load(self.idx_to_pix_path,
                                  allow_pickle=True).item()
        self.pix_to_wcs = np.load(self.pix_to_wcs_path,
                                  allow_pickle=True)

        idx_full_list = []
        pix_x = []
        pix_y = []
        pix_ra = []
        pix_dec = []
        self.results = {}
        self.filelist = glob.glob(
            os.path.join(self.ppxf_npy_folder, '*[0-9].npy'))
        for filename in self.filelist:

            if verbose:
                print('Loading {}.'.format(filename))

            idx, pix, ppxf_output = self.load_file(
                os.path.join(self.ppxf_output_folder, filename))
            self.results[idx] = {}
            self.results[idx]['pix'] = pix
            self.results[idx]['npix'] = ppxf_output.npix
            self.results[idx]['flux'] = ppxf_output.galaxy
            self.results[idx]['flux_err'] = ppxf_output.noise
            self.results[idx]['clean'] = ppxf_output.clean
            self.results[idx]['fraction'] = ppxf_output.fraction
            self.results[idx]['ftol'] = ppxf_output.ftol
            self.results[idx]['degree'] = ppxf_output.degree
            self.results[idx]['mdegree'] = ppxf_output.mdegree
            self.results[idx]['method'] = ppxf_output.method
            self.results[idx]['sky'] = ppxf_output.sky
            self.results[idx]['vsyst'] = ppxf_output.vsyst
            self.results[idx]['regul'] = ppxf_output.regul
            self.results[idx]['wave'] = ppxf_output.lam
            self.results[idx]['nfev'] = ppxf_output.nfev
            self.results[idx]['reddening'] = ppxf_output.reddening
            self.results[idx]['reg_dim'] = ppxf_output.reg_dim
            self.results[idx]['reg_ord'] = ppxf_output.reg_ord
            #taking up too much memory
            #self.results[idx]['templates'] = ppxf_output.templates
            self.results[idx]['npix_temp'] = ppxf_output.npix_temp
            self.results[idx]['ntemp'] = ppxf_output.ntemp
            self.results[idx]['sigma_diff'] = ppxf_output.sigma_diff
            self.results[idx]['status'] = ppxf_output.status
            self.results[idx]['velscale'] = ppxf_output.velscale
            self.results[idx]['velscale_ratio'] = ppxf_output.velscale_ratio
            self.results[idx]['tied'] = ppxf_output.tied
            self.results[idx]['gas_flux'] = ppxf_output.gas_flux
            self.results[idx]['gas_flux_error'] = ppxf_output.gas_flux_error
            self.results[idx]['gas_bestfit'] = ppxf_output.gas_bestfit
            self.results[idx]['gas_mpoly'] = ppxf_output.gas_mpoly
            self.results[idx]['gas_reddening'] = ppxf_output.gas_reddening
            self.results[idx]['gas_component'] = ppxf_output.gas_component
            self.results[idx]['gas_names'] = ppxf_output.gas_names
            self.results[idx]['gas_any'] = ppxf_output.gas_any
            self.results[idx]['gas_any_zero'] = ppxf_output.gas_any_zero
            self.results[idx][
                'gas_zero_template'] = ppxf_output.gas_zero_template
            self.results[idx]['linear_method'] = ppxf_output.linear_method
            self.results[idx]['x0'] = ppxf_output.x0
            self.results[idx]['reddening_func'] = ppxf_output.reddening_func
            self.results[idx]['polyval'] = ppxf_output.polyval
            self.results[idx]['polyvander'] = ppxf_output.polyvander
            self.results[idx]['component'] = ppxf_output.component
            self.results[idx]['ncomp'] = ppxf_output.ncomp
            self.results[idx]['fixall'] = ppxf_output.fixall
            self.results[idx]['moments'] = ppxf_output.moments
            self.results[idx]['goodpixels'] = ppxf_output.goodpixels
            self.results[idx]['bias'] = ppxf_output.bias
            self.results[idx]['nsky'] = ppxf_output.nsky
            self.results[idx]['ngh'] = ppxf_output.ngh
            self.results[idx]['npars'] = ppxf_output.npars
            self.results[idx]['A_eq_templ'] = ppxf_output.A_eq_templ
            self.results[idx]['b_eq_templ'] = ppxf_output.b_eq_templ
            self.results[idx]['A_ineq_templ'] = ppxf_output.A_ineq_templ
            self.results[idx]['b_ineq_templ'] = ppxf_output.b_ineq_templ
            self.results[idx]['A_ineq_kinem'] = ppxf_output.A_ineq_kinem
            self.results[idx]['b_ineq_kinem'] = ppxf_output.b_ineq_kinem
            self.results[idx]['A_eq_kinem'] = ppxf_output.A_eq_kinem
            self.results[idx]['b_eq_kinem'] = ppxf_output.b_eq_kinem
            self.results[idx]['npad'] = ppxf_output.npad
            #self.results[idx]['templates_rfft'] = ppxf_output.templates_rfft
            self.results[idx]['weights'] = ppxf_output.weights
            self.results[idx]['bestfit'] = ppxf_output.bestfit
            self.results[idx]['matrix'] = ppxf_output.matrix
            self.results[idx]['mpoly'] = ppxf_output.mpoly
            self.results[idx]['gas_mpoly'] = ppxf_output.gas_mpoly
            self.results[idx]['njev'] = ppxf_output.njev
            self.results[idx]['ndof'] = ppxf_output.ndof
            self.results[idx]['chi2'] = ppxf_output.chi2
            self.results[idx]['sol'] = ppxf_output.sol
            self.results[idx]['error'] = ppxf_output.error
            self.results[idx]['mpolyweights'] = ppxf_output.mpolyweights
            self.results[idx]['polyweights'] = ppxf_output.polyweights
            self.results[idx]['apoly'] = ppxf_output.apoly

            for j in range(np.shape(pix)[0]):
                idx_full_list.append(idx)
                pix_x.append(pix[j][0])
                pix_y.append(pix[j][1])
                ra, dec = self.pix_to_wcs[pix[j][0], pix[j][1]]
                pix_ra.append(ra)
                pix_dec.append(dec)

        self.idx = self.results.keys()
        self.idx_full_list = np.array(idx_full_list)
        self.pix_x = np.array(pix_x)
        self.pix_y = np.array(pix_y)
        self.pix_ra = np.array(pix_ra)
        self.pix_dec = np.array(pix_dec)

    def display_velscale(self, fig_type='png'):

        self.velscale = [
            self.results[i]['velscale'] for i in self.idx_full_list
        ]
        plt.figure(1)
        plt.clf()
        plot_velfield(self.pix_ra,
                      self.pix_dec,
                      self.velscale,
                      nodots=True,
                      colorbar=True,
                      origin='lower')
        plt.title('velscale')
        ax = plt.gca()
        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'velscale.' + fig_type))

    def display_flux(self, wave=[6553, 6573], fig_type='png'):

        self.flux = [
            np.sum(
                self.results[i]['flux'][(self.results[i]['wave'] > wave[0])
                                        & (self.results[i]['wave'] < wave[1])])
            for i in self.idx_full_list
        ]
        plt.figure(2)
        plt.clf()
        plot_velfield(self.pix_ra,
                      self.pix_dec,
                      self.flux,
                      nodots=True,
                      colorbar=True,
                      origin='lower')
        plt.title('Flux between {} and {} A'.format(wave[0], wave[1]))
        ax = plt.gca()
        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'flux.' + fig_type))

    def display_chi2(self, fig_type='png'):

        self.chi2 = [self.results[i]['chi2'] for i in self.idx_full_list]
        plt.figure(3)
        plt.clf()
        plot_velfield(self.pix_ra,
                      self.pix_dec,
                      self.chi2,
                      nodots=True,
                      colorbar=True,
                      origin='lower')
        plt.title('chi2')
        ax = plt.gca()
        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'chi2.' + fig_type))

    def display_sfh(self, fig_type='png'):

        self.sfh = np.column_stack([
            np.sum(self.results[i]['weights']
                   [~self.results[i]['gas_component']].reshape(
                       self.results[i]['reg_dim']),
                   axis=1) for i in self.idx_full_list
        ])
        if self.age is None:
            self.age = self.age_backup
        for i in range(len(self.sfh)):
            plt.figure(i)
            plt.clf()
            plot_velfield(self.pix_ra,
                          self.pix_dec,
                          self.sfh[i],
                          nodots=True,
                          colorbar=True,
                          origin='lower')
            plt.title('SFR at {} Gyr'.format(self.age[i]))
            ax = plt.gca()
            ax.invert_xaxis()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.plots_folder,
                    'sfh_{}_normalised_per_spexel.'.format(i) + fig_type))

    def display_gas_flux(self, fig_type='png'):

        gas_names = np.unique(
            np.concatenate(
                [self.results[i]['gas_names'] for i in range(len(self.idx))]))

        self.gas_flux = np.zeros((len(self.idx_full_list), len(gas_names)))

        for i, idx in enumerate(self.idx_full_list):
            for j, gn in enumerate(gas_names):
                try:
                    self.gas_flux[i][j] = self.results[idx]['gas_flux'][
                        np.where(self.results[idx]['gas_names'] == gn)]
                except Exception:
                    pass

        self.gas_flux[~np.isfinite(self.gas_flux)] = 0.0

        for j, gn in enumerate(gas_names):
            plt.figure(j + 100)
            plt.clf()
            plot_velfield(self.pix_ra,
                          self.pix_dec,
                          self.gas_flux[:, j],
                          nodots=True,
                          colorbar=True,
                          origin='lower')
            plt.title(gn)
            ax = plt.gca()
            ax.invert_xaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_folder, gn + '.' + fig_type))

    def compute_luminosity(self, z=0.):

        filter_r = sfilters.load_filters('sdss2010-r')
        filter_V = sfilters.load_filters('bessell-V')

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        D_L = cosmo.luminosity_distance(z).value * 1e6

        # The is the number of solar luminosity
        luminosity = {}

        for i, idx in enumerate(self.idx_full_list):
            luminosity[i] = {}
            try:
                mag_r = filter_r.get_ab_magnitudes(
                    self.results[idx]['flux'] * u.erg /
                    (u.cm**2 * u.s * u.Angstrom) * 1e-17,
                    self.results[idx]['wave'] *
                    u.Angstrom)['sdss2010-r'].value[0]
                luminosity[i]['r'] = 4. * np.pi * 10.**(
                    (4.65 - mag_r) / 2.5) * D_L
            except Exception:
                luminosity[i]['r'] = 0.

            try:
                mag_V = filter_V.get_ab_magnitudes(
                    self.results[idx]['flux'] * u.erg /
                    (u.cm**2 * u.s * u.Angstrom) * 1e-17,
                    self.results[idx]['wave'] *
                    u.Angstrom)['bessell-V'].value[0]
                luminosity[i]['V'] = 4. * np.pi * 10.**(
                    (4.80 - mag_V) / 2.5) * D_L
            except Exception:
                luminosity[i]['V'] = 0.

        self.luminosity = luminosity

    def compute_ml_ratio(self):

        ml_ratio = {}

        for i, idx in enumerate(self.idx_full_list):

            ml_ratio[i] = {}

            try:
                ml_ratio[i]['r'] = self.mass_to_light(
                    self.results[idx]['weights']
                    [~self.results[idx]['gas_component']].reshape(
                        self.results[idx]['reg_dim']),
                    band="r",
                    quiet=True)

            except Exception:

                ml_ratio[i]['r'] = 0.

            # Conversion taken from
            # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
            try:
                ml_ratio[i]['V'] = self.mass_to_light(
                    self.results[idx]['weights']
                    [~self.results[idx]['gas_component']].reshape(
                        self.results[idx]['reg_dim']),
                    band="V",
                    quiet=True) + 0.02

            except Exception:

                ml_ratio[i]['V'] = 0.

        self.ml_ratio = ml_ratio

    def display_sfh_by_mass(self, z=0., fig_type='png'):

        if self.luminosity is None:

            self.compute_luminosity(z=z)

        if self.ml_ratio is None:

            self.compute_ml_ratio()

        if self.sfh is None:

            self.display_sfh()

        for i in range(len(self.sfh)):

            for j, fi in enumerate(self.luminosity[0]):

                sfh_mass = self.sfh[i] * [
                    self.luminosity[i][fi] for i in range(len(self.luminosity))
                ] * [self.ml_ratio[i][fi] for i in range(len(self.ml_ratio))]

                plt.figure(i * len(self.ml_ratio) + j)
                plt.clf()
                plot_velfield(self.pix_ra,
                              self.pix_dec,
                              sfh_mass,
                              nodots=True,
                              colorbar=True,
                              origin='lower')
                plt.title('SFR at {} Gyr'.format(self.age[i]))
                ax = plt.gca()
                ax.invert_xaxis()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.plots_folder,
                        'sfh_{}_by_mass_in_filter_{}.'.format(i, fi) +
                        fig_type))
