import os
import glob

from matplotlib import pyplot as plt
import numpy as np
from plotbin.plot_velfield import plot_velfield
import ppxf as ppxf_package
import ppxf.miles_util as lib


class ppxf_reader:
    def __init__(self):

        ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
        pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

        # Only getting the age from the miles library, the values in the
        # second and third arguments are random.
        miles = lib.miles(pathname, 100, 2.5)

        self.age = miles.age_grid[:, 0]

    def load_file(self, ppxf_output_filename):

        idx = int(ppxf_output_filename.split('.')[0].split('_')[-1])
        ppxf_output = np.load(ppxf_output_filename, allow_pickle=True).item()
        pix = self.idx_to_pix[idx]

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
        '''
        self.ppxf_sfh_folder = os.path.join(self.ppxf_output_folder, 'sfh')
        self.ppxf_fitted_model_folder = os.path.join(self.ppxf_output_folder,
                                                     'fitted_model')
        self.ppxf_weights_folder = os.path.join(self.ppxf_output_folder, 'weights')
        '''
        self.plots_folder = os.path.join(self.ppxf_output_folder,
                                         'diagnostic_plots')

        if not os.path.exists(self.plots_folder):
            os.mkdir(self.plots_folder)

        # Get the voronoi idx to pixel position
        self.idx_to_pix_path = os.path.join(
            self.ppxf_output_folder,
            'manga_' + '_'.join(self.folder_name.split('-')[1:3]) +
            '_ppxf_idx_to_pix.npy')
        self.idx_to_pix = np.load(self.idx_to_pix_path,
                                  allow_pickle=True).item()

        idx_full_list = []
        pix_x = []
        pix_y = []
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
            self.results[idx]['flux'] = ppxf_output.__dict__['galaxy']
            self.results[idx]['flux_err'] = ppxf_output.__dict__['noise']
            self.results[idx]['vsyst'] = ppxf_output.__dict__['vsyst']
            self.results[idx]['regul'] = ppxf_output.__dict__['regul']
            self.results[idx]['wave'] = ppxf_output.__dict__['lam']
            self.results[idx]['reg_dim'] = ppxf_output.__dict__['reg_dim']
            self.results[idx]['templates'] = ppxf_output.__dict__['templates']
            self.results[idx]['velscale'] = ppxf_output.__dict__['velscale']
            self.results[idx]['gas_flux'] = ppxf_output.__dict__['gas_flux']
            self.results[idx]['gas_flux_error'] = ppxf_output.__dict__[
                'gas_flux_error']
            self.results[idx]['gas_bestfit'] = ppxf_output.__dict__[
                'gas_bestfit']
            self.results[idx]['component'] = ppxf_output.__dict__['component']
            self.results[idx]['gas_component'] = ppxf_output.__dict__[
                'gas_component']
            self.results[idx]['gas_names'] = ppxf_output.__dict__['gas_names']
            self.results[idx]['gas_any'] = ppxf_output.__dict__['gas_any']
            self.results[idx]['ncomp'] = ppxf_output.__dict__['ncomp']
            self.results[idx]['weights'] = ppxf_output.__dict__['weights']
            self.results[idx]['bestfit'] = ppxf_output.__dict__['bestfit']
            self.results[idx]['ndof'] = ppxf_output.__dict__['ndof']
            self.results[idx]['chi2'] = ppxf_output.__dict__['chi2']
            self.results[idx]['sol'] = ppxf_output.__dict__['sol']
            self.results[idx]['error'] = ppxf_output.__dict__['error']

            for j in range(np.shape(pix)[0]):
                idx_full_list.append(idx)
                pix_x.append(pix[j][0])
                pix_y.append(pix[j][1])

        self.idx = self.results.keys()
        self.idx_full_list = np.array(idx_full_list)
        self.pix_x = np.array(pix_x)
        self.pix_y = np.array(pix_y)

    def display_velscale(self, fig_type='png'):

        self.velscale = [
            self.results[i]['velscale'] for i in self.idx_full_list
        ]
        plt.figure(1)
        plt.clf()
        plot_velfield(self.pix_x, self.pix_y, self.velscale, nodots=True, colorbar=True, origin='lower')
        plt.title('velscale')
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
        plot_velfield(self.pix_x, self.pix_y, self.flux, nodots=True, colorbar=True, origin='lower')
        plt.title('Flux between {} and {} A'.format(wave[0], wave[1]))
        plt.savefig(os.path.join(self.plots_folder, 'flux.' + fig_type))

    def display_chi2(self, fig_type='png'):

        self.chi2 = [self.results[i]['chi2'] for i in self.idx_full_list]
        plt.figure(3)
        plt.clf()
        plot_velfield(self.pix_x, self.pix_y, self.chi2, nodots=True, colorbar=True, origin='lower')
        plt.title('chi2')
        plt.savefig(os.path.join(self.plots_folder, 'chi2.' + fig_type))

    def display_sfh(self, fig_type='png'):

        self.sfh = np.column_stack([
            np.sum(
                self.results[i]['weights'][~self.results[i]['gas_component']].reshape(self.results[i]['reg_dim']),
                axis=1) for i in self.idx_full_list
        ])
        for i in range(len(self.sfh)):
            plt.figure(i)
            plt.clf()
            plot_velfield(self.pix_x, self.pix_y, self.sfh[i], nodots=True, colorbar=True, origin='lower')
            plt.title('SFR at {} Gyr'.format(self.age[i]))
            plt.savefig(os.path.join(self.plots_folder, 'sfh_{}.'.format(i) + fig_type))

    def display_gas_flux(self, fig_type='png'):

        gas_names = np.unique(
            np.concatenate(
                [self.results[i]['gas_names'] for i in range(len(self.idx))]))

        self.gas_flux = np.zeros((len(self.idx_full_list), len(self.idx)))

        for idx in self.idx_full_list:
            for j, gn in enumerate(gas_names):
                try:
                    self.gas_flux[idx][j] = self.results[idx]['gas_flux'][
                        np.where(self.results[idx]['gas_names'] == gn)]
                except Exception:
                    pass

        for j, gn in enumerate(gas_names):
            plt.figure(j + 100)
            plt.clf()
            plot_velfield(self.pix_x, self.pix_y, self.gas_flux[:, j], nodots=True, colorbar=True, origin='lower')
            plt.title(gn)
            plt.savefig(os.path.join(self.plots_folder, gn + '.' + fig_type))
