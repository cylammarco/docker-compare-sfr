import numpy as np
import os
import glob
from plotbin.plot_velfield import plot_velfield
from matplotlib import pyplot as plt


class ppxf_reader:
    def __init__(self):

        pass

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
            'manga_' + '_'.join(self.ppxf_output_folder.split('-')[1:3]) +
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

    def display_vel(self, fig_type='png'):

        velscale = [self.results[i]['velscale'] for i in self.idx_full_list]
        plt.figure(1)
        plot_velfield(self.pix_x, self.pix_y, velscale)
        plt.savefig(os.path.join(self.plots_folder, 'velscale.' + fig_type))
