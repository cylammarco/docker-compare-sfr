import os

import numpy as np
import matplotlib.pyplot as plt
from pyFIT3D.common.auto_ssp_tools import auto_ssp_elines_rnd
from pyFIT3D.common.io import get_data_from_fits
from pyFIT3D.common.io import get_wave_from_header
from pyFIT3D.common.constants import __c__
from pyFIT3D.common.io import output_spectra
from pyFIT3D.common.stats import pdl_stats, _STATS_POS
from pyFIT3D.common.gas_tools import fit_elines
from pyFIT3D.common.gas_tools import read_fit_elines_output

os.chdir('pipe3d/')
data_path = '../example/pipe3d_example_data/'

#########################################################################
#
# --= pyFIT3D Auto SSP analysis =--
#
# Let's run the analysis of the central (5"x5") spectra of galaxy NGC5947
#
# See the similar script to run the analysis with intrumental dispersion:
#      bin/auto_ssp_elines_rnd_sigma_inst.py
#

name = 'NGC5947'
spec_file = data_path+f'{name}.spec_5.txt'

# using the initial values:
#
# Instrumental dispersion: 2.6 AA
sigma_inst = 2.6

# Masks:
#     File with list of ranges of wavelength to be masked in whole analysis
mask_list = data_path+'mask_elines.txt'

######################
######################
# Non linear analysis: (redshift, observed dispersion and dust attenuation)
#     wavelength range:
#         [3800, 4700] AA for the redshift and sigma analysis
#         [3800, 7000] AA for the dust attenuation
w_min_max = [3800, 7000]
nl_w_min_max = [3800, 4700]

#     models:
#         GSD01 3 models
ssp_nl_fit_file = data_path+'gsd01_3.fits'

#     emission lines to be masked:
elines_mask_file = data_path+'emission_lines.txt'

#     redshift:
#         initial value: 0.0195
#                 delta: 0.0001
#                   min: 0.0170
#                   max: 0.0225
redshift_set = [0.0195, 0.0001, 0.0170, 0.0225]

#     dispersion: (km/s)
#         initial value: 30
#                 delta: 20
#                   min: 1
#                   max: 350
sigma_set = [30, 20, 1, 350]

#     dust attenuation at V band (mag):
#         initial value: 0.3
#                 delta: 0.15
#                   min: 0
#                   max: 1.6
AV_set = [0.1, 0.05, 0, 2]

# SSP analysis:
#     wavelength range:
#         [3800, 7000] AA
#
#     models:
#         GSD01 156 models
ssp_file = data_path+'gsd01_156.fits'

#     configuration file:
config_file = data_path+'auto_ssp_V500_several_Hb.config'

# final output file:
out_file = f'auto_ssp.{name}.cen.out'

# run auto_ssp_elines_rnd_sigma_inst
auto_ssp_elines_rnd(spec_file=spec_file,
                    ssp_file=ssp_file,
                    ssp_nl_fit_file=ssp_nl_fit_file,
                    sigma_inst=sigma_inst,
                    out_file=out_file,
                    config_file=config_file,
                    mask_list=mask_list,
                    elines_mask_file=elines_mask_file,
                    min=-3,
                    max=50,
                    w_min=w_min_max[0],
                    w_max=w_min_max[1],
                    nl_w_min=nl_w_min_max[0],
                    nl_w_max=nl_w_min_max[0],
                    input_redshift=redshift_set[0],
                    delta_redshift=redshift_set[1],
                    min_redshift=redshift_set[2],
                    max_redshift=redshift_set[3],
                    input_sigma=sigma_set[0],
                    delta_sigma=sigma_set[1],
                    min_sigma=sigma_set[2],
                    max_sigma=sigma_set[3],
                    input_AV=AV_set[0],
                    delta_AV=AV_set[1],
                    min_AV=AV_set[2],
                    max_AV=AV_set[3],
                    plot=0)

data__tw, h = get_data_from_fits(f'output.{out_file}.fits', header=True)

# define spectra and wavelength axis
wave__w = get_wave_from_header(h, wave_axis=1)

flux_org__w = data__tw[0]
model__w = data__tw[2]
res__w = data__tw[3]

# plot spectra
sel__w = (wave__w > 4000) & (wave__w < 7000)
plt.xlabel('Wavelength')
plt.ylabel('flux')
plt.plot(wave__w[sel__w], flux_org__w[sel__w], label='Input spectrum')
plt.plot(wave__w[sel__w], model__w[sel__w], label='Emission line spectrum')
plt.plot(wave__w[sel__w], res__w[sel__w], label='Best-fit spectrum')
plt.savefig('fig_1_output_spectra.jpg')

# Gas spectrum
gas__w = data__tw[3]

# [NII]+[Ha] system
wl_br = [6600, 6800]
wlb = wl_br[0]
wlr = wl_br[1]
sel__w = (wave__w > wlb) & (wave__w < wlr)

# The system velocity should be around:
obs_Ha_center = wave__w[sel__w][np.argmax(gas__w[sel__w])]
rest_Ha_center = 6562.82
sys_vel = __c__ * (obs_Ha_center - rest_Ha_center) / rest_Ha_center
print(f'sys_vel: {sys_vel}')

# output the [NII]+[Ha] spectrum
#

system_wave__w = wave__w[sel__w]
system_flux__w = gas__w[sel__w]

# simulate an input error
system_res__w = data__tw[4, sel__w]
st_res = pdl_stats(system_res__w)
system_eflux__w = 0.1 * st_res[_STATS_POS['pRMS']] * system_flux__w

# output spectra
filename = f'NII_Ha.{name}.txt'
output_spectra(system_wave__w, [system_flux__w, system_eflux__w],
               filename=filename)

#####################################################################
#
# --= pyFIT3D fit elines analysis =--
#
# Let's run the analysis of the NII+Ha system of the previous spectra

# models: Ha + [NII]6583 + [NII]6548 + continuum
#     config file with models to fit:
config_filename = data_path+'Ha_NII.config'
name = 'NGC5947'
# output filename
output_filename = f'fit_elines_rnd_NII_Ha.{name}.out'

fit_elines(
    spec_file=filename,
    config_file=config_filename,
    w_min=wlb,
    w_max=wlr,
    redefine_max=1,
    n_MC=20,
    n_loops=5,
    plot=0,
    scale_ini=0.15,
    out_file=output_filename,
    run_mode='RND',
)

n_models_rnd, chi_sq_rnd, elsystems = read_fit_elines_output(output_filename,
                                                             verbose=1)

models = elsystems[0]
i_Ha = 0
i_NII_red = 1
i_NII_blue = 2
i_cont = 0
flux = models['flux']
e_flux = models['e_flux']
v0 = models['v0']
e_v0 = models['e_v0']
disp = models['disp']
e_disp = models['e_disp']
cont = models['cont']
e_cont = models['e_cont']

# comparing the calculated system velocity and the measured one:
print(f'calc sys vel: {sys_vel}')
print(f"derived velocity: {models['v0'][i_Ha]}")

print(f'Number of models: {n_models_rnd} - fit chi squared: {chi_sq_rnd}')
print('Ha:')
print(f"\t integrated flux: {flux[i_Ha]:.4f} +/- {e_flux[i_Ha]:.4f}")
print(f"\t        velocity: {v0[i_Ha]:.4f} +/- {e_v0[i_Ha]:.4f}")
print(f"\t      dispersion: {disp[i_Ha]:.4f} +/- {e_disp[i_Ha]:.4f}")
print('[NII]6583:')
print(f"\t integrated flux: {flux[i_NII_red]:.4f} +/- {e_flux[i_NII_red]:.4f}")
print(f"\t        velocity: {v0[i_NII_red]:.4f} +/- {e_v0[i_NII_red]:.4f}")
print(f"\t      dispersion: {disp[i_NII_red]:.4f} +/- {e_disp[i_NII_red]:.4f}")
print('[NII]6548')
print(f"\t integrated flux: {flux[i_NII_red]:.4f} +/- {e_flux[i_NII_red]:.4f}")
print(f"\t        velocity: {v0[i_NII_red]:.4f} +/- {e_v0[i_NII_red]:.4f}")
print(f"\t      dispersion: {disp[i_NII_red]:.4f} +/- {e_disp[i_NII_red]:.4f}")
print('continuum:')
print(f"\t integrated flux: {cont[0]:.4f} +/- {e_cont[0]:.4f}")

'''
# Perform the fit again with Levemberg-Marquadt minimization technique
# using the rnd output values as input values for the models.
# file generated by fit_elines_rnd()
config_filename = data_path+'out_config.fit_spectra'

# output filename
output_filename = f'fit_elines_LM_NII_Ha.{name}.out'

fit_elines(
    spec_file=filename,
    config_file=config_filename,
    w_min=wlb,
    w_max=wlr,
    n_MC=20,
    n_loops=5,
    plot=0,
    scale_ini=0.15,
    out_file=output_filename,
    run_mode='LM',
)

n_models_LM, chi_sq_LM, elsystems = read_fit_elines_output(output_filename,
                                                           verbose=1)

models = elsystems[0]
i_Ha = 0
i_NII_red = 1
i_NII_blue = 2
i_cont = 0
flux = models['flux']
e_flux = models['e_flux']
v0 = models['v0']
e_v0 = models['e_v0']
disp = models['disp']
e_disp = models['e_disp']
cont = models['cont']
e_cont = models['e_cont']
print(f'Number of models: {n_models_rnd} - fit chi squared: {chi_sq_rnd}')
print('Ha:')
print(f"\t integrated flux: {flux[i_Ha]:.4f} +/- {e_flux[i_Ha]:.4f}")
print(f"\t        velocity: {v0[i_Ha]:.4f} +/- {e_v0[i_Ha]:.4f}")
print(f"\t      dispersion: {disp[i_Ha]:.4f} +/- {e_disp[i_Ha]:.4f}")
print('[NII]6583:')
print(f"\t integrated flux: {flux[i_NII_red]:.4f} +/- {e_flux[i_NII_red]:.4f}")
print(f"\t        velocity: {v0[i_NII_red]:.4f} +/- {e_v0[i_NII_red]:.4f}")
print(f"\t      dispersion: {disp[i_NII_red]:.4f} +/- {e_disp[i_NII_red]:.4f}")
print('[NII]6548')
print(f"\t integrated flux: {flux[i_NII_red]:.4f} +/- {e_flux[i_NII_red]:.4f}")
print(f"\t        velocity: {v0[i_NII_red]:.4f} +/- {e_v0[i_NII_red]:.4f}")
print(f"\t      dispersion: {disp[i_NII_red]:.4f} +/- {e_disp[i_NII_red]:.4f}")
print('continuum:')
print(f"\t integrated flux: {cont[0]:.4f} +/- {e_cont[0]:.4f}")
'''

# This dual process of fitting with rnd and after with the LM method
# could be mimic by using the mode 'both':
# models: Ha + [NII]6583 + [NII]6548 + continuum
#     config file with models to fit:
config_filename = data_path+'Ha_NII.config'

# output filename
output_filename = f'fit_elines_mixed_NII_Ha.{name}.out'

fit_elines(spec_file=filename,
           config_file=config_filename,
           w_min=wlb,
           w_max=wlr,
           redefine_max=1,
           n_MC=20,
           n_loops=5,
           plot=0,
           scale_ini=0.15,
           out_file=output_filename,
           run_mode='both')

n_models_both, chi_sq_both, elsystems = read_fit_elines_output(output_filename,
                                                               verbose=1)

models = elsystems[0]
i_Ha = 0
i_NII_red = 1
i_NII_blue = 2
i_cont = 0
flux = models['flux']
e_flux = models['e_flux']
v0 = models['v0']
e_v0 = models['e_v0']
disp = models['disp']
e_disp = models['e_disp']
cont = models['cont']
e_cont = models['e_cont']
print(f'Number of models: {n_models_rnd} - fit chi squared: {chi_sq_rnd}')
print('Ha:')
print(f"\t integrated flux: {flux[i_Ha]:.4f} +/- {e_flux[i_Ha]:.4f}")
print(f"\t        velocity: {v0[i_Ha]:.4f} +/- {e_v0[i_Ha]:.4f}")
print(f"\t      dispersion: {disp[i_Ha]:.4f} +/- {e_disp[i_Ha]:.4f}")
print('[NII]6583:')
print(f"\t integrated flux: {flux[i_NII_red]:.4f} +/- {e_flux[i_NII_red]:.4f}")
print(f"\t        velocity: {v0[i_NII_red]:.4f} +/- {e_v0[i_NII_red]:.4f}")
print(f"\t      dispersion: {disp[i_NII_red]:.4f} +/- {e_disp[i_NII_red]:.4f}")
print('[NII]6548')
print(f"\t integrated flux: {flux[i_NII_red]:.4f} +/- {e_flux[i_NII_red]:.4f}")
print(f"\t        velocity: {v0[i_NII_red]:.4f} +/- {e_v0[i_NII_red]:.4f}")
print(f"\t      dispersion: {disp[i_NII_red]:.4f} +/- {e_disp[i_NII_red]:.4f}")
print('continuum:')
print(f"\t integrated flux: {cont[0]:.4f} +/- {e_cont[0]:.4f}")
