from ppxf_reader import ppxf_reader
from matplotlib import pyplot as plt

pr = ppxf_reader()
pr.load('../ppxf/manga-7495-12704-LOGCUBE-VOR10-GAU-MILESHC', verbose=False)
pr.display_gas_flux()
pr.display_velscale()
pr.display_chi2()
pr.display_flux()
pr.display_sfh()
pr.display_sfh_by_mass(z=0.02894)
