import glob
import os

from astropy.io import fits
import numpy as np
from matplotlib.pyplot import *

if "__file__" in locals():
    HERE = os.path.dirname(os.path.abspath(__file__))
else:
    HERE = os.path.dirname(os.path.abspath(__name__))


pipe3d_folder_path = r"../../../manga/data.sdss.org/sas/dr17/spectro/pipe3d/v3_1_1/3.1.1/*/"

# test locally use this
#pipe3d_folder_path = r"8078"

folder_list = np.sort(glob.glob(pipe3d_folder_path))

# DESC_156= 'Luminosity Fraction for age 0.0010 SSP'
# DESC_157= 'Luminosity Fraction for age 0.0030 SSP'
# DESC_158= 'Luminosity Fraction for age 0.0040 SSP'
# DESC_159= 'Luminosity Fraction for age 0.0056 SSP'
# DESC_160= 'Luminosity Fraction for age 0.0089 SSP'
# DESC_161= 'Luminosity Fraction for age 0.0100 SSP'
# DESC_162= 'Luminosity Fraction for age 0.0126 SSP'
# DESC_163= 'Luminosity Fraction for age 0.0141 SSP'
# DESC_164= 'Luminosity Fraction for age 0.0178 SSP'
# DESC_165= 'Luminosity Fraction for age 0.0199 SSP'
# DESC_166= 'Luminosity Fraction for age 0.0251 SSP'
# DESC_167= 'Luminosity Fraction for age 0.0316 SSP'
# DESC_168= 'Luminosity Fraction for age 0.0398 SSP'
# DESC_169= 'Luminosity Fraction for age 0.0562 SSP'
# DESC_170= 'Luminosity Fraction for age 0.0630 SSP'
# DESC_171= 'Luminosity Fraction for age 0.0631 SSP'
# DESC_172= 'Luminosity Fraction for age 0.0708 SSP'
# DESC_173= 'Luminosity Fraction for age 0.1000 SSP'
# DESC_174= 'Luminosity Fraction for age 0.1122 SSP'
# DESC_175= 'Luminosity Fraction for age 0.1259 SSP'
# DESC_176= 'Luminosity Fraction for age 0.1585 SSP'
# DESC_177= 'Luminosity Fraction for age 0.1995 SSP'
# DESC_178= 'Luminosity Fraction for age 0.2818 SSP'
# DESC_179= 'Luminosity Fraction for age 0.3548 SSP'
# DESC_180= 'Luminosity Fraction for age 0.5012 SSP'
# DESC_181= 'Luminosity Fraction for age 0.7079 SSP'
# DESC_182= 'Luminosity Fraction for age 0.8913 SSP'
# DESC_183= 'Luminosity Fraction for age 01.1220 SSP'
# DESC_184= 'Luminosity Fraction for age 01.2589 SSP'
# DESC_185= 'Luminosity Fraction for age 01.4125 SSP'
# DESC_186= 'Luminosity Fraction for age 01.9953 SSP'
# DESC_187= 'Luminosity Fraction for age 02.5119 SSP'
# DESC_188= 'Luminosity Fraction for age 03.5481 SSP'
# DESC_189= 'Luminosity Fraction for age 04.4668 SSP'
# DESC_190= 'Luminosity Fraction for age 06.3096 SSP'
# DESC_191= 'Luminosity Fraction for age 07.9433 SSP'
# DESC_192= 'Luminosity Fraction for age 10.000 SSP'
# DESC_193= 'Luminosity Fraction for age 12.5893 SSP'
# DESC_194= 'Luminosity Fraction for age 14.1254 SSP'

age_list = np.array([0.0010, 0.0030, 0.0040, 0.0056, 0.0089, 0.0100, 0.0126,
    0.0141, 0.0178, 0.0199, 0.0251, 0.0316, 0.0398, 0.0562, 0.0630, 0.0631,
    0.0708, 0.1000, 0.1122, 0.1259, 0.1585, 0.1995, 0.2818, 0.3548, 0.5012,
    0.7079, 0.8913, 1.1220, 1.2589, 1.4125, 1.9953, 2.5119, 3.5481, 4.4668,
    6.3096, 7.9433, 10.000, 12.5893, 14.1254])
age_idx = np.arange(155, 195).astype('int')




if os.path.exists("sfh_pipe3d_voronoi_binned.npy"):
    sfh_voronoi_binned = np.load("sfh_pipe3d_voronoi_binned.npy")

else:

    sfh_voronoi = []

    # Loop through the folders of galaxies here
    for folder_path in folder_list:

        file_path_list = glob.glob(os.path.join(folder_path, "*.gz"))

        # Loop through the galaxies here
        for file_path in file_path_list:

            data = fits.open(file_path)
            # >>> data[0].header['EXTNAME'] 
            # 'ORG_HDR'
            # >>> data[1].header['EXTNAME'] 
            # 'SSP'
            # >>> data[2].header['EXTNAME'] 
            # 'SFH'
            # >>> data[3].header['EXTNAME'] 
            # 'FLUX_ELINES'
            # >>> data[4].header['EXTNAME'] 
            # 'INDICES'

            # star formation history here
            sfh = data[2].data

            # voronoi ID
            voronoi_id = data[1].data[1]
            voronoi_id_flattened = np.sort(voronoi_id.flatten()).astype('int')
            voronoi_id_list, voronoi_counts = np.unique(voronoi_id_flattened, return_counts=True)
            voronoi_id_list = voronoi_id_list[1:]
            voronoi_counts = voronoi_counts[1:]

            # dust corrected total mass (m_Sun/spaxels^2) <- multiply by the # of spexels
            mass_dust_corrected = data[1].data[19]

            for vid, cnts in zip(voronoi_id_list, voronoi_counts):
                # get the array position
                _x, _y = np.column_stack(np.where(vid == voronoi_id))[0]
                sfh_voronoi_i = sfh[155:195][:, _x, _y] * cnts
                sfh_voronoi.append(sfh_voronoi_i)

    # save after looping through all the plates and galaxies
    np.save("sfh_pipe3d_voronoi_binned.npy", sfh_voronoi)

    sfh_voronoi_binned = sfh_voronoi
