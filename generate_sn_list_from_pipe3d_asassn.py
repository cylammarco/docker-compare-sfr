import glob
import os
import sys

from astropy import units
from astropy.io import fits
import numpy as np
from scipy import interpolate as itp


# Get the crossmatched data
manga_sn = np.loadtxt(
    "manga_asassn_crossmatch.csv", delimiter=",", skiprows=1, dtype="object"
)
manga_sn_plate_ifu = np.array(
    ["-".join(i.split("-")[1:]) for i in manga_sn[:, 3]]
)
manga_sn_ra = manga_sn[:, 4].astype("float")
manga_sn_dec = manga_sn[:, 5].astype("float")


pipe3d_folder_path = (
    r"../../../manga/data.sdss.org/sas/dr17/spectro/pipe3d/v3_1_1/3.1.1/*/"
)

# test locally use this
# pipe3d_folder_path = r"8078"

folder_list = np.sort(glob.glob(pipe3d_folder_path))

sn_list = []
N_sn = 0
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

        galaxy_plate_ifu_i = data[0].header["PLATEIFU"]

        if np.in1d(galaxy_plate_ifu_i, manga_sn_plate_ifu):
            idx = np.where(galaxy_plate_ifu_i == manga_sn_plate_ifu)[0][0]

            # Get the voronoi ID
            voronoi_id = data[1].data[1]

            # get the number of pixels here
            n_pix_x, n_pix_y = np.shape(data[2].data[0])

            x_center = data[0].header["CRPIX1"]
            y_center = data[0].header["CRPIX2"]

            voronoi_center_x, voronoi_center_y = np.concatenate(
                np.where(voronoi_id == 1)
            )

            ra_center = data[0].header["CRVAL1"]
            dec_center = data[0].header["CRVAL2"]

            delta_ra = data[0].header["CD1_1"] * np.cos(np.deg2rad(dec_center))
            delta_dec = data[0].header["CD2_2"]

            # Get the RA, Dec of the first pixel
            ra_1 = ra_center - delta_ra * voronoi_center_x
            dec_1 = dec_center - delta_dec * voronoi_center_y

            # Get the RA, Dec of the last pixel
            ra_last = ra_1 + delta_ra * n_pix_x
            dec_last = dec_1 + delta_dec * n_pix_y

            distance = (spexel_ra - manga_sn_ra[idx]) ** 2.0 + (
                spexel_dec - manga_sn_dec[idx]
            ) ** 2.0
            vor_sn_id = np.argmin(distance)
        else:
            vor_sn_id = None
        voronoi_id = data[1].data[1]
        voronoi_id_flattened = np.sort(voronoi_id.flatten()).astype("int")
        voronoi_id_list = np.unique(voronoi_id_flattened)
        for voronoi_id in voronoi_id_list:
            if vor_sn_id == voronoi_id:
                sn_list.append(1)
            else:
                sn_list.append(0)


np.save(
    os.path.join("sn_list_pipe3d_asassn"),
    sn_list,
)
