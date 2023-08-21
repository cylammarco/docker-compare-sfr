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


# Get the SFH from the firefly data
data_firefly = fits.open("firefly/manga-firefly-v3_1_1-miles.fits.gz")

firefly_z = data_firefly[1].data["REDSHIFT"]

galaxy_info = data_firefly["GALAXY_INFO"].data
firefly_voronoi_id_list = data_firefly["SPATIAL_BINID"].data
# np.shape(firefly_voronoi_id_list)
# (10735, 80, 80)
firefly_spatial_info_list = data_firefly["SPATIAL_INFO"].data
# np.shape(firefly_spatial_info_list)
# (10735, 2800, 5)
n_firefly = len(firefly_spatial_info_list)

sn_list = []
N_sn = 0
for i, spatial in enumerate(firefly_spatial_info_list):
    print(f"Galaxy {i+1} of {n_firefly}.")
    # Get the central (RA, Dec), i.e. position where voronoi_id = 0
    galaxy_plate_ifu_i = galaxy_info[i]["PLATEIFU"]
    if np.in1d(galaxy_plate_ifu_i, manga_sn_plate_ifu):
        idx = np.where(galaxy_plate_ifu_i == manga_sn_plate_ifu)[0][0]
        galaxy_ra_i = galaxy_info[i]["OBJRA"]
        galaxy_dec_i = galaxy_info[i]["OBJDEC"]
        # get the ra, dec offset to each spexel
        ra_offset_i = spatial[:, 1] / 3600.0 * np.cos(np.deg2rad(galaxy_dec_i))
        dec_offset_i = spatial[:, 2] / 3600.0
        spexel_ra = ra_offset_i + galaxy_ra_i
        spexel_dec = dec_offset_i + galaxy_dec_i
        distance = (spexel_ra - manga_sn_ra[idx]) ** 2.0 + (
            spexel_dec - manga_sn_dec[idx]
        ) ** 2.0
        vor_sn_id = np.argmin(distance)
    else:
        vor_sn_id = None
    voronoi_id_list = np.array(list(set(spatial[:, 0]))).astype("int")
    voronoi_id_list = np.sort(voronoi_id_list[voronoi_id_list >= 0])
    for voronoi_id in voronoi_id_list:
        if vor_sn_id == voronoi_id:
            sn_list.append(1)
        else:
            sn_list.append(0)


np.save(
    os.path.join("sn_list_firefly_asassn"),
    sn_list,
)
