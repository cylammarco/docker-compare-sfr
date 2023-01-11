from datetime import date
import glob
import re

from astropy import coordinates as coords
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import pandas
from matplotlib import pyplot as plt
import numpy as np

filelist = glob.glob("tns-sn1a-20221012/tns_search*.csv")
_file = []
for filename in filelist:
    _file.append(pandas.read_csv(filename))

sn_table = pandas.concat(_file)

sn_table["Discovery Date (UT)"] = sn_table["Discovery Date (UT)"].astype(
    "datetime64"
)


sn_table.set_index("Discovery Date (UT)", inplace=True)
histogram_all_sn = sn_table.resample("1Y").size()

red_shift = sn_table["Host Redshift"].values
red_shift_mask = (red_shift > 0.0)

sn_table_basic_filtered = sn_table[red_shift_mask]

vizier_results = []

for ra, dec in zip(sn_table_basic_filtered['RA'].values, sn_table_basic_filtered['DEC'].values):
    try:
        pos = coords.SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
        res = Vizier.query_region(pos, radius=Angle(0.5, "deg"), catalog='J/AJ/154/86')
        if len(res) > 0:
            vizier_results.append(res)
        else:
            vizier_results.append(None)
    except:
        vizier_results.append(None)

vizier_results = np.array(vizier_results)
vizier_mask = (np.array(vizier_results) != None)


manga_sample = sn_table_basic_filtered[vizier_mask]

histogram_manga_sn = manga_sample.resample("1Y").size()


"""
# Selection criteria
# https://www.sdss4.org/dr17/manga/manga-target-selection/
# All absolute magnitudes and stellar mass in the NSA catalog are computed assuming h=1.
# h = cosmo.H(0).value / 100.0
h = 1.0

A_upper = -0.056597
B_upper = -0.0039264
C_upper = -2.9119
D_upper = -22.8476

A_lower = -0.011377
B_lower = -0.0019220
C_lower = -1.2924
D_lower = -22.1610


z_upper = (A_upper + B_upper * (Mi - 5.0 * np.log10(h))) * (
    1.0 + np.exp(C_upper * (Mi - 5.0 * np.log10(h) - D_upper))
)
z_lower = (A_lower + B_lower * (Mi - 5.0 * np.log10(h))) * (
    1.0 + np.exp(C_lower * (Mi - 5.0 * np.log10(h) - D_lower))
)
"""

PTF_start = date(year=2009, month=6, day=1)
PTF_end = date(year=2013, month=1, day=1)
iPTF_start = date(year=2013, month=1, day=1)
iPTF_end = date(year=2016, month=1, day=1)
ZTF_start = date(year=2018, month=6, day=1)
ZTF_end = date(year=2023, month=1, day=1)


plt.figure(1, figsize=(8, 8))
plt.clf()
plt.plot(histogram_all_sn, label="No. of SNe in that year")
plt.plot(histogram_manga_sn, label="No. of SNe in that year in the MaNGA sample")
plt.axvspan(PTF_start, PTF_end, alpha=0.2, color="red", label="PTF")
plt.axvspan(iPTF_start, iPTF_end, alpha=0.2, color="green", label="iPTF")
plt.axvspan(ZTF_start, ZTF_end, alpha=0.2, color="blue", label="ZTF")
plt.xlabel("Year")
plt.ylabel("Number of SN Ia discovered")
plt.yscale("log")
plt.title("Number of SN Ia Discovery")
plt.legend()
plt.tight_layout()
plt.grid()
plt.xlim(right=date(year=2023, month=1, day=1))
plt.savefig("sn1a_rate.png")
