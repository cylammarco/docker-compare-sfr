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

filelist = glob.glob("tns-sn1a-20230416/tns_search*.csv")
_file = []
for filename in filelist:
    _file.append(pandas.read_csv(filename))

sn_table = pandas.concat(_file)

sn_table["Discovery Date (UT)"] = sn_table["Discovery Date (UT)"].astype(
    "datetime64"
)


sn_table.set_index("Discovery Date (UT)", inplace=True)
histogram_all_sn = sn_table.resample("1Y").size()

red_shift = sn_table["Redshift"].values
host_red_shift = sn_table["Host Redshift"].values
red_shift_mask = (red_shift <= 0.125) | (host_red_shift <= 0.125)

sn_table_low_redshift = sn_table[red_shift_mask]
sn_table_basic_filtered = sn_table

vizier_results = []
v = Vizier(columns=["**"], catalog="J/AJ/154/86")
for ra, dec in zip(
    sn_table_basic_filtered["RA"].values, sn_table_basic_filtered["DEC"].values
):
    try:
        pos = coords.SkyCoord(ra, dec, frame="icrs", unit=(u.hourangle, u.deg))
        # This is to get everything that can fit into a 127-Fiber IFU
        res = v.query_region(pos, radius=Angle(32.0, "arcsec"))
        if len(res) > 0:
            vizier_results.append(res)
        else:
            vizier_results.append(None)
    except:
        vizier_results.append(None)

vizier_results = np.array(vizier_results)
vizier_mask = np.array(vizier_results) != None
manga_catalogue = vizier_results[vizier_mask]
manga_sample = sn_table_basic_filtered[vizier_mask]


fibre_to_sep = {
    "7": 7.0,
    "19": 12.0,
    "37": 17.0,
    "61": 22.0,
    "91": 27.0,
    "127": 32.0,
}

IFU_mask = []
# Filter based on the IFU-size returned from Vizier
for i, manga in enumerate(manga_catalogue):
    pos_galaxy = coords.SkyCoord(
        manga_sample["RA"].values[i],
        manga_sample["DEC"].values[i],
        frame="icrs",
        unit=(u.hourangle, u.deg),
    )
    matched_ifudsgn = manga[0]["IFUdsgn"].value.data
    ra = manga[0]["RAJ2000"].value.data
    dec = manga[0]["DEJ2000"].value.data
    _contain_sn = False
    for j, ifu in enumerate(matched_ifudsgn):
        n_fibre = ifu[:-2]
        pos_sn = coords.SkyCoord(
            ra[j], dec[j], frame="icrs", unit=(u.deg, u.deg)
        )
        if pos_galaxy.separation(pos_sn) <= fibre_to_sep[n_fibre] * u.arcsec:
            _contain_sn = True
    if _contain_sn:
        IFU_mask.append(True)
    else:
        IFU_mask.append(False)


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


# Apply the mask and then create a histogram
histogram_manga_sn = manga_sample[IFU_mask].resample("1Y").size()
histogram_manga_sn_2y = manga_sample[IFU_mask].resample("2Y").size()

PTF_start = date(year=2009, month=6, day=1)
PTF_end = date(year=2013, month=1, day=1)
iPTF_start = date(year=2013, month=1, day=1)
iPTF_end = date(year=2016, month=1, day=1)
ZTF_start = date(year=2018, month=6, day=1)
ZTF_end = date(year=2023, month=4, day=1)

time = np.arange(1998, 2023, 2)
log_n_sn = np.log10(histogram_manga_sn_2y[:-1] / 2.0)

mask = log_n_sn > 0

coeff = np.polyfit(time[mask], log_n_sn[mask], 1)
predicted = 10.0 ** np.polyval(coeff, time)

plt.figure(1, figsize=(8, 8))
plt.clf()
plt.plot(histogram_all_sn, label="No. of SN Ia in that year")
plt.plot(
    histogram_manga_sn, label="No. of SN Ia in that year in the MaNGA sample"
)
plt.plot(
    [date(year=i, month=1, day=1) for i in range(1935, 2024)],
    10.0 ** np.polyval(coeff, np.arange(1935, 2024)),
    label="Fitted trend",
)
plt.axvspan(PTF_start, PTF_end, alpha=0.2, color="red", label="PTF")
plt.axvspan(iPTF_start, iPTF_end, alpha=0.2, color="green", label="iPTF")
plt.axvspan(ZTF_start, ZTF_end, alpha=0.2, color="blue", label="ZTF")
plt.xlabel("Year")
plt.ylabel("Number of SN Ia Discovered")
plt.yscale("log")
plt.title("Number of SN Ia Discovery")
plt.legend()
plt.tight_layout()
plt.grid()
plt.xlim(
    left=date(year=1935, month=1, day=1), right=date(year=2023, month=4, day=1)
)
plt.savefig("sn1a_rate.png")
