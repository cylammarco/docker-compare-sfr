from datetime import date
import glob
import os
import re

from astropy import coordinates as coords
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


sn_table_1 = np.genfromtxt(
    f"ASASSN{os.sep}asassn_sn_list.txt",
    dtype=[
        ("Num", "<i4"),
        ("ID", "<U13"),
        ("Date", "<U15"),
        ("RA", "<f8"),
        ("Dec", "<f8"),
        ("Redshift", "<f8"),
        ("V_disc", "<f8"),
        ("V_abs", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U11"),
        ("Disc_Age", "<U11"),
        ("Classification_Age", "<U21"),
        ("Galaxy_Name", "<U26"),
    ],
    delimiter=[5, 13, 15, 11, 11, 10, 8, 8, 8, 11, 11, 21, 26],
    autostrip=True,
)

table_1_Ia_mask = ["Ia" in i for i in sn_table_1["Type"]]

dateobs = []
sn_name = []
ra = []
dec = []
types = []
sc = []
for i in sn_table_1[table_1_Ia_mask]:
    _date, _hour = i[2].split(".")
    _hour = float(_hour) / 10.0 ** (len(_hour)) * 24
    _minute = _hour % 1 * 60
    _second = _minute % 1 * 60
    _hour = int(_hour)
    _minute = int(_minute)
    _second = int(_second)
    dateobs.append(
        _date + " " + str(_hour) + ":" + str(_minute) + ":" + str(_second)
    )

d1 = {
    "Num": sn_table_1[table_1_Ia_mask]["Num"],
    "ID": sn_table_1[table_1_Ia_mask]["ID"],
    "dateobs": dateobs,
    "RA": sn_table_1[table_1_Ia_mask]["RA"],
    "DEC": sn_table_1[table_1_Ia_mask]["Dec"],
    "Redshift": sn_table_1[table_1_Ia_mask]["Redshift"],
    "V_disc": sn_table_1[table_1_Ia_mask]["V_disc"],
    "V_abs": sn_table_1[table_1_Ia_mask]["V_abs"],
    "Offset": sn_table_1[table_1_Ia_mask]["Offset"],
    "Type": sn_table_1[table_1_Ia_mask]["Type"],
    "Disc_Age": sn_table_1[table_1_Ia_mask]["Disc_Age"],
    "Classification_Age": sn_table_1[table_1_Ia_mask]["Classification_Age"],
    "Galaxy_Name": sn_table_1[table_1_Ia_mask]["Galaxy_Name"],
}
asas_sn_table = pd.DataFrame(data=d1)


asas_sn_table["dateobs"] = asas_sn_table["dateobs"].astype("datetime64")
asas_sn_table.set_index("dateobs", inplace=True)
histogram_all_asas_sn = asas_sn_table.resample("1Y").size()

filelist = glob.glob("tns-sn1a-20230416/tns_search*.csv")
_file = []
for filename in filelist:
    _file.append(pd.read_csv(filename))

tns_sn_table = pd.concat(_file)


tns_sn_table["Discovery Date (UT)"] = tns_sn_table[
    "Discovery Date (UT)"
].astype("datetime64")


tns_sn_table.set_index("Discovery Date (UT)", inplace=True)
histogram_all_tns_sn = tns_sn_table.resample("1Y").size()


plt.figure(1, figsize=(8, 8))
plt.clf()
plt.plot(
    histogram_all_asas_sn,
    label="No. of SN Ia in that year discovered by ASAS-SN",
)

plt.plot(
    histogram_all_tns_sn, label="No. of SN Ia in that year reported in TNS"
)
plt.xlabel("Year")
plt.ylabel("Number of SN Ia Discovered")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.grid()
plt.xlim(
    left=date(year=1935, month=1, day=1), right=date(year=2023, month=4, day=1)
)


vizier_results_1 = []
# PyPIPE3D DR17 J/ApJS/262/36
# DR15 J/AJ/154/86
# AMUSING++ J/AJ/159/167/table3

v1 = Vizier(columns=["**"], catalog="J/ApJS/262/36")
for ra, dec in zip(asas_sn_table["RA"].values, asas_sn_table["DEC"].values):
    try:
        pos = coords.SkyCoord(ra, dec, frame="icrs", unit=(u.deg, u.deg))
        # This is to get everything that can fit into a 127-Fiber IFU
        res = v1.query_region(pos, radius=Angle(32.0, "arcsec"))
        if len(res) > 0:
            vizier_results_1.append(res)
        else:
            vizier_results_1.append(None)
    except:
        vizier_results_1.append(None)

vizier_results_1 = np.array(vizier_results_1)
vizier_mask_1 = np.array(vizier_results_1) != None
manga_catalogue = vizier_results_1[vizier_mask_1]
manga_sample = asas_sn_table[vizier_mask_1]

_names_1 = np.array([i[0]["Name"].data[0] for i in manga_catalogue])
_ra_1 = np.array([i[0]["RAJ2000"].data[0] for i in manga_catalogue])
_dec_1 = np.array([i[0]["DEJ2000"].data[0] for i in manga_catalogue])
manga_sample.insert(2, "Name", _names_1, True)
manga_sample.insert(5, "RA_manga", _ra_1, True)
manga_sample.insert(6, "DEC_manga", _dec_1, True)

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
        unit=(u.deg, u.deg),
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


manga_final_catalogue = manga_catalogue[IFU_mask]
manga_final_sample = manga_sample[IFU_mask]


np.save("manga_asassn_crossmatch_vizier_results", manga_final_catalogue)
manga_final_sample.to_csv("manga_asassn_crossmatch.csv")


# AMUSING ++


vizier_results_2 = []
v2 = Vizier(columns=["**"], catalog="J/AJ/159/167/table3")
for ra, dec in zip(asas_sn_table["RA"].values, asas_sn_table["DEC"].values):
    try:
        pos = coords.SkyCoord(ra, dec, frame="icrs", unit=(u.deg, u.deg))
        # This is to get everything that can fit into a 127-Fiber IFU
        res = v2.query_region(pos, radius=Angle(32.0, "arcsec"))
        if len(res) > 0:
            vizier_results_2.append(res)
        else:
            vizier_results_2.append(None)
    except:
        vizier_results_2.append(None)

vizier_results_2 = np.array(vizier_results_2)
vizier_mask_2 = np.array(vizier_results_2) != None
amusing_catalogue = vizier_results_2[vizier_mask_2]
amusing_sample = asas_sn_table[vizier_mask_2]


np.save("manga_amusing_crossmatch_vizier_results", amusing_catalogue)
amusing_sample.to_csv("manga_amusing_crossmatch.csv")
