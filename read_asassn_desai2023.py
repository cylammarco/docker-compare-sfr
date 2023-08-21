from datetime import date
import glob
import re

from astropy import coordinates as coords
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.coordinates import Angle
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def remove_field_name(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b


# Table 1
_file = []
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_464_2672\\table1.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U7"),
        ("DiscDate", "<U14"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("VmagD", "<f8"),
        ("VmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U8"),
    ],
    delimiter=[12, 7, 14, 3, 3, 6, 4, 3, 5, 9, 5, 5, 6, 8],
    autostrip=True,
)
_file.append(_temp)
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_467_1098\\table1.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U7"),
        ("DiscDate", "<U14"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("VmagD", "<f8"),
        ("VmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U8"),
    ],
    delimiter=[12, 8, 14, 3, 3, 6, 4, 3, 5, 8, 5, 5, 7, 8],
    autostrip=True,
)
_file.append(_temp)
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_471_4966\\table1.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U7"),
        ("DiscDate", "<U14"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("VmagD", "<f8"),
        ("VmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U9"),
    ],
    delimiter=[12, 8, 14, 3, 3, 6, 4, 3, 5, 8, 5, 5, 6, 9],
    autostrip=True,
)
_file.append(_temp)
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_484_1899\\table1.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U7"),
        ("DiscDate", "<U14"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("mdiscmag", "<f8"),
        ("VmagD", "<f8"),
        ("VmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U8"),
    ],
    delimiter=[12, 8, 14, 3, 3, 6, 4, 3, 6, 8, 5, 5, 5, 6, 8],
    autostrip=True,
)
_temp = remove_field_name(_temp, 'mdiscmag')
_file.append(_temp)

sn_table_1 = np.concatenate(_file, axis=0)



# Table 2
_file = []
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_464_2672\\table2.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U7"),
        ("DiscDate", "<U14"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("VmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U8"),
    ],
    delimiter=[30, 7, 14, 3, 3, 6, 4, 3, 5, 9, 5, 7, 8],
    autostrip=True,
)
_file.append(_temp)
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_467_1098\\table2.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U7"),
        ("DiscDate", "<U14"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("VmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U8"),
    ],
    delimiter=[30, 7, 15, 3, 3, 6, 4, 3, 5, 8, 5, 6, 8],
    autostrip=True,
)
_file.append(_temp)
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_471_4966\\table2.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U7"),
        ("DiscDate", "<U14"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("VmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U16"),
    ],
    delimiter=[30, 8, 14, 3, 3, 6, 4, 3, 5, 8, 5, 7, 16],
    autostrip=True,
)
_file.append(_temp)
_temp = np.genfromtxt(
    'ASASSN\\J_MNRAS_484_1899\\table2.dat',
    dtype=[
        ("SNName", "<U30"),
        ("IAUName", "<U8"),
        ("DiscDate", "<U15"),
        ("RAh", "<i4"),
        ("RAm", "<i4"),
        ("RAs", "<f8"),
        ("DEd", "<U4"),
        ("DEm", "<i4"),
        ("DEs", "<f8"),
        ("zSN", "<f8"),
        ("mpeakmag", "<f8"),
        ("VmagP", "<f8"),
        ("gmagP", "<f8"),
        ("Offset", "<f8"),
        ("Type", "<U16"),
    ],
    delimiter=[30, 8, 14, 3, 3, 6, 4, 3, 6, 9, 5, 5, 5, 7, 8],
    autostrip=True,
)
_temp = remove_field_name(_temp, 'mpeakmag')
_temp = remove_field_name(_temp, 'gmagP')
_file.append(_temp)

sn_table_2 = np.concatenate(_file, axis=0)


table_1_Ia_mask = ['Ia' in i for i in sn_table_1['Type']]
table_2_Ia_mask = ['Ia' in i for i in sn_table_2['Type']]

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
    _ra = str(i[3]) + ":" + str(i[4]) + ":" + str(i[5])
    _dec = str(i[6]) + ":" + str(i[7]) + ":" + str(i[8])
    _type = i[-1]
    ra.append(_ra)
    dec.append(_dec)
    sn_name.append(i[0])
    types.append(i[-1])
    # sc.append(coords.SkyCoord(_ra, _dec, unit=(u.hourangle, u.deg)))


d1 = {"dateobs": dateobs, "RA": ra, "DEC": dec, "SNName": sn_name, "Type": types}
asas_sn_table_1 = pd.DataFrame(data=d1)


dateobs = []
sn_name = []
ra = []
dec = []
types = []
sc = []
for i in sn_table_2[table_2_Ia_mask]:
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
    _ra = str(i[3]) + ":" + str(i[4]) + ":" + str(i[5])
    _dec = str(i[6]) + ":" + str(i[7]) + ":" + str(i[8])
    ra.append(_ra)
    dec.append(_dec)
    sn_name.append(i[0])
    types.append(i[-1])
    # sc.append(coords.SkyCoord(_ra, _dec, unit=(u.hourangle, u.deg)))


d2 = {"dateobs": dateobs, "RA": ra, "DEC": dec, "SNName": sn_name, "Type": types}
asas_sn_table_2 = pd.DataFrame(data=d2)


asas_sn_table_1["dateobs"] = asas_sn_table_1["dateobs"].astype("datetime64")
asas_sn_table_1.set_index("dateobs", inplace=True)
histogram_all_asas_sn_1 = asas_sn_table_1.resample("1Y").size()

asas_sn_table_2["dateobs"] = asas_sn_table_2["dateobs"].astype("datetime64")
asas_sn_table_2.set_index("dateobs", inplace=True)
histogram_all_asas_sn_2 = asas_sn_table_2.resample("1Y").size()

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
    histogram_all_asas_sn_1,
    label="No. of SN Ia in that year discovered by ASAS-SN",
)
plt.plot(
    histogram_all_asas_sn_2,
    label="No. of SN Ia in that year NOT discovered by ASAS-SN",
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
