import pandas
import glob
from matplotlib import pyplot as plt

filelist = glob.glob("tns-sn1a-20221012/tns_search*.csv")
_file = []
for filename in filelist:
    _file.append(pandas.read_csv(filename))

sn_table = pandas.concat(_file)

sn_table["Discovery Date (UT)"] = sn_table["Discovery Date (UT)"].astype(
    "datetime64"
)

sn_table.set_index("Discovery Date (UT)", inplace=True)
histogram = sn_table.resample("1Y").size()

plt.figure(1, figsize=(8, 8))
plt.plot(histogram)
plt.xlabel("Year")
plt.ylabel("Number of SN Ia discovered")
plt.yscale("log")
plt.title("Number of SN Ia Discovery")
plt.tight_layout()
plt.grid()
plt.savefig("sn1a_rate.png")
