import copy
import os
from astropy.io import fits

gsd = fits.open(os.path.join("pipe3d_example_data", "gsd01_156.fits"))[0]

header_0 = copy.deepcopy(gsd.header)
header_1 = copy.deepcopy(gsd.header)
header_2 = copy.deepcopy(gsd.header)
header_3 = copy.deepcopy(gsd.header)

gsd_0 = gsd.data[::4]
gsd_1 = gsd.data[1::4]
gsd_2 = gsd.data[2::4]
gsd_3 = gsd.data[3::4]

metal_0 = "z0037047"
metal_1 = "z0075640"
metal_2 = "z0190000"
metal_3 = "z0315321"

header_0["NAXIS2"] = 39
header_1["NAXIS2"] = 39
header_2["NAXIS2"] = 39
header_3["NAXIS2"] = 39

for i in range(39):
    # header_0
    header_0.remove("NAME{}".format(i * 4 + 1))
    header_0.remove("NAME{}".format(i * 4 + 2))
    header_0.remove("NAME{}".format(i * 4 + 3))
    # header_1
    header_1.remove("NAME{}".format(i * 4))
    header_1.remove("NAME{}".format(i * 4 + 2))
    header_1.remove("NAME{}".format(i * 4 + 3))
    # header_2
    header_2.remove("NAME{}".format(i * 4))
    header_2.remove("NAME{}".format(i * 4 + 1))
    header_2.remove("NAME{}".format(i * 4 + 3))
    # header_3
    header_3.remove("NAME{}".format(i * 4))
    header_3.remove("NAME{}".format(i * 4 + 1))
    header_3.remove("NAME{}".format(i * 4 + 2))
    # header_0
    header_0.remove("NORM{}".format(i * 4 + 1))
    header_0.remove("NORM{}".format(i * 4 + 2))
    header_0.remove("NORM{}".format(i * 4 + 3))
    # header_1
    header_1.remove("NORM{}".format(i * 4))
    header_1.remove("NORM{}".format(i * 4 + 2))
    header_1.remove("NORM{}".format(i * 4 + 3))
    # header_2
    header_2.remove("NORM{}".format(i * 4))
    header_2.remove("NORM{}".format(i * 4 + 1))
    header_2.remove("NORM{}".format(i * 4 + 3))
    # header_3
    header_3.remove("NORM{}".format(i * 4))
    header_3.remove("NORM{}".format(i * 4 + 1))
    header_3.remove("NORM{}".format(i * 4 + 2))

# rename the numbers
for i in range(39):
    # header_0
    if i != 0:
        header_0.rename_keyword("NAME{}".format(i * 4), "NAME{}".format(i))
        header_0.rename_keyword("NORM{}".format(i * 4), "NORM{}".format(i))
    # header_1
    header_1.rename_keyword("NAME{}".format(i * 4 + 1), "NAME{}".format(i))
    header_1.rename_keyword("NORM{}".format(i * 4 + 1), "NORM{}".format(i))
    # header_2
    header_2.rename_keyword("NAME{}".format(i * 4 + 2), "NAME{}".format(i))
    header_2.rename_keyword("NORM{}".format(i * 4 + 2), "NORM{}".format(i))
    # header_3
    header_3.rename_keyword("NAME{}".format(i * 4 + 3), "NAME{}".format(i))
    header_3.rename_keyword("NORM{}".format(i * 4 + 3), "NORM{}".format(i))
    # header_0


fits_image_0 = fits.PrimaryHDU(data=gsd_0, header=header_0)
fits_image_1 = fits.PrimaryHDU(data=gsd_1, header=header_1)
fits_image_2 = fits.PrimaryHDU(data=gsd_2, header=header_2)
fits_image_3 = fits.PrimaryHDU(data=gsd_3, header=header_3)

fits_image_0.writeto(
    os.path.join("pipe3d_example_data", "gsd01_z0037047.fits"), overwrite=True
)
fits_image_1.writeto(
    os.path.join("pipe3d_example_data", "gsd01_z0075640.fits"), overwrite=True
)
fits_image_2.writeto(
    os.path.join("pipe3d_example_data", "gsd01_z0190000.fits"), overwrite=True
)
fits_image_3.writeto(
    os.path.join("pipe3d_example_data", "gsd01_z0315321.fits"), overwrite=True
)
