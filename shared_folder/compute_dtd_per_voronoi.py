from astropy.io import fits
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from numba import jit

age_maoz2012 = np.array([0.21, 1.41, 8.2])
age_err_maoz2012 = np.array([[0.21, 0.21], [0.99, 0.99], [5.8, 5.8]]).T
dtd_maoz2012 = np.array([140, 25.1, 1.83]) * 1e-14
dtd_err_maoz2012 = np.array([[30, 30], [6.3, 6.3], [0.41, 0.41]]).T * 1e-14

data = fits.open("../firefly/manga_firefly-v2_4_3.fits.bz2")
results = np.load(
    "../firefly/manga_firefly-v2_4_3_tns_matched.npy", allow_pickle=True
)
full_results = np.load(
    "../firefly/manga_firefly-v2_4_3_tns_matched_full_details.npy",
    allow_pickle=True,
)

matched = []
matched_id = []

sn_ra = {}
sn_dec = {}

for i, res in enumerate(results):
    if res["data"]["reply"] != []:
        matched.append(i)

for j, res in enumerate(full_results):
    name = res["data"]["reply"]["object_type"]["name"]
    if name is not None:
        if name[:2] in ["SN", "SL"]:
            matched_id.append(matched[j])
            sn_ra[matched[j]] = res["data"]["reply"]["radeg"]
            sn_dec[matched[j]] = res["data"]["reply"]["decdeg"]

sfh_hdu = data[14].data
mass_hdu = data[11].data
stn_hdu = data[15].data
spatial_hdu = data[4].data
galaxy_hdu = data[1].data


age = np.round(
    10.0 ** np.sort(list(set(sfh_hdu[i][:, :, 0].flatten())))[1:], 4
)
age = np.concatenate(([0], age))

age_bin_2 = np.array([
    sum(age[i : i + 2]) / 2 for i in range(0, len(age), 2)
])

age_bin_5 = np.array([
    sum(age[i : i + 5]) / 5 for i in range(0, len(age), 5)
])

age_bin_10 = np.array([
    sum(age[i : i + 10]) / 10 for i in range(0, len(age), 10)
])


mass_grid = {}
mass_grid_bin_2 = {}
mass_grid_bin_5 = {}
mass_grid_bin_10 = {}
for i in range(len(results)):
    mass_grid[i] = {}
    count = 0
    for j in spatial_hdu[i][:, 0]:
        count += 1
    mass_grid[i] = np.zeros((count, len(age)))
    mass_grid_bin_2[i] = np.zeros((count, int(len(age_bin_2))))
    mass_grid_bin_5[i] = np.zeros((count, int(len(age_bin_5))))
    mass_grid_bin_10[i] = np.zeros((count, int(len(age_bin_10))))

#mass_grid = np.zeros((len(results), len(sfh_hdu[0]), len(age)))

ra = np.zeros_like(results)
dec = np.zeros_like(results)
redshift = np.zeros_like(results)

sn_vor_id = np.zeros_like(results)

for i in range(len(results)):
    print(i)
    sfh_vor = sfh_hdu[i]
    mass_vor = mass_hdu[i]
    # stn_vor = stn_hdu[i]
    spatial_vor = spatial_hdu[i]
    ra[i] = galaxy_hdu[i]["OBJRA"]
    dec[i] = galaxy_hdu[i]["OBJDEC"]
    redshift[i] = galaxy_hdu[i]["REDSHIFT"]
    _delta_ra = {}
    _delta_dec = {}
    for sfh_vor_i, mass_vor_i, spatial_vor_i in zip(sfh_vor, mass_vor, spatial_vor):
        vor_id = int(spatial_vor_i[0])
        if vor_id > 0:
            population_age = np.round(10.0 ** sfh_vor_i[:, 0], 4)
            # meatllicity = sfh_vor_i[:, 1]
            mass_weight = sfh_vor_i[:, 2]
            mass_weight[np.where(mass_weight < 0)] = 0.0
            mass = 10.0 ** mass_vor_i[0]
            # Get the index for the age that stars formed
            age_idx = [np.where(age == a)[0][0] for a in population_age]
            age_bin_2_idx = np.array(age_idx) // 2
            age_bin_5_idx = np.array(age_idx) // 5
            age_bin_10_idx = np.array(age_idx) // 10
            # Get the spatial information
            _delta_ra[vor_id] = spatial_vor_i[1]
            _delta_dec[vor_id] = spatial_vor_i[2]
            # Put the mass formed at the right time
            mass_grid[i][vor_id][age_idx] = mass_weight * mass
            mass_grid_bin_2[i][vor_id][age_bin_2_idx] = mass_weight * mass
            mass_grid_bin_5[i][vor_id][age_bin_5_idx] = mass_weight * mass
            mass_grid_bin_10[i][vor_id][age_bin_10_idx] = mass_weight * mass
    if i in matched_id:
        # convert the delta ra and dec into unit of degrees
        dist = (
            ra[i]
            + np.array(list(_delta_ra.values())) / 60.0 / 60.0
            - sn_ra[i]
        ) ** 2.0 + (
            dec[i]
            + np.array(list(_delta_dec.values())) / 60.0 / 60.0
            - sn_dec[i]
        ) ** 2
        sn_vor_id[i] = np.argmin(dist)


# np.save("manga_firefly-v2_4_3_mass_grid", mass_grid)
np.save("manga_firefly-v2_4_3_matched_id", matched_id)
np.save("manga_firefly-v2_4_3_sn_vor_id", sn_vor_id)

# effective visibility time 2007 to 2022 = 15 years
t = 15.0
epsilon = 1.0

# r in Eq 2. Maoz et al. 2012
dtd = np.ones_like(age) * -13  # this is log-10-ed
n_sn = np.zeros((len(results), len(sfh_hdu[0])))
for i in matched_id:
    n_sn[i][sn_vor_id[i]] = 1.0


size = len(mass_grid)


@jit(nopython=False)
def _likelihood_voronoi(dtd, mass_grid, n_sn, size):
    # Eq. 2, the lamb is for each galaxy
    lamb = np.sum(np.array([dtd * mass_grid[i] for i in range(size)]), axis=2) * t * epsilon
    mask = lamb > 0.0
    # Eq. 6, currently assuming either 0 or 1 SN per voronoi cell
    ln_like = -np.sum(lamb[mask]) + np.sum(np.log(lamb**n_sn)[mask])
    print(-ln_like)
    return -ln_like


answer = minimize(
    _likelihood_voronoi,
    10**dtd,
    args=(mass_grid, n_sn, size),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)


dtd_bin_2 = np.ones_like(age_bin_2) * -13  # this is log-10-ed
answer_bin_2 = minimize(
    _likelihood_voronoi,
    10.**dtd_bin_2,
    args=(mass_grid_bin_2, n_sn, size),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)

dtd_bin_5 = np.ones_like(age_bin_5) * -13  # this is log-10-ed
answer_bin_5 = minimize(
    _likelihood_voronoi,
    10.**dtd_bin_5,
    args=(mass_grid_bin_5, n_sn, size),
    method="Nelder-Mead",
    options={"maxiter": 100000},
    bounds=[(1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10)]
)

dtd_bin_10 = np.ones_like(age_bin_10) * -13  # this is log-10-ed
answer_bin_10 = minimize(
    _likelihood_voronoi,
    10.**dtd_bin_10,
    args=(mass_grid_bin_10, n_sn, size),
    method="Nelder-Mead",
    options={"maxiter": 100000},
    bounds=[(1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10), (1e-20, 1e-10)]
)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.scatter(age, answer.x, label="FIREFLY native time bin")
plt.scatter(age_bin_2, answer_bin_2.x, label="bin2")
plt.scatter(age_bin_5, answer_bin_5.x, label="bin5")
plt.scatter(age_bin_10, answer_bin_10.x, label="bin10")
plt.errorbar(
    age_maoz2012,
    dtd_maoz2012,
    xerr=age_err_maoz2012,
    yerr=dtd_err_maoz2012,
    marker="+",
    color="black",
    label="Maoz et al. 2012",
    ls="",
)
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.xlim(1e-2, 13)
plt.ylim(1e-16, 1e-10)
plt.xlabel("Delay Time (Gyr)")
plt.ylabel(r"SN / yr / M$_\odot$")
plt.legend(loc="upper right")
plt.title("Non-parametric fit")
plt.tight_layout()
plt.savefig("DTD.png")
