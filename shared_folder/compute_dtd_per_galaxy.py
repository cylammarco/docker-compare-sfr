from astropy.io import fits
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize


age_maoz2012 = np.array([0.21, 1.41, 8.2])
age_err_maoz2012 = np.array([[0.21, 0.21], [0.99, 0.99], [5.8, 5.8]]).T
dtd_maoz2012 = np.array([140, 25.1, 1.83]) * 1e-14
dtd_err_maoz2012 = np.array([[30, 30], [6.3, 6.3], [0.41, 0.41]]).T * 1e-14

data = fits.open("manga_firefly-v2_4_3.fits")
results = np.load("manga_firefly-v2_4_3_tns_matched.npy", allow_pickle=True)
full_results = np.load(
    "manga_firefly-v2_4_3_tns_matched_full_details.npy", allow_pickle=True
)

matched = []
matched_id = []

for i, res in enumerate(results):
    if res["data"]["reply"] != []:
        matched.append(i)

for j, res in enumerate(full_results):
    name = res["data"]["reply"]["object_type"]["name"]
    if name is not None:
        if name[:2] in ["SN", "SL"]:
            matched_id.append(matched[j])


sfh_hdu = data[14].data
mass_hdu = data[11].data
stn_hdu = data[15].data

age = np.round(
    10.0 ** np.sort(list(set(sfh_hdu[:, :, :, 0].flatten())))[1:], 4
)
age = np.concatenate(([0], age))

total_mass_grid = np.zeros((len(sfh_hdu), len(age)))

try:
    total_mass_grid = np.load("manga_firefly-v2_4_3_total_mass_grid.npy")
except Exception as e:
    print(e)
    for id in range(len(results)):
        sfh_vor = sfh_hdu[id]
        mass_vor = mass_hdu[id]
        stn_vor = stn_hdu[id]
        for sfh_vor_i, mass_vor_i, stn_vor_i in zip(
            sfh_vor, mass_vor, stn_vor
        ):
            population_age = np.round(10.0 ** sfh_vor_i[:, 0], 4)
            # meatllicity = sfh_vor_i[:, 1]
            mass_weight = sfh_vor_i[:, 2]
            mass_weight[np.where(mass_weight < 0)] = 0.0
            mass = 10.0 ** mass_vor_i[0]
            # Get the index for the age that stars formed
            age_idx = [np.where(age == a)[0][0] for a in population_age]
            # Put the mass formed at the right time
            total_mass_grid[id][age_idx] += mass_weight * mass
    np.save("manga_firefly-v2_4_3_total_mass_grid", total_mass_grid)

# effective visibility time 2007 to 2022 = 15 years
t = 15.0
epsilon = 1.0

# r in Eq 2. Maoz et al. 2012
dtd = np.ones_like(age) * -13  # this is log-10-ed
n_sn = np.zeros(len(results))
n_sn[matched_id] = 1.0


def _likelihood(dtd, total_mass_grid, n_sn):
    # Eq. 2, the lamb is for each galaxy
    lamb = np.sum(10**dtd * total_mass_grid, axis=1) * t * epsilon
    mask = lamb > 0.0
    # Eq. 6, currently assuming either 0 or 1 SN per voronoi cell
    ln_like = -np.sum(lamb[mask]) + np.sum(np.log(lamb**n_sn)[mask])
    return -ln_like


answer = minimize(
    _likelihood,
    dtd,
    args=(total_mass_grid, n_sn),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)

age_output = deepcopy(age)
dtd_output = deepcopy(answer.x)

age_output_bin_2 = [
    sum(age_output[i : i + 2]) / 2 for i in range(0, len(age_output), 2)
]
dtd_output_bin_2 = [
    sum(dtd_output[i : i + 2]) / 2 for i in range(0, len(dtd_output), 2)
]

age_output_bin_5 = [
    sum(age_output[i : i + 5]) / 5 for i in range(0, len(age_output), 5)
]
dtd_output_bin_5 = [
    sum(dtd_output[i : i + 5]) / 5 for i in range(0, len(dtd_output), 5)
]

age_output_bin_10 = [
    sum(age_output[i : i + 10]) / 10 for i in range(0, len(age_output), 10)
]
dtd_output_bin_10 = [
    sum(dtd_output[i : i + 10]) / 10 for i in range(0, len(dtd_output), 10)
]


total_mass_bin_2_01 = np.sum(total_mass_grid[:, 0:2], axis=1)
total_mass_bin_2_02 = np.sum(total_mass_grid[:, 2:4], axis=1)
total_mass_bin_2_03 = np.sum(total_mass_grid[:, 4:6], axis=1)
total_mass_bin_2_04 = np.sum(total_mass_grid[:, 6:8], axis=1)
total_mass_bin_2_05 = np.sum(total_mass_grid[:, 8:10], axis=1)
total_mass_bin_2_06 = np.sum(total_mass_grid[:, 10:12], axis=1)
total_mass_bin_2_07 = np.sum(total_mass_grid[:, 12:14], axis=1)
total_mass_bin_2_08 = np.sum(total_mass_grid[:, 14:16], axis=1)
total_mass_bin_2_09 = np.sum(total_mass_grid[:, 16:18], axis=1)
total_mass_bin_2_10 = np.sum(total_mass_grid[:, 18:20], axis=1)
total_mass_bin_2_11 = np.sum(total_mass_grid[:, 20:22], axis=1)
total_mass_bin_2_12 = np.sum(total_mass_grid[:, 22:24], axis=1)
total_mass_bin_2_13 = np.sum(total_mass_grid[:, 24:26], axis=1)
total_mass_bin_2_14 = np.sum(total_mass_grid[:, 26:28], axis=1)
total_mass_bin_2_15 = np.sum(total_mass_grid[:, 28:30], axis=1)
total_mass_bin_2_16 = np.sum(total_mass_grid[:, 30:32], axis=1)
total_mass_bin_2_17 = np.sum(total_mass_grid[:, 32:34], axis=1)
total_mass_bin_2_18 = np.sum(total_mass_grid[:, 34:36], axis=1)
total_mass_bin_2_19 = np.sum(total_mass_grid[:, 36:38], axis=1)
total_mass_bin_2_20 = np.sum(total_mass_grid[:, 38:40], axis=1)
total_mass_bin_2_21 = np.sum(total_mass_grid[:, 40:42], axis=1)
total_mass_bin_2_22 = np.sum(total_mass_grid[:, 42:44], axis=1)
total_mass_bin_2_23 = np.sum(total_mass_grid[:, 44:46], axis=1)
total_mass_bin_2_24 = np.sum(total_mass_grid[:, 46:48], axis=1)
total_mass_bin_2_25 = np.sum(total_mass_grid[:, 48:], axis=1)

total_mass_grid_bin_2 = np.column_stack(
    (
        total_mass_bin_2_01,
        total_mass_bin_2_02,
        total_mass_bin_2_03,
        total_mass_bin_2_04,
        total_mass_bin_2_05,
        total_mass_bin_2_06,
        total_mass_bin_2_07,
        total_mass_bin_2_08,
        total_mass_bin_2_09,
        total_mass_bin_2_10,
        total_mass_bin_2_11,
        total_mass_bin_2_12,
        total_mass_bin_2_13,
        total_mass_bin_2_14,
        total_mass_bin_2_15,
        total_mass_bin_2_16,
        total_mass_bin_2_17,
        total_mass_bin_2_18,
        total_mass_bin_2_19,
        total_mass_bin_2_20,
        total_mass_bin_2_21,
        total_mass_bin_2_22,
        total_mass_bin_2_23,
        total_mass_bin_2_24,
        total_mass_bin_2_25,
    )
)
dtd_bin_2 = np.ones_like(age_output_bin_2) * -13  # this is log-10-ed


total_mass_bin_5_01 = np.sum(total_mass_grid[:, 0:5], axis=1)
total_mass_bin_5_02 = np.sum(total_mass_grid[:, 5:10], axis=1)
total_mass_bin_5_03 = np.sum(total_mass_grid[:, 10:15], axis=1)
total_mass_bin_5_04 = np.sum(total_mass_grid[:, 15:20], axis=1)
total_mass_bin_5_05 = np.sum(total_mass_grid[:, 20:25], axis=1)
total_mass_bin_5_06 = np.sum(total_mass_grid[:, 25:30], axis=1)
total_mass_bin_5_07 = np.sum(total_mass_grid[:, 30:35], axis=1)
total_mass_bin_5_08 = np.sum(total_mass_grid[:, 35:40], axis=1)
total_mass_bin_5_09 = np.sum(total_mass_grid[:, 40:45], axis=1)
total_mass_bin_5_10 = np.sum(total_mass_grid[:, 45:], axis=1)

total_mass_grid_bin_5 = np.column_stack(
    (
        total_mass_bin_5_01,
        total_mass_bin_5_02,
        total_mass_bin_5_03,
        total_mass_bin_5_04,
        total_mass_bin_5_05,
        total_mass_bin_5_06,
        total_mass_bin_5_07,
        total_mass_bin_5_08,
        total_mass_bin_5_09,
        total_mass_bin_5_10,
    )
)
dtd_bin_5 = np.ones_like(age_output_bin_5) * -13  # this is log-10-ed


total_mass_bin_10_01 = np.sum(total_mass_grid[:, 0:10], axis=1)
total_mass_bin_10_02 = np.sum(total_mass_grid[:, 10:20], axis=1)
total_mass_bin_10_03 = np.sum(total_mass_grid[:, 20:30], axis=1)
total_mass_bin_10_04 = np.sum(total_mass_grid[:, 30:40], axis=1)
total_mass_bin_10_05 = np.sum(total_mass_grid[:, 40:50], axis=1)

total_mass_grid_bin_10 = np.column_stack(
    (
        total_mass_bin_10_01,
        total_mass_bin_10_02,
        total_mass_bin_10_03,
        total_mass_bin_10_04,
        total_mass_bin_10_05,
    )
)
dtd_bin_10 = np.ones_like(age_output_bin_10) * -13  # this is log-10-ed

answer_bin_2 = minimize(
    _likelihood,
    dtd_bin_2,
    args=(total_mass_grid_bin_2, n_sn),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)
answer_bin_5 = minimize(
    _likelihood,
    dtd_bin_5,
    args=(total_mass_grid_bin_5, n_sn),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)
answer_bin_10 = minimize(
    _likelihood,
    dtd_bin_10,
    args=(total_mass_grid_bin_10, n_sn),
    method="Nelder-Mead",
    options={"maxiter": 100000},
)

plt.figure(1, figsize=(8, 6))
plt.clf()
plt.plot(age, 10.0 ** (answer.x), label="FIREFLY native time bin")
plt.plot(age_output_bin_2, 10.0 ** (answer_bin_2.x), label="bin2")
plt.plot(age_output_bin_5, 10.0 ** (answer_bin_5.x), label="bin5")
plt.plot(age_output_bin_10, 10.0 ** (answer_bin_10.x), label="bin10")
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
