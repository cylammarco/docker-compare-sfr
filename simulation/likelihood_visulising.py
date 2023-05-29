
trial = answer.x.copy()
trial[10] -= 10
total_1 = np.zeros_like(trial)
for i in sfh_voronoi_with_sn:
    total_1 += 10.**trial * i


total_likelihood_1 = likelihood_voronoi(trial, np.sum(sfh_voronoi, axis=0), sfh_voronoi_with_sn, sn_list[sn_mask],special.factorial(sn_list[sn_mask]))


trial = answer.x.copy()
trial[10] -= 0
total_2 = np.zeros_like(trial)
total_likelihood_2 = 0
for i in sfh_voronoi_with_sn:
    total_2 += 10.**trial * i


total_likelihood_2 = likelihood_voronoi(trial, np.sum(sfh_voronoi, axis=0), sfh_voronoi_with_sn, sn_list[sn_mask],special.factorial(sn_list[sn_mask]))


plot(input_age, total_1, color='C0')
plot(input_age, total_2, color='C1')


figure(1)
clf()
for i in sfh_voronoi_with_sn:
    if i[2] > 0.0:
        scatter(input_age, i, s=1)



figure(2)
clf()
for i in sfh_voronoi_with_sn:
    if i[3] > 0.0:
        scatter(input_age, i, s=1)


