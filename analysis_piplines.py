import velocyto as vcy

import numpy as np

vlm = vcy.VelocytoLoom("XXXXXXXXXXX.loom")


vlm.normalize("S", size=True, log=True)

vlm.S_norm  # contains log normalized
vlm.plot_fractions()

vlm.filter_cells(bool_array=vlm.initial_Ucell_size > np.percentile(vlm.initial_Ucell_size, 0.5))
#vlm.set_clusters(vlm.ca["ClusterName"])
vlm.score_detection_levels(min_expr_counts=0.1, min_cells_express=1)
vlm.filter_genes(by_detection_levels=True)
vlm.score_cv_vs_mean(1000, plot=True, max_expr_avg=1000)
vlm.filter_genes(by_cv_vs_mean=True)
vlm._normalize_S(relative_size=vlm.S.sum(0),target_size=vlm.S.sum(0).mean())
vlm._normalize_U(relative_size=vlm.U.sum(0),target_size=vlm.U.sum(0).mean())




vlm.S_norm=np.nan_to_num(vlm.S_norm)
#np.nan_to_num(vlm.A)
#np.nan_to_num(vlm.U)
vlm.perform_PCA()
print(vlm.pcs)

vlm.knn_imputation()
#vlm.knn_imputation(n_pca_dims=20, k=500, balanced=True, b_sight=854, b_maxl=1500, n_jobs=16)
vlm.fit_gammas()
#vlm.fit_gammas(maxmin_perc=[1,59],maxmin_weighted_pow = 5)

#vlm.plot_phase_portraits(["Igfbpl1", "Pdgfra"])

from sklearn.manifold import TSNE


bh_tsne = TSNE()
#vlm.ts = bh_tsne.fit_transform(vlm.pcs[:, :25])
vlm.ts = bh_tsne.fit_transform(vlm.pcs[:, :25])

print(vlm.ts)

vlm.predict_U()
vlm.delta_t = 1
vlm.used_delta_t = vlm.delta_t
#vlm.delta_S = vlm.delta_t * vlm.velocity
vlm.calculate_velocity()
vlm.extrapolate_cell_at_t()
#vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt", psc=1, n_neighbors=850, knn_random=True, sampled_fraction=0.5)
vlm.estimate_transition_prob()
vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=True)

print(vlm.delta_t)
print(vlm.delta_embedding)
vlm.calculate_grid_arrows(smooth=0.8, steps=(40, 40), n_neighbors=300)

from  matplotlib import pyplot

pyplot.figure(None,(20,10))




vlm.colorandum = "red"


vlm.flow=np.nan_to_num(vlm.flow)

vlm.flow=np.nan_to_num(vlm.flow_grid)




vlm.plot_grid_arrows(quiver_scale=12,
                    scatter_kwargs_dict={"alpha":0.35, "lw":0.35, "edgecolor":"0.4", "s":38, "rasterized":True}, width =0.006,
                    minlength=3,min_mass=0.5, angles='xy', scale_units='xy',
                    plot_random=True, scale_type="absolute",color = "green",alpha = 0.70)

