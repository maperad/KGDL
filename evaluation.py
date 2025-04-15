import numpy as np
from sklearn.metrics import accuracy_score, pairwise_distances


def get_cluster_mean_invcov(emb_X, emb_y):
    clust_centroids = []
    clust_inv_covs = []
    for y_j in sorted(set(emb_y)):
        mask_y_j = emb_y == y_j
        cen_y_j = emb_X[mask_y_j].mean(axis=0)
        clust_centroids.append(cen_y_j)
        if sum(mask_y_j) == 1:
            # Set cov to ID matrix if only one instance in cluster
            # This reduces Mahalanobis distance to Euclidean distance
            inv_cov_y_j = np.eye(emb_X.shape[1])
        else:
            cov_y_j = np.cov(emb_X[mask_y_j].T)
            inv_cov_y_j = np.linalg.pinv(cov_y_j)
        clust_inv_covs.append(inv_cov_y_j)
    clust_centroids = np.array(clust_centroids)
    clust_inv_covs = np.array(clust_inv_covs)
    return clust_centroids, clust_inv_covs


def get_mah_dist(
        emb_X: np.array, clust_centroids: np.array, clust_inv_covs: np.array
) -> np.array:
    mah_dist = np.zeros((emb_X.shape[0], clust_centroids.shape[0]))
    for i, cen in enumerate(clust_centroids):
        mah_dist[:, i] = pairwise_distances(
            emb_X, cen.reshape(1, -1), metric='mahalanobis',
            VI=clust_inv_covs[i]
        ).reshape(-1)
    return mah_dist


def evaluate_model_performance(model, dataloaders, device, parts):
    metric_dict = {}
    projection = True if 'embedding' in parts else False
    proj_X, proj_dist_min, clf_y, ood_y = {}, {}, {}, {}
    proj_mah_dist = {}
    for dl_k, dl in dataloaders.items():
        proj_X[dl_k], emb_X, clf_X, clf_y[dl_k] = \
            model.get_proj_emb_clf_y_from_dataloader(
                dl, device, projection=projection
            )
        if not projection:
            proj_X[dl_k] = emb_X
        metric_dict[f'clf_{dl_k}_acc'] = round(float(
            accuracy_score(clf_y[dl_k], clf_X.argmax(axis=1))
        ), 4)

        if dl_k == 'train':
            proj_clust_m, proj_clust_inv_cov = get_cluster_mean_invcov(
                proj_X[dl_k], clf_y[dl_k]
            )

            proj_mah_dist[dl_k] = get_mah_dist(
                proj_X[dl_k], proj_clust_m, proj_clust_inv_cov
            )
            try:
                # Use correct predictions to get threshold per class and all
                proj_dist_thr = np.quantile(
                    proj_mah_dist[dl_k].min(axis=1)[
                        proj_mah_dist[dl_k].argmin(axis=1) == clf_y[dl_k]
                    ],
                    0.99
                ).item()
            except IndexError:
                proj_dist_thr = 0
            metric_dict |= {'proj_dist_thr': round(float(proj_dist_thr), 4)}
        else:
            proj_mah_dist[dl_k] = get_mah_dist(
                proj_X[dl_k], proj_clust_m, proj_clust_inv_cov
            )
        proj_dist_min[dl_k] = proj_mah_dist[dl_k].min(axis=1)
        if 'ood' in dl_k:
            ood_y[dl_k] = np.ones_like(clf_y[dl_k])
        else:
            ood_y[dl_k] = np.zeros_like(clf_y[dl_k])
        proj_ood_res = accuracy_score(
            ood_y[dl_k], proj_dist_min[dl_k] > proj_dist_thr
        )
        metric_dict[f'ood_{dl_k}_acc'] = round(float(proj_ood_res), 4)

    return metric_dict
