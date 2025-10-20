# utils/simulation.py
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform

from .optimization import port_minvol, port_maxret, port_minvol_ro


def bootstrap(returns):
    return resample(returns, replace=True, n_samples=None)


def efficient_frontier(mean, sigma, truemean, truesigma, nb_assets):
    # This mirrors the logic of your original notebook
    weights_matrix = np.zeros((20, nb_assets + 2))
    v = np.zeros((1, 40))

    min_weights_RS = port_minvol(mean, sigma)
    min_ptfmean_RS = min_weights_RS.T @ mean
    min_ptfsig_RS = (min_weights_RS.T @ sigma @ min_weights_RS) ** 0.5

    max_weights_RS = port_maxret(mean, sigma)
    max_ptfmean_RS = max_weights_RS.T @ mean
    max_ptfsig_RS = (max_weights_RS.T @ sigma @ max_weights_RS) ** 0.5

    step_RS = (max_ptfmean_RS - min_ptfmean_RS) / 19 if max_ptfmean_RS != min_ptfmean_RS else 0

    for i in range(20):
        ro = min_ptfmean_RS + i * step_RS
        wei_ro_RS = port_minvol_ro(mean, sigma, ro)
        mean_ro_RS = wei_ro_RS.T @ truemean
        sig_ro_RS = (wei_ro_RS.T @ truesigma @ wei_ro_RS) ** 0.5
        weights_matrix[i, :-2] = wei_ro_RS
        weights_matrix[i, -2] = mean_ro_RS
        weights_matrix[i, -1] = sig_ro_RS
        v[0, 2 * i] = mean_ro_RS
        v[0, 2 * i + 1] = sig_ro_RS
    return weights_matrix, v


def simul_EF(returns, mean, cov, nb_assets):
    nb_simul = 100
    weights_matrix_S = np.zeros((20, nb_assets + 2, nb_simul))
    vect = np.zeros((nb_simul, 40))
    i = 0

    while i < nb_simul:
        sm = bootstrap(returns)
        meanS = np.average(sm, axis=0)
        sigmaS = np.cov(sm, rowvar=False)
        weights_matrix_S[:, :, i], vect[i, :] = efficient_frontier(meanS, sigmaS, meanS, sigmaS, nb_assets)
        i += 1

    standard = preprocessing.scale(vect, axis=1)
    D = squareform(pdist(standard, metric="euclidean"))
    sum_dist = D.sum(axis=1)
    clusteroid_idx = np.argmin(sum_dist)
    clusteroid = vect[clusteroid_idx]

    RES_weights = weights_matrix_S[:, :, clusteroid_idx]
    ptf = {}
    for i in range(20):
        ptf[i] = RES_weights[i, :-2]

    INV = np.linalg.pinv(np.cov(vect, rowvar=False))
    distance = np.zeros((nb_simul, 2))
    for i in range(nb_simul):
        v = vect[i, :]
        distance[i, 0] = int(i)
        distance[i, 1] = ((v - clusteroid).T @ INV @ (v - clusteroid)) ** 0.5

    col = -1
    threshold = np.percentile(distance[:, col], 95)
    sim_indices = distance[:, 0][distance[:, col] <= threshold]

    subset_efficient = weights_matrix_S[:, :, np.array(sim_indices, dtype=int)]

    # return five portfolios like in your original notebook (0,4,9,14,19)
    return ptf[0], ptf[4], ptf[9], ptf[14], ptf[19]
