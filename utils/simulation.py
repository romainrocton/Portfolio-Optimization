# utils/simulation.py
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform

from .optimization import port_minvol, port_maxret, port_minvol_ro, port_maxsr
import streamlit as st
import matplotlib.pyplot as plt

def bootstrap(returns):
    return resample(returns, replace=True, n_samples=None)


def efficient_frontier(mean, sigma, truemean, truesigma, nb_assets):
    # This mirrors the logic of your original notebook
    weights_matrix = np.zeros((20, nb_assets + 2))
    v = np.zeros((1, 40))
    
    # Defines the two extremities of the frontier
    min_weights_RS = port_minvol(mean, sigma)
    min_ptfmean_RS = min_weights_RS.T @ mean
    min_ptfsig_RS = (min_weights_RS.T @ sigma @ min_weights_RS) ** 0.5

    max_weights_RS = port_maxret(mean, sigma)
    max_ptfmean_RS = max_weights_RS.T @ mean
    max_ptfsig_RS = (max_weights_RS.T @ sigma @ max_weights_RS) ** 0.5

    # Defines the step in terms of return to reach to go to the next portfolio in terms of return
    step_RS = (max_ptfmean_RS - min_ptfmean_RS) / 19 if max_ptfmean_RS != min_ptfmean_RS else 0

    # Computation of each portfolio, storage of values inside of arrays
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

@st.cache_data(ttl=86400)
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

    # return five portfolios like in your original notebook (0,4,9,14,19)
    return ptf[0], ptf[4], ptf[9], ptf[14], ptf[19]

@st.cache_data(ttl=86400)
def simul_Single_PTF(type,returns, mean, cov, nb_assets,targetreturn=0,rf=0):
    
    nb_simul = 2000
    weights_matrix_S = np.zeros((nb_simul,nb_assets))
    vect = np.zeros((nb_simul, 2))
    i = 0

    # Simulation of the characteristic portfolio as defined in the type, and storage of the results
    while i < nb_simul:
        sm = bootstrap(returns)
        meanS = np.average(sm, axis=0)
        sigmaS = np.cov(sm, rowvar=False)
        if type=="Min_Vol":
            intermediary_weights=port_minvol(meanS,sigmaS)
        elif type=="Target_Return":
            intermediary_weights=port_minvol_ro(meanS,sigmaS,targetreturn)
        elif type=="Max_Return":
            intermediary_weights=port_maxret(meanS,sigmaS)
        elif type=="Max_Sharpe":
            intermediary_weights=port_maxsr(meanS,sigmaS,rf)
        else:
            raise TypeError("Wrong type input, should be Min_Vol, Target_Return, Max_Return or Max_Sharpe")
        
        weights_matrix_S[i,:]=intermediary_weights
        vect[i,0]=intermediary_weights.T @ meanS
        vect[i,1]=(intermediary_weights.T @ sigmaS @ intermediary_weights) ** 0.5
        i += 1


    standardized_valued = preprocessing.scale(vect, axis=1)
    D = squareform(pdist(standardized_valued, metric="euclidean"))
    sum_dist = D.sum(axis=1)
    clusteroid_idx = np.argmin(sum_dist)
    clusteroid = vect[clusteroid_idx]
    PTF_Weights = weights_matrix_S[clusteroid_idx,:]

    ###to remove after plotting for checking
    original=port_minvol_ro(meanS,sigmaS,targetreturn)
    og_m=original.T @ mean
    og_v=(original.T @ cov @ original) ** 0.5
    col_x, col_y = 1, 0  # the two columns you want from the 2nd dimension

    n_slices = weights_matrix_S.shape[0]

    plt.figure(figsize=(6, 5))

    x_vals = vect[:, 1]  # volatility
    y_vals = vect[:, 0]  # mean return

    plt.figure(figsize=(6,5))
    plt.scatter(x_vals, y_vals, color="blue", label="Simulated portfolios")

    # Clusteroid
    plt.scatter(clusteroid[1], clusteroid[0], color="red", s=80, marker="X", label="Clusteroid")

    # Original portfolio
    plt.scatter(og_v, og_m, color="green", s=80, marker="D", label="Original")

    plt.xlabel("Volatility")
    plt.ylabel("Mean Return")
    plt.title("Simulated Efficient Frontiers + Clusteroid")
    plt.grid(True)
    plt.legend()

    st.pyplot(plt)
    # Returns the clusteriod portfolio 
    return PTF_Weights