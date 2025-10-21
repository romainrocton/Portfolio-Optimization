# utils/optimization.py
import numpy as np
import scipy.optimize as opt


def port_minvol_ro(mean, cov, ro):
    def objective(W, R, C, ro):
        varp = np.dot(np.dot(W.T, cov), W)
        util = varp ** 0.5
        return util

    n = len(cov)
    W = np.ones([n]) / n
    b_ = [(0.0, 1.0) for _ in range(n)]
    c_ = (
        {"type": "eq", "fun": lambda W: sum(W) - 1.},
        {"type": "eq", "fun": lambda W: np.dot(W.T, mean) - ro},
    )
    optimized = opt.minimize(objective, W, (mean, cov, ro), method="SLSQP", constraints=c_, bounds=b_, options={"maxiter": 100, "ftol": 1e-08})
    return optimized.x


def port_minvol(mean, cov):
    def objective(W, R, C):
        varp = np.dot(np.dot(W.T, cov), W)
        util = varp ** 0.5
        return util

    n = len(cov)
    W = np.ones([n]) / n
    b_ = [(0.0, 1.0) for _ in range(n)]
    c_ = ({"type": "eq", "fun": lambda W: sum(W) - 1.},)
    optimized = opt.minimize(objective, W, (mean, cov), method="SLSQP", constraints=c_, bounds=b_, options={"maxiter": 100, "ftol": 1e-08})
    return optimized.x


def port_maxret(mean, cov):
    def objective(W, R, C):
        meanp = np.dot(W.T, mean)
        util = 1 / meanp
        return util

    n = len(cov)
    W = np.ones([n]) / n
    b_ = [(0.0, 1.0) for _ in range(n)]
    c_ = ({"type": "eq", "fun": lambda W: sum(W) - 1.},)
    optimized = opt.minimize(objective, W, (mean, cov), method="SLSQP", constraints=c_, bounds=b_, options={"maxiter": 100, "ftol": 1e-08})
    return optimized.x

def port_maxsr(mean, cov, rf):
    def objective(W, R, C, rf):
        # calculate mean/variance of the portfolio
        meanp=np.dot(W.T,mean)
        varp=np.dot(np.dot(W.T,cov),W)
        #objective: Max Sharpe ratio
        util=(meanp-rf)/varp**0.5
        return 1/util
    n=len(cov)
    # initial conditions: equal weights
    W=np.ones([n])/n                 
    # weights between 0%..100%: no shorts
    b_=[(0.,1.) for i in range(n)]   
    # No leverage: unitary constraint (sum weights = 100%)
    c_= ({'type':'eq', 'fun': lambda W: sum(W)-1. })
    optimized=opt.minimize(objective,W,(mean,cov,rf),
        method='SLSQP',constraints=c_,bounds=b_,options={'maxiter': 100, 'ftol': 1e-08})
    return optimized.x
