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
