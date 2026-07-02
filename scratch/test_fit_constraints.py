import pandas as pd
import numpy as np
from scipy.optimize import minimize

tau = np.array([4.49, 3.86, 3.61, 4.83, 3.22, 3.88, 4.54, 7.28, 6.37, 6.95, 5.99, 3.81, 4.06, 6.43, 3.55, 6.02, 1.86, 4.49])
y = np.array([6.83, 6.62, 5.97, 6.55, 5.97, 6.31, 6.47, 7.13, 6.76, 7.10, 6.86, 6.11, 5.95, 6.90, 5.96, 6.69, 4.29, 5.97])

def nelson_siegel(tau, beta0, beta1, beta2, lmbda):
    tau = np.array(tau, dtype=float)
    if lmbda <= 0:
        return np.zeros_like(tau) + 1e10
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = (1.0 - np.exp(-tau / lmbda)) / (tau / lmbda)
        factor = np.where(tau == 0, 1.0, factor)
        term1 = factor
        term2 = factor - np.exp(-tau / lmbda)
    return beta0 + beta1 * term1 + beta2 * term2

# We want y(t) >= 2.0% for all t >= 0
# Let's write constraints for a grid: 0.0, 0.25, 0.5, 1.0, 1.5
grid_pts = [0.0, 0.25, 0.5, 1.0, 1.5]
constraints = []
for pt in grid_pts:
    constraints.append({
        'type': 'ineq',
        'fun': lambda params, p=pt: nelson_siegel(p, params[0], params[1], params[2], params[3]) - 2.0
    })

bounds = [
    (0.0, 50.0),       # beta0
    (-50.0, 50.0),     # beta1
    (-50.0, 50.0),     # beta2
    (0.1, 20.0)        # lambda
]

initial_guess = [7.95, -7.88, -28.48, 0.18]

def loss_function(params):
    b0, b1, b2, lmb = params
    y_pred = nelson_siegel(tau, b0, b1, b2, lmb)
    return np.sum((y - y_pred) ** 2)

res = minimize(loss_function, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP', options={'maxiter': 1000})
print("Success:", res.success)
print("Params:", res.x)
print("RMSE:", np.sqrt(res.fun/len(tau)))
# print yields at grid points
for pt in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
    print(f"y({pt:.2f}) = {nelson_siegel(pt, *res.x):.4f}%")
