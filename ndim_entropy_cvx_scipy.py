# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:01:58 2025

@author: sable
"""

import cvxpy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
dim = 5
p = 0.1
dims = np.array([2,3,4,5,10])#,30])
dim_time = np.array([])
dim_evals = np.array([])
for dim in dims:
    def project_rho(rho_guess):
        """
        Project an arbitrary matrix onto the PSD + trace=1 set.
        Uses CVXPY for exact constraint satisfaction.
        """
        rho = cp.Variable((dim, dim), complex=True)
        constraints = [
            rho >> 0,       # PSD
            cp.trace(rho) == 1,
            rho == rho.H    # Hermitian
        ]
        # Minimize Frobenius distance to rho_guess
        obj = cp.Minimize(cp.norm(rho - rho_guess, "fro"))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)
        return rho.value
    def von_neumann_entropy(rho):
        # Depolarizing channel
        rho_q = (1 - p) * rho + (p / dim) * np.eye(dim)
        eigvals = np.linalg.eigvalsh(rho_q)
        eigvals = np.clip(eigvals, 1e-12, 1.0)  # avoid log(0)
        return -np.sum(eigvals * np.log2(eigvals))
    from scipy.optimize import minimize
    
    # Flatten rho into a vector for optimization
    def rho_to_vec(rho):
        return np.hstack([rho.real.flatten(), rho.imag.flatten()])
    
    def vec_to_rho(x):
        n = dim
        re = x[:n*n].reshape((n, n))
        im = x[n*n:].reshape((n, n))
        rho = re + 1j*im
        # Project onto feasible set
        return project_rho(rho)
    
    # Objective in terms of vector x
    def objective(x):
        rho = vec_to_rho(x)
        return von_neumann_entropy(rho)
    
    # Initial guess: random Hermitian matrix
    rho0 = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho0 = (rho0 + rho0.conj().T)/2
    x0 = rho_to_vec(rho0)
    
    # Use derivative-free optimizer
    initial_time = time.time()
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter':1000})
    operation_time = time.time()-initial_time
    dim_time=np.append(dim_time,operation_time)
    dim_evals = np.append(dim_evals,res.nfev)
    print("General Time to Optimal:", operation_time)
    print("# of Evals:", res.nfev)
    #rho_opt = vec_to_rho(res.x)
    #S_min = von_neumann_entropy(rho_opt)

    #print("Optimized entropy:", S_min)
    #print("Eigenvalues of rho:", np.linalg.eigvalsh(rho_opt))

plt.figure(figsize=(7, 4))
plt.plot(dims, dim_time, 'o-', linewidth=2)
plt.xlabel("Matrix dimension (dim)")
plt.ylabel("Optimization time (seconds)")
plt.title("Scaling of von Neumann entropy optimization with dim")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print(dim_time)
plt.figure(figsize=(7, 4))
plt.plot(dims, dim_evals, 's-', color='orange', linewidth=2)
plt.xlabel("Matrix dimension (dim)")
plt.ylabel("Number of function evaluations")
plt.title("Scaling of function evaluations with matrix dimension")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print(dim_evals)

def algor_check(dim):

    correct_value = -((1 - p + p/dim) * np.log2(1 - p + p/dim) + (dim - 1) * (p/dim) * np.log2(p/dim))
    
    return(correct_value)

#print(algor_check(dim))
#Results: dimensions 3: 10 sec, 5- (45 seconds,1492 , 10- (58 seconds, 1206 evals) 20- (146 seconds,1801 evals) 30- 1100/386 seconds 50-81.1 minutes