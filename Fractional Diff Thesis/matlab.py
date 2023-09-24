import numpy as np
from scipy.special import gamma

def MT_FDE_PI1_Im(al, lam, f_fun, J_fun, t0, T, y0, h, param=None, tol=1.0e-6, itmax=100):
    # Check inputs
    if param is None:
        param = []

    # Check order of the multiterm FDE
    if any(al < 0):
        raise ValueError("The orders ALPHA of the muliterm FDE must be positive. One of the values cannot be accepted.")

    # Check the step-size of the method
    if h <= 0:
        raise ValueError("The step-size H for the method must be positive. The value H = {} cannot be accepted.".format(h))

    # Ascending ordering (w.r.t. to the fractional order) of the terms of the equation and transformation in normal form
    Q = len(al)
    al_i = np.sort(al)
    i_al = np.argsort(al)
    lam = [lam[i] for i in i_al]
    al_Q = al_i[-1]
    al_i = al_i[:-1]
    lam_Q = lam[-1]
    lam_rat_i = np.array(lam[:-1]) / lam_Q
    m_Q = int(np.ceil(al_Q))
    m_i = [int(np.ceil(a)) for a in al_i]
    beta = [al_Q - a for a in al_i] + [al_Q]

    # Structure for storing initial conditions
    ic = {
        "t0": t0,
        "problem_size": y0.shape[0],
        "y0": y0,
        "Q": Q,
        "m_Q": m_Q,
        "m_i": m_i,
        "beta": beta,
        "lam_rat_i": lam_rat_i,
    }

    gamma_val = np.zeros((Q, m_Q))
    for i in range(Q - 1):
        k = np.arange(m_i[i])
        gamma_val[i, k] = gamma(k + beta[i] + 1)
    k = np.arange(m_Q)
    gamma_val[Q - 1, k] = np.math.factorial(k)

    ic["gamma_val"] = gamma_val

    # Structure for storing information on the problem
    Probl = {
        "ic": ic,
        "fdefun": f_fun,
        "Jfdefun": J_fun,
        "problem_size": y0.shape[0],
        "param": param,
        "Q": Q,
        "lam_Q": lam_Q,
        "lam_rat_i": lam_rat_i,
    }

    # Check number of initial conditions
    if y0.shape[1] < m_Q:
        raise ValueError("A not sufficient number of initial conditions are assigned. Y0 must have as many columns as the number of derivatives at the origin involved by initial conditions ({} for this problem).".format(m_Q))

    # Check compatibility size of the problem with size of the vector field
    f_temp = f_vectorfield(t0, y0[:, 0], Probl)
    if Probl["problem_size"] != f_temp.shape[0]:
        raise ValueError("Size {} of the problem as obtained from initial conditions (i.e., the number of rows of Y0) not compatible with the size {} of the output of the vector field F_FUN.".format(Probl["problem_size"], f_temp.shape[0]))

    # Number of points in which to evaluate the solution or the weights
    r = 64
    N = int(np.ceil((T - t0) / h))
    Nr = int(np.ceil((N + 1) / r) * r)
    Qr = int(np.ceil(np.log2(Nr / r))) - 1
    NNr = 2**(Qr + 1) * r

    # Preallocation of some variables
    y = np.zeros((Probl["problem_size"], N + 1))
    fy = np.zeros((Probl["problem_size"], N + 1))
    zn = np.zeros((Probl["problem_size"], NNr + 1, Q))

    # Evaluation of weights of the method
    nvett = np.arange(NNr + 1)
    bn = np.zeros((Q, NNr + 1))
    for i in range(Q):
        nbeta = nvett**beta[i]
        bn[i, :] = (nbeta[1:] - nbeta[:-1]) * h**beta[i] / gamma(beta[i] + 1)
    C = 0
    for i in range(Q - 1):
        C += lam_rat_i[i] * bn[i, 0]

    METH = {"bn": bn, "C": C, "itmax": itmax, "tol": tol}

    # Initializing solution and process of computation
    t = np.arange(N + 1) * h
    y[:, 0] = y0[:, 0]
    fy[:, 0] = f_vectorfield(t0, y0[:, 0], Probl)
    y, fy = Triangolo(1, r - 1, t, y, fy, zn, N, METH, Probl)

    # Main process of computation by means of the FFT algorithm
    ff = np.zeros(2**(Qr + 2))
    ff[0:2] = [0, 2]
    card_ff = 2
    nx0 = 0
    ny0 = 0

    for qr in range(Qr + 1):
        L = 2**qr
        y, fy = DisegnaBlocchi(L, ff, r, Nr, nx0 + L * r, ny0, t, y, fy, zn, N, METH, Probl)
        ff[0:2 * card_ff] = np.hstack((ff[0:card_ff], ff[0:card_ff]))
        card_ff = 2 * card_ff
        ff[card_ff] = 4 * L

    # Evaluation solution in TFINAL when TFINAL is not in the mesh
    if T < t[N]:
        c = (T - t[N]) / h
        t[N] = T
        y[:, N] = (1 - c) * y[:, N] + c * y[:, N]

    t = t[:N + 1]
    y = y[:, :N + 1]

    return t, y

# Replace f_vectorfield with the actual implementation of your vector field function
def f_vectorfield(t, y, Probl):
    # Implement your vector field function here
    pass

# Implement the Triangolo and DisegnaBlocchi functions here

# Example usage:
# t, y = MT_FDE_PI1_Im(al, lam, f_fun, J_fun, t0, T, y0, h, param, tol, itmax)


# Example usage:
if __name__ == "__main__":
    # Define your input parameters
    al = [0.5, 0.7]
    lam = [1.0, 2.0]
    f_fun = lambda t, y: y  # Replace with your function
    J_fun = lambda t, y: np.eye(len(y))  # Replace with your Jacobian function
    t0 = 0.0
    T = 1.0
    y0 = np.array([1.0, 0.0])  # Initial conditions
    h = 0.1
    param = None
    tol = 1.0e-6
    itmax = 100
    
    t, y = MT_FDE_PI1_Im(al, lam, f_fun, J_fun, t0, T, y0, h, param, tol, itmax)
    print(t)
    print(y)
