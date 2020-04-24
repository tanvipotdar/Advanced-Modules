import numpy as np
from scipy.sparse import spdiags
from scipy.interpolate import interp2d


def get_solution(I=4, J=4, M=4, T=3, S0=1.3, Y0=0.01):
    '''
    Computes the solution for a european put using the explicit scheme for the heston model
    Parameters
    ----------
    I - number of S steps in grid
    J - number of Y steps in grid
    M - number of time steps
    T - maturity
    S0 - value of S the option should be priced at
    Y0 - value of Y the option should be priced at

    Returns
    -------
    Option price given S0 and Y0
    '''
    # set constants
    r = 0.03
    rho = -0.2
    xi = 0.2
    kappa = 2
    theta = 0.0015

    # generate grid for S
    Smin = 0
    Smax = 3
    dS = (Smax - Smin) / float(I)
    S = np.linspace(Smin, Smax, I + 1)[1:]

    # generate grid for Y
    Ymin = 0
    Ymax = 0.1
    dY = (Ymax - Ymin) / float(J)
    Y = np.linspace(Ymin, Ymax, J + 1)[1:]
    Y.shape = (J, 1)

    # number of time steps
    dt = T / float(M)

    # terminal condition
    K = 1.2
    # create a vector V = [V11 V21 V31 V41, V21 V22 V32 V42, ...]^T
    V = map(lambda x: max(K - x, 0), S)
    V = np.concatenate([V] * I, axis=0)

    # calculate coefficients
    i = S / dS
    i.shape = (1, I)
    j = Y / dY
    # j.shape = (J+1, 1)

    # create matrices of shape J x I - e.g.first row of term_c will have a(11) a(21) a(31) a(41) prefixed a(ij)
    term1 = np.matmul(j, i * i)
    term2 = np.matmul(j, i)

    j = np.array(map(lambda x: [x[0]] * J, j))
    Y = np.array(map(lambda x: [x[0]] * J, Y))

    term_c = 1 - dt * (term1 * dY + j * xi ** 2 / dY + r)
    term_e = 0.5 * dt * (term1 * dY + i * r)
    term_w = 0.5 * term1 * dY * dt - 0.5 * dt * i * r

    term_n = 0.5 * dt * (xi ** 2 * j / dY + (kappa / dY) * (theta - Y))
    term_s = 0.5 * dt * (xi ** 2 * j / dY - (kappa / dY) * (theta - Y))
    term_b = 0.25 * xi * rho * term2 * dt

    # final matrix will be IJ x IJ
    matrix = np.zeros((J, I, J, I))

    for i in range(I):
        data = np.array([term_w[i], term_c[i], term_e[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        matrix[i][i] = mat

    for i in range(I - 1):
        data = np.array([-term_b[i], term_n[i], term_b[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        matrix[i][i + 1] = mat

    for i in range(1, I):
        data = np.array([term_b[i], term_s[i], -term_b[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        matrix[i][i - 1] = mat

    A = matrix.swapaxes(1, 2).reshape((I ** 2, J ** 2))

    # boundary conditions
    b_terms = term_b.copy()
    w_terms = term_w.copy()
    e_terms = term_e.copy()
    for i in range(I):
        w_terms[i][1:].fill(0)
        e_terms[i][:-1].fill(0)
        b_terms[i][:-1].fill(0)

    for m in range(M, -1, -1):
        # when S=0/i=0, option val is discounted K
        initial_val = K * np.exp(-r * (T - m * dt))

        # boundary for j=J/ stencil terms include i,J+1
        jmax_values = term_n[-1] * V[-I:] + term_b[-1] * np.append(V[-(I - 1):], V[-1]) - term_b[-1] * np.append(initial_val, V[-I:-1])
        jmax_boundary_vals = np.zeros(V.shape)
        jmax_boundary_vals[-I:] = jmax_values

        # boundary for i=I/ stencil terms include I+1,j
        e_terms = e_terms.reshape(V.shape)
        e_boundary_vals = e_terms * V
        v_vals = [x for i, x in enumerate(V) if (i + 1) % I == 0]
        v_vals = np.append(0, np.append(v_vals, 0))
        v_vals = v_vals[2:] - v_vals[:-2]
        b_vals = np.array([x[-1] for x in term_b])
        b_boundary_vals = b_vals * v_vals
        for i in range(0, (I + 1) * J, J)[:-1]:
            b_boundary_vals = np.insert(b_boundary_vals, i, tuple([0] * (J - 1)))
        smax_boundary = e_boundary_vals + b_boundary_vals

        # added to values where i=1/ terms for i = 0
        w_terms = w_terms.reshape((V.shape))
        smin_boundary = w_terms * initial_val

        V = np.matmul(A, V) + smin_boundary + jmax_boundary_vals + smax_boundary

    f = interp2d(S, [x[0] for x in Y], V.reshape(I, J))
    return f(S0, Y0)


print get_solution()
