import numpy as np
from scipy.sparse import spdiags
from scipy.interpolate import interp2d


def create_stock_price_grid(Smin, Smax, I):
    S = np.linspace(Smin, Smax, I + 1)[1:]
    return S

def create_vol_grid(Ymin, Ymax, J):
    Y = np.linspace(Ymin, Ymax, J + 1)[1:]
    Y.shape = (J, 1)
    return Y

def get_payoff_values_at_maturity(K, S, J):
    V = map(lambda x: max(K - x, 0), S)
    V = np.concatenate([V] * J, axis=0)
    return V

def calculate_stencil_coefficients(i, j, Y, J, dY, dt, r=0.03, kappa=2, xi=0.2, rho=-0.2, theta=0.015):
    term1 = np.matmul(j, i * i)
    term2 = np.matmul(j, i)

    j = np.array(map(lambda x: [x[0]] * J, j))
    Y = np.array(map(lambda x: [x[0]] * J, Y))

    j = np.array(map(lambda x: [x[0]] * J, j))
    Y = np.array(map(lambda x: [x[0]] * J, Y))

    term_c = 1 - dt * (term1 * dY + j * xi ** 2 / dY + r)
    term_e = 0.5 * dt * (term1 * dY + i * r)
    term_w = 0.5 * term1 * dY * dt - 0.5 * dt * i * r

    term_n = 0.5 * dt * (xi ** 2 * j / dY + (kappa / dY) * (theta - Y))
    term_s = 0.5 * dt * (xi ** 2 * j / dY - (kappa / dY) * (theta - Y))
    term_b = 0.25 * xi * rho * term2 * dt

    return term_c, term_e, term_w, term_n, term_s, term_b

def get_matrix(term_c, term_e, term_w, term_n, term_s, term_b, I, J):
    matrix = np.zeros((J, I, J, I))
    for i in range(I):
        data = np.array([term_w[i], term_c[i], term_e[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        mat[I-1][I-1]+=term_e[i][-1]
        matrix[i][i] = mat

    for i in range(I - 1):
        data = np.array([-term_b[i], term_n[i], term_b[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        mat[I-1][I-1]+=term_b[i][-1]
        matrix[i][i + 1] = mat

    for i in range(1, I):
        data = np.array([term_b[i], term_s[i], -term_b[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        mat[I-1][I-1]+=-term_b[i][-1]
        matrix[i][i - 1] = mat

    jmax_data = np.array([-term_b[I-1], term_n[I-1], term_b[I-1]])
    jmax_mat = spdiags(jmax_data, [1, 0, -1], J, I).toarray().transpose()
    matrix[J-1][I-1]+=jmax_mat


    A = matrix.swapaxes(1, 2).reshape((I ** 2, J ** 2))
    return A

def get_smin_boundary_values(V, term_w, initial_val, I):
    w_terms = term_w.copy()
    for i in range(I):
        w_terms[i][1:].fill(0)
    w_terms = w_terms.reshape((V.shape))
    smin_boundary = w_terms * initial_val
    return smin_boundary


def get_jmin_boundary_values(V, I, J, i, r, kappa, theta, dt, dY, jmins, initial_val, term_s, term_b):
    payoff_values_for_j_equals_one = V[:J]
    const = (kappa * theta * dt)/dY
    V_i_1 = const * payoff_values_for_j_equals_one 

    a = np.array([1- r*dt - const]*I)   
    b = (r*i*dt)/2
    b = b[0]

    data = np.array([-b, a, b])
    mat = spdiags(data, [1, 0, -1], I, I).toarray().transpose()

    jmin_vals = jmins[:J]
    jmin_vals = np.matmul(mat, jmin_vals) + V_i_1
    jmin_vals[0]+=-b[0]*initial_val
    jmin_vals[-1]+=b[0]*jmin_vals[-1]

    jmin_boundaries = np.append(initial_val, np.append(jmin_vals, jmin_vals[-1]))
    jmin_diff = jmin_boundaries[:-2] - jmin_boundaries[2:]

    jmins[:J] = term_s[0]*jmin_vals + term_b[0]*jmin_diff
    return jmins


def compute_price(I=4, J=4, M=4):
    r = 0.03
    rho = -0.2
    xi = 0.2
    kappa = 2
    theta = 0.015
    K = 1.2
    T = 3

    Smin = 0
    Ymin = 0
    Smax = 3
    Ymax = 0.1

    S0 = 1.3
    Y0 = 0.01

    S = create_stock_price_grid(Smin, Smax, I)
    Y = create_vol_grid(Ymin, Ymax, J)
    V = get_payoff_values_at_maturity(K, S, J)

    dS = (Smax - Smin) / float(I)
    dY = (Ymax - Ymin) / float(J)
    i = S / dS
    i.shape = (1,I)
    j = Y / dY
    dt = T/float(M)

    term_c, term_e, term_w, term_n, term_s, term_b = calculate_stencil_coefficients(i, j, Y, J, dY, dt)
    A = get_matrix(term_c, term_e, term_w, term_n, term_s, term_b, I, J)

    jmin_boundary_values = np.zeros(V.shape)
    jmin_boundary_values[:J] = V[:J]
    for m in range(M, 0, -1):
        initial_val = K * np.exp(-r * (T - m * dt))
        smin_boundary_values = get_smin_boundary_values(V, term_w, initial_val, I)

        V_prev = V.copy()

        V = np.matmul(A,V) + smin_boundary_values + jmin_boundary_values
        jmin_boundary_values = get_jmin_boundary_values(V_prev, I, J, i, r, kappa, theta, dt, dY, jmin_boundary_values, initial_val, term_s, term_b)

    f = interp2d(S, Y, V.reshape(I, J))
    return f(S0, Y0)
    