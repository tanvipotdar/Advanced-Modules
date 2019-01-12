import numpy as np
from scipy.sparse import spdiags
from scipy.interpolate import interp2d



def create_stock_price_grid(Smin, Smax, I):
    '''
    Return a linspace vector of S values from Smin to Smax
    '''
    S = np.linspace(Smin, Smax, I + 1)[1:]
    return S

def create_vol_grid(Ymin, Ymax, J):
    '''
    Return a linspace vector of Y values from Ymin to Ymax
    '''
    Y = np.linspace(Ymin, Ymax, J + 1)[1:]
    Y.shape = (J, 1)
    return Y

def get_payoff_values_at_maturity(K, S, J):
    '''
    Return a vector V = [V11 V21 V31 V41, V21 V22 V32 V42, ...]^T
    '''
    V = map(lambda x: max(K - x, 0), S)
    V = np.concatenate([V] * J, axis=0)
    return V


def calculate_stencil_coefficients(i, j, Y, J, dY, dt, r=0.03, kappa=2, xi=0.2, rho=-0.2, theta=0.015):
    '''
    Calculate the stencil coefficients and return 6 IxI matrices:
    term_c - V(i,j)
    term_e - V(i+1,j)
    term_w - V(i-1,j)
    term_n - V(i, j+1)
    term_s - V(i, j-1)
    term_b - V(i+1,j+1), V(i-1,j-1), V(i-1, j+1), V(i+1, j-1)
    '''

    # i has shape [1,2,3,4] and j has shape [[1], [2], [3], [4]] so j x i gives a matrix with structure
    # a(11) a(21) a(31) a(41)
    # a(12) a(22) a(32) a(42) ... prefixed a(ij)
    term1 = np.matmul(j, i * i)
    term2 = np.matmul(j, i)

    # j x i matrices for j and Y
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
    return A


def get_smin_boundary_values(V, term_w, initial_val, I):
    '''
    Return an array with values filled for V11, V12, V13, V14
    array contains values for P(i-1,j) when i=1 so P(0,j) with coefficient term_w (west)
    Return  -  [w(0,1), 0, 0, 0, w(0,2), 0, 0, 0, ...]
    '''
    w_terms = term_w.copy()
    for i in range(I):
        w_terms[i][1:].fill(0)
    w_terms = w_terms.reshape((V.shape))
    smin_boundary = w_terms * initial_val
    return smin_boundary


def get_smax_boundary_values(V, term_e, term_b, I, J):
    '''
    Return an array with values filled for V41, V42, V43 
    array contains values: term_e(I,j)*P(I+1,j) + term_b*(-P(I+1, j-1) + P(I+1, j+1))

    We know from discretised boundary condition (5) that P(I+1,j) = P(I,j)

    term_e(I,j)*P(I+1,j) + term_b*(-P(I+1, j-1) + P(I+1, j+1)) = term_e(I,j)*P(I,j) + term_b*(-P(I, j-1) + P(I, j+1))
    Return - [0,0,0,w(4,1),0,0,0,w(4,2),0...]
    '''
    b_terms = term_b.copy()
    e_terms = term_e.copy()
    for i in range(I):
        e_terms[i][:-1].fill(0)
        b_terms[i][:-1].fill(0)

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
    return smax_boundary


def get_jmin_boundary_values(V, I, J, i, r, kappa, theta, dt, dY, jmins, initial_val, term_s, term_b):
    '''
    Return an array with values filled for V11, V21, V31, V41 as those are the terms to add to
    array contains values for V(i,j-1), V(i+1, j-1), V
    Discretise boundary condition which is a pde and use explicit scheme to calculate V^{m-1}_(i,0) given V^m_(i,0)
    Return - [w(1,0),w(2,0),w(3,0),w(4,), 0...0]
    '''
    payoff_values_for_j_equals_one = V[:J]
    const = (kappa * theta * dt)/dY
    V_i_1 = const * payoff_values_for_j_equals_one 

    a = np.array([1- r*dt - const]*I)   
    b = (r*i*dt)/2
    b = b[0]

    data = np.array([-b, a, b])
    mat = spdiags(data, [1, 0, -1], I, I).toarray().transpose()

    jmin_vals = jmins[:J]
    jmin_vals = np.matmul(mat, jmin_vals) + V_i_1 + np.array([-b[0]*initial_val]*I) + np.array([b[0]*jmin_vals[-1]]*I)

    jmin_boundaries = np.append(initial_val, np.append(jmin_vals, jmin_vals[-1]))
    jmin_diff = jmin_boundaries[:-2] - jmin_boundaries[2:]

    jmins[:J] = term_s[0]*jmin_vals + term_b[0]*jmin_diff
    return jmins
    

def get_jmax_boundary_values(V, term_n, term_b, initial_val, I):
    '''
    Return an array with values filled for V14, V24, V34, V44 as those are the terms to add to
    array contains values: term_e(I,j)*P(I+1,j) + term_b*(-P(I+1, j-1) + P(I+1, j+1))

    We know from discretised boundary condition (5) that P(I+1,j) = P(I,j)

    term_e(I,j)*P(I+1,j) + term_b*(-P(I+1, j-1) + P(I+1, j+1)) = term_e(I,j)*P(I,j) + term_b*(-P(I, j-1) + P(I, j+1))
    Return - [0,0,0,...,w(4,1),w(4,2),w(4,3),w(4,4)]
    '''
    jmax_values = term_n[-1] * V[-I:] + term_b[-1] * np.append(V[-(I - 1):], V[-1]) - term_b[-1] * np.append(initial_val, V[-I:-1])
    jmax_boundary = np.zeros(V.shape)
    jmax_boundary[-I:] = jmax_values
    return jmax_boundary


def calculate_payoff_for_previous_timestep(A, V, smin_boundary_values, smax_boundary_values, jmin_boundary_values, jmax_boundary_values):
    return np.matmul(A,V) + smin_boundary_values + smax_boundary_values + jmin_boundary_values + jmax_boundary_values


def compute_price(I=4, J=4, M=4):
    # set constants
    r = 0.03
    rho = -0.2
    xi = 0.2
    kappa = 2
    theta = 0.015
    K = 1.2
    T = 1

    # grid parameters
    Smin = 0
    Ymin = 0
    Smax = 3
    Ymax = 0.1

    # values for solution
    S0 = 1.3
    Y0 = 0.01

    S = create_stock_price_grid(Smin, Smax, I)
    Y = create_vol_grid(Ymin, Ymax, J)
    V = get_payoff_values_at_maturity(K, S, J)

    # calculate i and j matrices
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
    for m in range(1,M+1):
        initial_val = K * np.exp(-r * (T - m * dt))
        smin_boundary_values = get_smin_boundary_values(V, term_w, initial_val, I)
        smax_boundary_values = get_smax_boundary_values(V, term_e, term_b, I, J)
        jmax_boundary_values =get_jmax_boundary_values(V, term_n, term_b, initial_val, I)

        V_prev = V.copy()

        V = calculate_payoff_for_previous_timestep(A, V, smin_boundary_values, smax_boundary_values, jmin_boundary_values, jmax_boundary_values)
        jmin_boundary_values = get_jmin_boundary_values(V_prev, I, J, i, r, kappa, theta, dt, dY, jmin_boundary_values, initial_val, term_s, term_b)

    f = interp2d(S, Y, V.reshape(I, J))
    return f(S0, Y0)

