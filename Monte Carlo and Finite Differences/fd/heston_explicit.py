import numpy as np
from scipy.sparse import spdiags
from scipy.interpolate import interp2d


r = 0.03
rho = -0.2
xi = 0.2
kappa = 2
theta = 0.015
K = 1.2
T = 1
Smax = 3
Ymax = 0.1
S0 = 1.3
Y0 = 0.01


def calculate_stencil_coefficients(i, j, Y, I, J, dY, dt, r=0.03, kappa=2, xi=0.2, rho=-0.2, theta=0.015):
    '''
    Calculate the stencil coefficients and return 6 IxI matrices:
    term_c - V(i,j)
    term_e - V(i+1,j)
    term_w - V(i-1,j)
    term_n - V(i, j+1)
    term_s - V(i, j-1)
    term_b - V(i+1,j+1), V(i-1,j-1), V(i-1, j+1), V(i+1, j-1)
    '''
    i.shape = (I,1)
    j.shape = (1,J)
    term1 = np.matmul(i*i, j)
    term2 = np.matmul(i, j)

    # i x j matrices for j and Y
    j = np.meshgrid(j,j)[0]
    Y = np.meshgrid(Y,Y)[0]

    term_c = 1 - dt * (term1 * dY + j * xi ** 2 / dY + r)
    term_e = 0.5 * dt * (term1 * dY + i * r)
    term_w = 0.5 * term1 * dY * dt - 0.5 * dt * i * r

    term_n = 0.5 * dt * (xi ** 2 * j / dY + (kappa / dY) * (theta - Y))
    term_s = 0.5 * dt * (xi ** 2 * j / dY - (kappa / dY) * (theta - Y))
    term_b = 0.25 * xi * rho * term2 * dt

    return term_c, term_e, term_w, term_n, term_s, term_b


def get_matrix(term_c, term_e, term_w, term_n, term_s, term_b, I, J):
    matrix = np.zeros((I, J, I, J))
    for i in range(I):
        data = np.array([term_s[i], term_c[i], term_n[i]])
        mat = spdiags(data, [1, 0, -1], I, J).toarray().transpose()
        matrix[i][i] = mat

    for i in range(I - 1):
        data = np.array([-term_b[i], term_e[i], term_b[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        matrix[i][i + 1] = mat

    for i in range(1, I):
        data = np.array([term_b[i], term_w[i], -term_b[i]])
        mat = spdiags(data, [1, 0, -1], J, I).toarray().transpose()
        matrix[i][i - 1] = mat

    A = matrix.swapaxes(1, 2).reshape((I ** 2, J ** 2))
    return A


def get_smin_boundary_values(V, term_w, initial_val, I):
    w_terms = term_w.copy()
    w_terms[I:] = 0
    w_terms = w_terms.reshape((V.shape))
    smin_boundary = w_terms * initial_val
    return smin_boundary


def get_jmin_boundary_values(V, I, J, i, r, kappa, theta, dt, dY, current_jmin_vals, initial_val, term_s, term_b):
    v = V.copy().reshape(I,J)
    payoff_values_for_j_equals_one = np.array([v[ind][0] for ind in range(I)])
    const = (kappa * theta * dt)/dY
    V_i_1 = const * payoff_values_for_j_equals_one 

    from ipdb import set_trace
    set_trace()

    a = np.array([1- r*dt - const]*I)   
    b = (r*i*dt)/2.       
    data = np.array([-b, a, b])
    mat = spdiags(data, [1, 0, -1], I, I).toarray().transpose()

    current_jmin_vals = current_jmin_vals.reshape(I,J)
    jmin_vals = [current_jmin_vals[ind][0] for ind in range(I)]
    jmin_vals = np.matmul(mat, jmin_vals) + V_i_1
    jmin_vals[0]+=-b[0]*initial_val
    jmin_vals[-1]+=b[0]*jmin_vals[-1]

    jmin_boundaries = np.append(initial_val, np.append(jmin_vals, jmin_vals[-1]))
    jmin_diff = jmin_boundaries[:-2] - jmin_boundaries[2:]

    new_jmin_vals = [term_s[ind][0] for ind in range(I)]*jmin_vals + [term_b[ind][0] for ind in range(I)]*jmin_diff
    
    for ind in range(I):
        current_jmin_vals[ind][0] = new_jmin_vals[ind]

    return current_jmin_vals.reshape(I*J)


def calculate_payoff_for_previous_timestep(A, V, smin_boundary_values, smax_boundary_values, jmin_boundary_values, jmax_boundary_values):
    return np.matmul(A,V) + smin_boundary_values + smax_boundary_values + jmin_boundary_values + jmax_boundary_values


def calculate_price(I=4, J=4, M=4):
    S = np.linspace(0, Smax, I + 1)[1:]
    Y = np.linspace(0, Ymax, J + 1)[1:]


    # V - [V11, V12, V13, V14, V21, V22...V34, V44]
    V = map(lambda x: max(K - x, 0), S)
    V = np.meshgrid(V,V)[0].transpose().reshape(I*J)

    dS = Smax / float(I)
    dY = Ymax / float(J)
    i = S / dS
    j = Y / dY
    dt = T/float(M)

    term_c, term_e, term_w, term_n, term_s, term_b = calculate_stencil_coefficients(i.copy(), j.copy(), Y, I, J, dY, dt)
    A = get_matrix(term_c, term_e, term_w, term_n, term_s, term_b, I, J)

    jmin_boundary_values = V.reshape(I,J)
    for ind in range(I):
        jmin_boundary_values[ind][1:] = 0
    jmin_boundary_values = jmin_boundary_values.reshape(I*J)

    for m in range(M, 0, -1):
        initial_val = K * np.exp(-r * (T - m * dt))
        smin_boundary_values = get_smin_boundary_values(V, term_w, initial_val, J)
        V_prev = V.copy()
        V = np.matmul(A,V) + smin_boundary_values + jmin_boundary_values
        jmin_boundary_values = get_jmin_boundary_values(V_prev, I, J, i, r, kappa, theta, dt, dY, jmin_boundary_values, initial_val, term_s, term_b)


    f = interp2d(Y, S, V.reshape(I, J))
    return f(Y0, S0)

