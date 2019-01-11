import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import spdiags
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import norm, gamma

r = 0.03
rho = -0.2
xi = 0.2
kappa = 2
theta = 0.015
K = 1.2


def create_stock_price_grid(Smax, I):
    """
    Return a linspace vector of S values from Smin to Smax
    """
    S = np.linspace(0, Smax, I + 1)[1:]
    return S


def create_vol_grid(Ymax, J):
    """
    Return a linspace vector of Y values from Ymin to Ymax
    """
    Y = np.linspace(0, Ymax, J + 1)[1:]
    Y.shape = (J, 1)
    return Y


def calculate_stencil_coefficients(i, j, Y, J, dY, dt):
    """
    Calculate the stencil coefficients and return 6 IxI matrices:
    term_c - V(i,j)
    term_e - V(i+1,j)
    term_w - V(i-1,j)
    term_n - V(i, j+1)
    term_s - V(i, j-1)
    term_b - V(i+1,j+1), V(i-1,j-1), V(i-1, j+1), V(i+1, j-1)
    """

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

    term_n = 0.5 * dt * ((xi ** 2 * j) / dY + (kappa / dY) * (theta - Y))
    term_s = 0.5 * dt * ((xi ** 2 * j) / dY - (kappa / dY) * (theta - Y))
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


def get_initial_values(S0, Y0, S, Y, dt, I, J):
    """
    The initial condition for the FPE is a dirac delta function around (S0,Y0). Approximate this using a bivariate normal distribution to find
    probability values for P(0,i,j)
    """
    # stock_var = 0.001
    # vol_var = 0.001
    # term1 = 1 / (2 * 3.14 * np.sqrt(stock_var) * np.sqrt(vol_var))
    # s_terms = map(lambda x: -(1 / (2 * stock_var)) * (x - S0) ** 2, S)
    # y_terms = map(lambda x: -(1 / (2 * vol_var)) * (x - Y0) ** 2, Y)
    # y_terms = np.concatenate(y_terms, axis=0)
    # return term1 * np.array(map(np.exp, [y + s for y in y_terms for s in s_terms]))
    mu = S0*np.exp(r*dt)
    shape = (2 * kappa * theta) / xi ** 2
    rate = shape/theta
    s_values = norm(mu,0.03).pdf(S)
    y_values = gamma.pdf(Y, shape, scale=1/rate)
    s_values.shape = (I,1)
    y_values.shape = (1,J)
    return np.matmul(s_values, y_values).reshape(I*J)


def get_min_vol_boundary_values(P, I, dY, dt):
    """
    Need to fulfil the zero flux condition when Y=0/j=0.
    Generate vector of P(i,0) values and then multiples by relevant coefficients as shown below
    To be added to j=1/P(i,1) terms for P(i, j-1) values: term_n(i,0)P(i,0) + term_b(i-1,0)P(i-1,0) + term_b(i+1,0)P(i+1,0)
    term_b = 0 when j = 0 thus return [term_n(1,0)P(1,0), term_n(2,0)P(2,0), ...] of len I*J
    """
    const = xi ** 2 / (2 * dY * kappa * theta)
    term_n_for_min_j = (dt * kappa * theta) / (2 * dY)
    min_vol_values = np.zeros(P.shape)
    min_vol_values[:I] = term_n_for_min_j * const * P[:I]
    return min_vol_values


def get_max_vol_boundary_values(P, I, J, i, dY, dt):
    """
    Calculate term_s(i,J+1)P(i,J+1) - term_b(i-1,J+1)P(i-1,J) + term_b(i+1,J+1)P(i+1,J) to be added to P(i,J)
    Uses P(i,J+1) = P(i,J) from derivative boundaries
    """
    max_vol_values = P[-I:]
    term_s_for_max_j = 0.5 * dt * ((xi ** 2 * (J + 1)) / dY - (kappa / dY) * (theta - (J + 1) * dY))
    term2 = np.append(0, np.append(i, I + 1)) * (J + 1)
    term_b_for_max_j = 0.25 * xi * rho * term2 * dt
    jmax_boundary = np.zeros(P.shape)
    jmax_boundary[-I:] = term_s_for_max_j * max_vol_values - term_b_for_max_j[:-2] * np.append(0, max_vol_values[:-1]) + term_b_for_max_j[2:]*np.append(max_vol_values[1:], max_vol_values[-1])
    return jmax_boundary


# def get_max_stock_boundary_values(P, I, J, i, dY, dt):
#     """
#     Calculate term_s(i,J+1)P(i,J+1) - term_b(i-1,J+1)P(i-1,J) + term_b(i+1,J+1)P(i+1,J) to be added to P(i,J)
#     Uses P(i,J+1) = P(i,J) from derivative boundaries
#     """
#     max_vol_values = P[-I:]
#     term_s_for_max_j = 0.5 * dt * ((xi ** 2 * (J + 1)) / dY - (kappa / dY) * (theta - (J + 1) * dY))
#     term2 = np.append(0, np.append(i, I + 1)) * (J + 1)
#     term_b_for_max_j = 0.25 * xi * rho * term2 * dt
#     jmax_boundary = np.zeros(P.shape)
#     jmax_boundary[-I:] = term_s_for_max_j * max_vol_values - term_b_for_max_j[:-2] * np.append(0, max_vol_values[:-1]) + term_b_for_max_j[2:]
#     np.append(max_vol_values[2:], max_vol_values[-1])
#     return jmax_boundary

def create_plot(S, Y, P, I, J):
    """
    Surface plot of transition probability density
    """
    s_grid = np.meshgrid(S, S)[0]
    y_grid = np.meshgrid(Y, Y)[0].transpose()
    p_grid = P.reshape(I, J)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(s_grid, y_grid, p_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def plot_transition_density(I=4, J=4, M=4):
    # set parameters
    T = 1
    Smax = 3
    Ymax = 0.1
    S0 = 1.3
    Y0 = 0.01

    # create grid
    dt = T / float(M)
    S = create_stock_price_grid(Smax, I)
    Y = create_vol_grid(Ymax, J)
    P = get_initial_values(S0, Y0, S, Y, dt, I, J)

    # calculate i and j matrices
    dS = Smax / float(I)
    dY = Ymax / float(J)
    i = S / dS
    i.shape = (1, I)
    j = Y / dY

    # calculate matrix A^T
    term_c, term_e, term_w, term_n, term_s, term_b = calculate_stencil_coefficients(i, j, Y, J, dY, dt)
    A = get_matrix(term_c, term_e, term_w, term_n, term_s, term_b, I, J)
    A = A.transpose()

    for m in range(M):
        min_vol_boundary_values = get_min_vol_boundary_values(P, I, dY, dt)
        max_vol_boundary_values = get_max_vol_boundary_values(P, I, J, i, dY, dt)
        P = np.matmul(A, P) + max_vol_boundary_values + min_vol_boundary_values

    create_plot(S, Y, P, I, J)

