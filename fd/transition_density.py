from heston import create_stock_price_grid, create_vol_grid, calculate_stencil_coefficients, get_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.stats import norm, gamma
from scipy.interpolate import interp1d
from scipy.integrate import simps, cumtrapz
from scipy.sparse import spdiags


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


def get_initial_values(S0, Y0, S, Y, dt, I, J):
    stock_var = 0.001
    vol_var = 0.001
    term1 = 1 / (2 * 3.14 * np.sqrt(stock_var) * np.sqrt(vol_var))
    s_terms = map(lambda x: -(1 / (2 * stock_var)) * (x - S0) ** 2, S)
    y_terms = map(lambda x: -(1 / (2 * vol_var)) * (x - Y0) ** 2, Y)
    y_terms = np.concatenate(y_terms, axis=0)
    return term1 * np.array(map(np.exp, [y + s for y in y_terms for s in s_terms]))


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


def calculate_transition_density_and_var(P, S, Y, I, J, M):
    dt = T / float(M)
    dS = Smax / float(I)
    dY = Ymax / float(J)
    i = S / dS
    i.shape = (1, I)
    j = Y / dY

    term_c, term_e, term_w, term_n, term_s, term_b = calculate_stencil_coefficients(i, j, Y, J, dY, dt)
    A = get_matrix(term_c, term_e, term_w, term_n, term_s, term_b, I, J)
    A = A.transpose()

    var = np.zeros((M,I))

    for m in range(M):
        P = np.matmul(A, P)
        var[m] = compute_var_for_timestep(P,Y,I,J,dY)

    return P, var


def compute_var_for_timestep(P, Y, I, J, dY):
    P = P.reshape(I,J)
    Y.shape = (1,J)
    y = Y[0]
    var = simps(P.transpose()*y,y,axis=1)/simps(P,y, axis=0)
    return var


def plot_variance(S, I, M, var):
    s_grid = np.meshgrid(S, S)[0]
    t_grid = np.array([[x]*I for x in range(1,M+1)])
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Time')
    ax.set_zlabel('Variance')
    ax.plot_surface(s_grid, t_grid, var, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def plot_transition_density(S, Y, I, J, P):
    s_grid = np.meshgrid(S, S)[0]
    y_grid = np.meshgrid(Y, Y)[0].transpose()
    p_grid = P.reshape(I, J)
    fig = plt.figure(figsize=(10,7))
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Variance')
    ax.set_zlabel('Transition Density')
    ax.plot_surface(s_grid, y_grid, p_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def create_plots(I=4, J=4, M=4):
    dt = T / float(M)
    S = create_stock_price_grid(0, Smax, I)
    Y = create_vol_grid(0, Ymax, J)
    initial_p = get_initial_values(S0, Y0, S, Y, dt, I, J)
    P, var = calculate_transition_density_and_var(initial_p, S, Y, I, J, M)
    plot_transition_density(S, Y, I, J, P)
    plot_variance(S, I, M, var)


def compute_price(I=4, J=4, M=4):
    T=3;
    r=0.03;
    S = create_stock_price_grid(0, Smax, I)
    Y = create_vol_grid(0, Ymax, J)
    dt = T / float(M)
    dS = Smax / float(I)
    i = S / dS

    V = map(lambda x: max(K - x, 0), S)

    initial_p = get_initial_values(S0, Y0, S, Y, dt, I, J)
    _, vol = calculate_transition_density_and_var(initial_p, S, Y, I, J, M)

    for m in range(M, 0, -1):
        initial_val = K * np.exp(-r * (T - m * dt))
        sigma = vol[m-1]
        term1=sigma*i
        term1=term1*term1
        term2=r*i

        A=0.5*dt*(term1-term2)
        B=1-dt*(term1+r)
        C=0.5*dt*(term1+term2)

        data = np.array([C, B, A])
        mat = spdiags(data, [-1, 0, 1], J, I).toarray().transpose()
        V=np.matmul(mat,V)
        V[0] += initial_val * A[0]

    f = interp1d(S, V)
    return f(S0)

