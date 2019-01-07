import numpy as np
from scipy.sparse import spdiags

# set constants
T = 3
r = 0.03
rho = -0.2
xi = 0.2
kappa = 2
theta = 0.0015

# generate grid for S
S0 = 1.3
Smax = 3
I = 4
dS = (Smax - 0)/float(I)
S = np.linspace(S0, Smax, I+1)

# generate grid for Y
Y0 = 0.01
Ymax = 0.1
J = 4
dY = (Ymax-0)/float(J)
Y = np.linspace(Y0, Ymax, J+1)
Y.shape = (J+1,1)

# number of time steps
M = 4
dt = T/float(M)

# boundary conditions
# boundary at maturity
K = 1.5
V = map(lambda x: max(K-x,0), S)
V = np.concatenate([V]*(I+1), axis=0)

# calculate coefficients
i = S/dS
i.shape = (1, I+1)
j = Y/dY
# j.shape = (J+1, 1)

# create matrices of shape J+1 x I+1 - e.g.first row of term_c will have a(11) a(21) a(31) a(41) prefixed a(ij)
term1 = np.matmul(j, i*i)
term2 = np.matmul(j,i)

j = np.array(map(lambda x:[x[0]]*(J+1), j))
Y = np.array(map(lambda x:[x[0]]*(J+1), Y))

term_c = 1-term1*dt*dY -j*dt*xi**2/dY -r*dt
term_e = 0.5*term1*dY*dt + 0.5*dt*i*r
term_w = 0.5*term1*dY*dt - 0.5*dt*i*r

term_n = 0.5*dt*(xi**2*j/dY + (kappa/dY)*(theta-Y))
term_s = 0.5*dt*(xi**2*j/dY - (kappa/dY)*(theta-Y))
term_b = 0.25*xi*rho*term2*dt

# final matrix will be (J+1)(I+1) x (J+1)(I+1)
matrix = np.zeros((J+1,I+1,J+1,I+1))

for i in range(I+1):
    data = np.array([term_w[i], term_c[i], term_e[i]])
    mat = spdiags(data, [1,0,-1], J+1, I+1).toarray().transpose()
    matrix[i][i] = mat

for i in range(I):
    data = np.array([-term_b[i], term_n[i], term_b[i]])
    mat = spdiags(data, [1,0,-1], J+1, I+1).toarray().transpose()
    matrix[i][i+1] = mat

for i in range(1,I+1):
    data = np.array([term_b[i], term_s[i], -term_b[i]])
    mat = spdiags(data, [1,0,-1], J+1, I+1).toarray().transpose()
    matrix[i][i-1] = mat

A = matrix.swapaxes(1,2).reshape((25,25))
for i in range(M):
    V = np.matmul(A,V)

print V
