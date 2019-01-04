import numpy as np

# set constants
T = 3
r = 0.03
rho = -0.2
xi = 0.2
kappa = 2
theta = 0.0015

# generate grid for S
# TODO: make this 1.3
S0 = 0
Smax = 3
I = 4
dS = (Smax - S0)/float(I)
S = np.linspace(S0, Smax, I+1)

# generate grid for Y
# TODO: mak this 0.01
Y0 = 0.0
Ymax = 0.1
J = 4
dY = (Ymax-Y0)/float(J)
Y = np.linspace(Y0, Ymax, J+1)
Y.shape = (J+1,1)

# number of time steps
M = 4
dt = T/float(M)

# boundary conditions
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

j = map(lambda x:[x[0]]*5, j)
Y = map(lambda x:[x[0]]*5, Y)

term_c = 1-term1*dt*dY -j*dt*xi**2/dY -r*dt
term_e = 0.5*term1*dY*dt + 0.5*dt*i*r
term_w = 0.5*term1*dY*dt - 0.5*dt*i*r

term_n = 0.5*dt*(xi**2*j/dY + (kappa/dY)*(theta-Y))
term_s = 0.5*dt*(xi**2*j/dY - (kappa/dY)*(theta-Y))
term_b = 0.25*xi*rho*term2*dt

# final matrix will be (J+1)(I+1) x (J+1)(I+1)
dims = (J+1)*(I+1)
M = np.zeros((J+1,I+1,J+1,I+1))

for i in range(I+1):
    data = np.array([term_w[i], term_c[i], term_e[i]])
    mat = spdiags(data, [1,0,-1], J+1, I+1).toarray().transpose()
    M[i][i] = mat

for i in range(I):
    data = np.array([-term_b[i], term_n[i], term_b[i]])
    mat = spdiags(data, [1,0,-1], J+1, I+1).toarray().transpose()
    M[i][i+1] = mat

for i in range(1,I+1):
    data = np.array([term_b[i], term_s[i], -term_b[i]])
    mat = spdiags(data, [1,0,-1], J+1, I+1).toarray().transpose()
    M[i+1][i] = mat

# data1 = np.array([term_w,term_c,term_e])
# A = spdiags(data1,[1,0,-1],5,5).toarray().transpose()
#
# data2 = np.array([-term_b,term_n,term_b])
# F = spdiags(data2,[1,0,-1],5,5).toarray().transpose()
#
# data3 = np.array([term_b,term_s,-term_b])
# E = spdiags(data3,[1,0,-1],5,5).toarray().transpose()
