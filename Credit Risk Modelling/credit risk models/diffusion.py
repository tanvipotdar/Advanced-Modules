S0 =100
r =0.05
sigma =0.2
lamb =0.75
mu = -0.6
delta =0.25
T =1.0
M =120
I =1
dt = T / M

rj = lamb *( np.exp ( mu +0.5* delta **2) -1)
S_j = np.zeros (( M +1 , I ))
S_j[0]= S0
S = np.zeros (( M +1 , I ))
S[0]= S0


z1 = npr.standard_normal (( M +1 , I ))
z2 = npr.standard_normal (( M +1 , I ))
y = npr.poisson ( lamb * dt , ( M +1 , I ))

for t in range (1 , M +1):
	S[t]= S[t-1]* np.exp (( r -0.5* sigma **2)* dt + sigma * np . sqrt (dt)* npr.randn(I))

	S_j[t]= S_j[t-1]*( np.exp (( r - rj -0.5* sigma **2)* dt + sigma * np.sqrt ( dt )* z1 [t])
	+( np.exp ( mu + delta * z2 [t]) -1)* y [ t ])
	S_j[t]= np.maximum (S_j[t] ,0)

paths = pd.DataFrame()
paths['jump'] = [x[0] for x in S_j]
paths['pure'] = [x[0] for x in S]
ax = paths.plot(title='Jump vs pure diffusion trajectories')
ax.set(xlabel='time', ylabel='asset price')
plt.show()