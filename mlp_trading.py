import numpy as np 
from TFMLP import MLPR
import matplotlib.pyplot as mpl 
from sklearn.preprocessing import scale

pth = 'ge.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 4))
A = scale(A)

y = A[:, 1].reshape(-1,1)
A = A[:, 0].reshape(-1,1)

i = 1
o = 1
h = 32

layers = [i, h, h, h, h, h, h, h, h, h, o]
mlpr = MLPR(layers, maxItr = 1000, tol = 0.40, reg = 0.001, verbose = True)

nDays = 5
n = len(A)
mlpr.fit(A[0:(n-nDays)], y[0:(n-nDays)])

yHat = mlpr.predict(A)
mpl.plot(A, y, c='#b0403f')
mpl.plot(A, yHat, c='#5aa9ab')
mpl.show()