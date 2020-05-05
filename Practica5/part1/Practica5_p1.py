import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat     
import scipy.optimize as opt

def hipotesis(X, Theta):
    return np.dot(X,Theta)
    

def costeYgrad(Theta, X, Y, reg):
    m = X.shape[0]
    x = np.hstack([np.ones([m, 1]), X])
    H = hipotesis(x, Theta)
    coste = (np.sum((H-Y.ravel()) ** 2))/(2*m) + ((Theta[1] ** 2) * reg)/(2*m)
    grad = np.sum((H-Y.ravel())[:,np.newaxis]*x,0)/m
    grad[1:] += (reg/m)*(Theta[1:] ** 2)
    return coste, grad

data = loadmat("ex5data1.mat")
X = data['X']
Y = data['y']
m = X.shape[0]
Xval = data['Xval']
Yval = data['yval']

Xtest = data['Xtest']
Ytest = data['ytest']

Theta = np.array([1,1])

reg = 0

print(costeYgrad(Theta, X, Y, 1))

res = opt.minimize(fun=costeYgrad, x0=Theta, args=(X, Y, reg),
                    method="TNC", jac = True, options={"maxiter":70})

plt.figure()
plt.scatter(X, Y, c="red", marker='x')
plt.plot(X, hipotesis(np.hstack([np.ones([m, 1]), X]),res.x), c="blue", linestyle='-')
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of the dam (y)")
plt.savefig("line.png")

print("Fin"*5)