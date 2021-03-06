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

reg = 0

Hs = np.array([])
ErrTrain = np.array([])
for i in range(0,m):
    Theta = np.array([1,1])
    res = opt.minimize(fun=costeYgrad, x0=Theta, args=(X[0:i+1], Y[0:i+1], reg),
                        method="TNC", jac = True, options={"maxiter":70})
    Hs = np.concatenate((Hs,res.x))
    aux = np.dot(np.hstack([np.ones([i+1, 1]), X[0:i+1]]), res.x[:, np.newaxis] )
    aux = np.sum(((aux - Y[0:i+1]) ** 2)/(2*(i+1)))
    ErrTrain = np.concatenate((ErrTrain,np.array([aux])))


Hs = np.reshape(Hs, (m,2))

Hval = np.dot(np.hstack([np.ones([Xval.shape[0], 1]), Xval]), Hs.transpose())
ErrVal = ((Hval - Yval) ** 2)/(2*Xval.shape[0])
ErrVal = np.sum(ErrVal, 0)

plt.figure()
plt.plot(np.arange(1, X.shape[0]+1), ErrTrain, c="blue", label="Train", linestyle='-')
plt.plot(np.arange(1, X.shape[0]+1), ErrVal, c="orange", label="Cross validation", linestyle='-')
plt.legend()
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.savefig("Curva.png")

print("Fin"*5)