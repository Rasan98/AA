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
    coste = (np.sum((H-Y.ravel()) ** 2))/(2*m) + (np.sum(Theta[1:] ** 2) * reg)/(2*m)
    grad = np.sum((H-Y.ravel())[:,np.newaxis]*x,0)/m
    grad[1:] += (reg/m)*(Theta[1:] ** 2)
    return coste, grad

def transformaX(X,p):
    for i in range(2,p+1):
       X = np.hstack((X, (X[:,:1] ** i)))
    return X

def calc_norm(X):
    medias = np.mean(X,0)
    desv = np.std(X,0)
    Xnorm = (X-medias)/desv
    return Xnorm, medias, desv

def aplica_norm(X, medias, desv):
    Xnorm = (X-medias)/desv
    return Xnorm

def calcula_error(Theta, X, Y):
    H = hipotesis(X, Theta)
    return np.sum(((H[:, np.newaxis] - Y) ** 2))/(2*X.shape[0])
    
def calcula_errores_lambda(X, Y, Xval, Yval, p, lambd):
    Xnorm = transformaX(X,p)
    Xnorm, medias, desv = calc_norm(Xnorm)
    Xnormmas1 = np.hstack((np.ones([Xnorm.shape[0], 1]), Xnorm))
    
    Xnorm_val = transformaX(Xval,p)
    Xnorm_val = aplica_norm(Xnorm_val, medias, desv)
    Xnorm_val = np.hstack((np.ones([Xnorm_val.shape[0], 1]), Xnorm_val))
    
    theta = np.ones(p+1)
    errTrain = np.array([])
    errVal = np.array([])
    thetas = np.empty((0, p+1))
    for i in lambd:
        aux =  opt.minimize(fun=costeYgrad, x0=theta, args=(Xnorm, Y, i),
                                method="TNC", jac = True, options={"maxiter":70})
        thetas = np.vstack([thetas, aux.x])
        errTrain = np.concatenate((errTrain, [calcula_error(aux.x, Xnormmas1, Y)]))
        errVal = np.concatenate((errVal, [calcula_error(aux.x, Xnorm_val, Yval)]))
    return errTrain, errVal, lambd[np.argmin(errVal)], thetas[np.argmin(errVal)], medias, desv

data = loadmat("ex5data1.mat")
X = data['X']
Y = data['y']
m = X.shape[0]
Xval = data['Xval']
Yval = data['yval']

Xtest = data['Xtest']
Ytest = data['ytest']


lambd = np.array([0, 0.001, 0.003, 0.01, 0.03, 1, 3, 10])
errTrain, errVal, lambd_opt, theta_opt, medias, desv = calcula_errores_lambda(X, Y, Xval, Yval, 8, lambd)


plt.figure()
plt.plot(lambd, errTrain, c="blue", label="Train", linestyle='-')
plt.plot(lambd, errVal, c="orange", label="Cross Validation", linestyle='-')
plt.legend()
plt.xlabel("lambda")
plt.ylabel("error")
plt.savefig("lambda_opt.png")

Xtest_norm = transformaX(Xtest, 8)
Xtest_norm = aplica_norm(Xtest_norm, medias, desv)
Xtest_norm = np.hstack((np.ones([Xtest_norm.shape[0], 1]), Xtest_norm))

print("El error para lambda =", lambd_opt, "es -->", calcula_error(theta_opt, Xtest_norm, Ytest))

print("Fin"*5)