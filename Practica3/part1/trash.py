import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.io import loadmat 

def sigmoide(X):
    return 1/(1+np.exp(-X))

def hipotesis(X, Theta):
    return sigmoide(np.dot(X, np.transpose(np.array([Theta]))))

def coste(Theta, X, Y, reg):
    m = np.shape(X)[0]
    H = hipotesis(X, Theta)
    aux = Y*np.log(H) + (1-Y)*np.log(1 - H)  
    aux = -aux.sum()/m
    aux2 = np.sum((Theta ** 2))
    aux2 = (reg/(2*m))*aux2
    return aux + aux2

def gradienteIter(Theta, X, Y, lda):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    grad = np.zeros(n)
    grad[0] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,0:1])
    for i in range(1, n):
        grad[i] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,i:i+1]) + (lda/m)*Theta[i]
    return grad  

def gradienteRecurs(Theta, X, Y, reg):
    m = np.shape(X)[0]
    grad = np.ravel((1/m)*np.dot(np.transpose(X), (hipotesis(X,Theta) - Y))) + (reg/m)*Theta 
    grad[0] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,0:1])
    return grad  

def fun(Hi, Y):
    return (Hi < 0.5 and Y == 0) or (Hi >= 0.5 and Y == 1)

def calcula_porcentaje(X, Y, Theta):
    H = np.ravel(np.transpose(hipotesis(X, Theta)))
    aux = [fun(H[i], Y[i, 0]) for i in range(len(H))]
    return np.sum(aux)/len(H)

def oneVsAll(Xp, Yp, num_etiquetas, reg):
    n = np.shape(Xp)[1]
    thetas = np.empty((0,n), float)
    ies = np.arange(1, num_etiquetas)
    ies = np.insert(ies, 0, num_etiquetas)
    for i in ies:
        Theta = np.zeros(n)
        z = np.where(Yp == i)
        X = Xp[z[0]]
        Y = Yp[z[0]]
        result = opt.fmin_tnc(func=coste, x0=Theta, fprime=gradienteRecurs, args=(X, Y, reg))
        thetas = np.vstack((thetas, result[0]))
    return thetas
    

data = loadmat("ex3data1.mat")
X = data['X']
Y = data['y']
Y = Y.astype(int)
m = np.shape(X)[0]
n = np.shape(X)[1]

sample = np.random.choice(X.shape[0],10)
plt.imshow(X[sample,:].reshape(-1,20).T)
plt.axis('off')
plt.savefig('prueba.png')

X = np.hstack([np.ones([m,1]), X])

thetas = oneVsAll(X, Y, 10, 0.1)

print("FIN")