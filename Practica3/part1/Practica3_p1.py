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
    aux = Y*np.log(H + 1e-6) + (1-Y)*np.log(1 - H + 1e-6)  
    aux = -aux.sum()/m
    aux2 = np.sum((Theta ** 2))
    aux2 = (reg/(2*m))*aux2
    return aux + aux2 

def coste2(Theta,X,Y):
    H = sigmoide(np.matmul(X, Theta))
    return (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))

def gradienteRecurs(Theta, X, Y, reg):
    m = np.shape(X)[0]
    grad = np.ravel((1/m)*np.dot(np.transpose(X), (hipotesis(X,Theta) - Y))) #+ (reg/m)*Theta 
    grad[0] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,0:1])
    return grad  

def fun(thetas, X, etiq):
    return np.argmax(np.dot(thetas, X)) + 1 == etiq
    
def oneVsAll(Xp, Yp, num_etiquetas, reg):
    n = np.shape(Xp)[1]
    thetas = np.empty((0,n), float)
    ies = np.arange(1, num_etiquetas + 1)
    for i in ies:
        Y = np.copy(Yp)
        Theta = np.zeros(n)
        tr = np.where(Yp == i)
        fls = np.where(Yp != i)
        X = Xp
        Y[tr[0]] = 1
        Y[fls[0]] = 0
        print(Y)
        result = opt.fmin_tnc(func=coste, x0=Theta, fprime=gradienteRecurs, args=(X, Y, reg))
        thetas = np.vstack((thetas, result[0]))
    return thetas
    

data = loadmat("ex3data1.mat")
X = data['X']
Y = data['y']
Y = Y.astype(int) 
m = np.shape(X)[0]


sample = np.random.choice(X.shape[0],10)
plt.imshow(X[sample,:].reshape(-1,20).T)
plt.axis('off')
plt.savefig('prueba.png')

X = np.hstack([np.ones([m,1]), X])


thetas = oneVsAll(X, Y, 10, 0.1)

aux = [fun(thetas, X[i], Y[i][0]) for i in range(m)]

print("Sol -->", np.sum(aux)/m)

#i = 756

#calculo = np.dot(thetas, X[i])

#print("Sol -->", np.argmax(calculo) + 1, "realmente es ", Y[i])



print("FIN")