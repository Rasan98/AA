import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def sigmoide(X):
    return 1/(1+np.exp(-X))

def hipotesis(X, Theta):
    return sigmoide(np.dot(X, np.transpose(np.array([Theta]))))

def coste(Theta, X, Y):
    H = hipotesis(X, Theta)
    aux = Y*np.log(H) + (1-Y)*np.log(1 - H)  
    return -aux.sum()/np.shape(X)[0]

def gradiente(Theta, X, Y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,i:i+1])
    return grad  


def fun(Hi, Y):
    return (Hi < 0.5 and Y == 0) or (Hi >= 0.5 and Y == 1)

def calcula_porcentaje(X, Y, Theta):
    H = np.ravel(np.transpose(hipotesis(X, Theta)))
    aux = [fun(H[i], Y[i, 0]) for i in range(len(H))]
    return np.sum(aux)/len(H)
    
def calcula_porcentajeVector(X, Y, Theta):
    H = np.ravel(np.transpose(hipotesis(X, Theta)))
    aux = [fun(H[i], Y[i, 0]) for i in range(len(H))]
    return np.sum(aux)/len(H)

def pinta_frontera_recta(X, Y, theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    



datos = carga_csv("ex2data1.csv")

X = datos[:, :-1]
m = np.shape(X)[0]
X = np.hstack([np.ones([m,1]), X])
n = np.shape(X)[1]
Y = datos[:, -1:]

Theta = np.zeros(n)

result = opt.fmin_tnc(func=coste, x0=Theta, fprime=gradiente, args=(X, Y))
theta_opt = result[0]

pinta_frontera_recta(X[:, 1:], Y, theta_opt)

aux = np.where(Y == 1)
plt.scatter(X[aux, 1], X[aux, 2], c="black", label="admitted", marker="+")
aux = np.where(Y == 0)
plt.scatter(X[aux, 1], X[aux, 2], c="yellow", label="not admitted", marker="o")
#plt.plot(np.random.uniform(30, 100, 100), hipotesis(X, theta_opt))
plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.savefig("frontera.png")
plt.close()

print("Porcentaje de éxito --> ", calcula_porcentaje(X,Y, theta_opt))

print("FIN")