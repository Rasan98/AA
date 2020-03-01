import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def calcula_normalizado(valor, med, desv):
    return (valor - med)/desv

def normaliza(X):
    medias = np.array([1.])
    desvs = np.array([1.])
    for i in range(1, np.shape(X)[1]):
        medias = np.append(medias, np.mean(X[:,i]))
        desvs = np.append(desvs, np.std(X[:,i]))
        X[:,i] = calcula_normalizado(X[:,i], medias[i], desvs[i])
    return (medias, desvs)

def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    aux = (H - Y) ** 2
    return aux.sum() / (2*len(X)) 
         
def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta

datos = carga_csv("ex1data2.csv")

X = datos[:, :-1]
Y = datos[:, -1:]


m = np.shape(X)[0]
X = np.hstack([np.ones([m,1]), X])
n = np.shape(X)[1]
medias, desvs = normaliza(X)

Theta = np.array([[0.]])

for i in range(n-1):
    Theta = np.vstack((Theta, [0.]))

costes = np.array(coste(X, Y, Theta))

alpha = 0.01

j = 0

while True:
    j = j + 1
    Theta = gradiente(X,Y,Theta, alpha)
    costes = np.append(costes, coste(X, Y, Theta))
    if (costes[-2] - costes[-1]) < 0.001:
        break


print(costes[-1])