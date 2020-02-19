import numpy as np
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def hipotesis(X, Theta):
    return np.dot(X,Theta)

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    aux = (H - Y) ** 2
    return aux.sum() / (2*len(X))

def descenso_gradiente(X, Y, alpha):
    theta= np.array([[0], [0]])
    for i in np.arange(6):
        aux = hipotesis(X, theta)
        temp0 = theta[0] - alpha*(1/len(X))*np.sum(aux - Y)
        aux2 = aux - Y
        aux2 = np.dot(aux2, X)
        temp1 = theta[1] - alpha*(1/len(X))*np.sum(aux2)
        theta[0] = temp0
        theta[1] = temp1
        print(theta)
        print(coste(X,Y, theta))
         

datos = carga_csv('ex1data1.csv')

X = datos[:, :-1]
m = np.shape(X)[0]

Y = datos[:, -1]
n = np.shape(X)[1]

X = np.hstack([np.ones([m,1]), X])

alpha = 0.01

#thetas, costes = 
descenso_gradiente(X, Y, alpha) 

print("FIN")