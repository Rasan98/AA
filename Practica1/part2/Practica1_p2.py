import numpy as np
import matplotlib.pyplot as plt
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

def descenso_gradiente(X, Y, alpha, Theta, n):
    
    ths = np.array([[0.]])
    temps = np.array([[0.]])
    for i in range(n-1):
        ths = np.vstack((ths, [0.]))
        temps = np.vstack((temps, [0.]))
    costes = np.array([coste(X, Y, Theta)])
    
    for j in range(5):

        for i in range(n):
            temps[i] = Theta[i] - alpha*(1/len(X))*np.sum((hipotesis(X,Theta) - Y) * X[:,i:i+1])   
        
        Theta = temps
        np.hstack((ths, temps))
        costes = np.append(costes, coste(X,Y, Theta))
        
        #print(Theta)
        #print(costes[-1])
        #if (costes[-2] - costes[-1]) < 0.001:
        #    break
        
    return (Theta, ths, costes) 
         

datos = carga_csv("ex1data2.csv")

X = datos[:, :-1]
Y = datos[:, -1:]


m = np.shape(X)[0]
X = np.hstack([np.ones([m,1]), X])
n = np.shape(X)[1]


Theta = np.array([[0.]])
for i in range(n-1):
    Theta = np.vstack((Theta, [0.]))


alpha = 0.01

Theta, ths, costes = descenso_gradiente(X, Y, alpha, Theta, n) 

print("FIN")


