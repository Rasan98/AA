import numpy as np
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def hipotesis(th0, th1, x)
    return th0 + th1*x

def coste()



datos = carga_csv('ex1data1.csv')

X = datos[:, :-1]
m = np.shape(X)[0]

Y = datos[:, -1]
n = np.shape(X)[1]

X = np.hstack([np.ones([m,1]), X])

print(X)

alpha = 0.01

thetas, costes = descenso_gradiente(X, Y, alpha) 