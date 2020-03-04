import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def hipotesis(X, Theta):
    return np.dot(X,Theta)

def descenso_gradiente(X, Y, alpha):
    theta= np.array([[0.], [0.]])
    ths0 = np.array([0.])
    ths1 = np.array([0.])
    costes = np.array([coste(X,Y, theta)])
    while True:
        aux = hipotesis(X, theta)
        temp0 = theta[0][0] - alpha*(1/len(X))*np.sum(aux - Y)
        aux2 = aux - Y
        aux2 = aux2 * X[:,-1:]
        temp1 = theta[1][0] - alpha*(1/len(X))*np.sum(aux2)
        theta[0][0] = temp0
        theta[1][0] = temp1
        ths0 = np.append(ths0, temp0)
        ths1 = np.append(ths1, temp0)
        costes = np.append(costes, coste(X,Y, theta))
        if (costes[-2] - costes[-1]) < 0.001:
            break
        #print(theta)
        #print(coste(X,Y, theta))
    
    return (theta, np.array([ths0, ths1]), costes) 
         

datos = carga_csv("ex2data1.csv")

X = datos[:, :-1]
m = np.shape(X)[0]

Y = datos[:, -1:]
n = np.shape(X)[1]

plt.figure()
aux = np.where(Y == 1)
plt.scatter(X[aux, 0], X[aux, 1], c="black", label="admitted", marker="+")
aux = np.where(Y == 0)
plt.scatter(X[aux, 0], X[aux, 1], c="yellow", label="not admitted", marker="o")
plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.savefig("data.png")

print("FIN")


