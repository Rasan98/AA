import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def calcula_normalizado(valor, med, desv):
    return (valor - med)/desv

def normaliza(X):
    medias = np.array([])
    desvs = np.array([])
    for i in range(1, np.shape(X)[1]):
        medias = np.append(medias, np.mean(X[:,i]))
        desvs = np.append(desvs, np.std(X[:,i]))
        X[:,i] = calcula_normalizado(X[:,i], medias[i-1], desvs[i-1])
    return (medias, desvs)

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
    j = 0
    while True:
        j = j + 1
        for i in range(n):
            temps[i] = Theta[i] - alpha*(1/len(X))*np.sum((hipotesis(X,Theta) - Y) * X[:,i:i+1])   
        
        Theta = temps
        np.hstack((ths, temps))
        costes = np.append(costes, coste(X,Y, Theta))
        
        #print(Theta)
        #print(costes[-1])
        if (costes[-2] - costes[-1]) < 0.001:
            break
    #print(j)
    return (Theta, costes, j) 
         

datos = carga_csv("ex1data2.csv")

X = datos[:, :-1]
Y = datos[:, -1:]

m = np.shape(X)[0]
X = np.hstack([np.ones([m,1]), X])
n = np.shape(X)[1]

Xp = np.copy(X)

medias, desvs = normaliza(X)

Theta = np.array([[0.]])

for i in range(n-1):
    Theta = np.vstack((Theta, [0.]))


#alpha = 0.01

#Theta, ths, costes = descenso_gradiente(X, Y, alpha, Theta, n) 

j = 0
alphas = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])

colors = np.array(["blue","green","red","cyan","magenta","yellow","black"])
labels = np.array(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1"])
plt.figure()

for i in range(np.shape(alphas)[0]):
    #j = j+1
    a, costes, iter = descenso_gradiente(X, Y, alphas[i], Theta, n)
    print(iter)
    #print(j, "  ", costes[-1], "  ", i) 
    plt.plot(np.arange(iter + 1), costes, c=colors[i], label=labels[i], linestyle='-')

plt.legend()

plt.xlabel("Num iter")
plt.ylabel("Costes")

plt.savefig("alphas.png")

#Xq = np.array([2500, 3])
Xq = np.array([1650,3])

Xq_norm = calcula_normalizado(np.transpose(Xq), medias, desvs)
Xq_norm = np.hstack([np.ones([1]), Xq_norm])


Xq = np.hstack([np.ones([1]), Xq])

Theta, costes, iter = descenso_gradiente(X, Y, 1.2, Theta, n)

Yp = hipotesis(Xq_norm, Theta)

X = Xp
aux = np.dot(np.transpose(X), X)
aux = np.linalg.pinv(aux)
aux = np.dot(aux, X.transpose())
eqTheta = np.dot(aux, Y)

Y = hipotesis(Xq, eqTheta)

print(Yp, Y)

print("FIN")


