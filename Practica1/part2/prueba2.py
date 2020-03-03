from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv

def make_data(t0_range, t1_range, X, Y):
    """Genera las matrices X,Y,Z para generar un plot en 3D
    """
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    # Theta0 y Theta1 tienen las misma dimensiones, de forma que
    # cogiendo un elemento de cada uno se generan las coordenadas x,y
    # de todos los puntos de la rejilla
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [[Theta0[ix, iy]], [Theta1[ix, iy]]])
    return [Theta0, Theta1, Coste]

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

datos = carga_csv("ex1data1.csv")

X = datos[:, :-1]
m = np.shape(X)[0]

Y = datos[:, -1:]
n = np.shape(X)[1]

X = np.hstack([np.ones([m,1]), X])

alphas = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])

colors = np.array(["blue","green","red","cyan","magenta","yellow","black"])
labels = np.array(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1"])

plt.figure()

Theta = np.array([[0.]])

for i in range(n-1):
    Theta = np.vstack((Theta, [0.]))

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

print("FIN")


