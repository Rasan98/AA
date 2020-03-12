import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.preprocessing import PolynomialFeatures
def carga_csv(file_name):
    valores = read_csv(file_name, header = None).values
    return valores.astype(float)

def sigmoide(X):
    return 1/(1+np.exp(-X))

def hipotesis(X, Theta):
    return sigmoide(np.dot(X, np.transpose(np.array([Theta]))))

def coste(Theta, X, Y, lda):
    m = np.shape(X)[0]
    H = hipotesis(X, Theta)
    aux = Y*np.log(H) + (1-Y)*np.log(1 - H)  
    aux = -aux.sum()/m
    aux2 = np.sum((Theta ** 2))
    aux2 = (lda/(2*m))*aux2
    return aux + aux2

def gradiente(Theta, X, Y, lda):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    grad = np.zeros(n)
    grad[0] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,0:1])
    for i in range(1, n):
        grad[i] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,i:i+1]) + (lda/m)*Theta[i]
    return grad  

def fun(Hi, Y):
    return (Hi < 0.5 and Y == 0) or (Hi >= 0.5 and Y == 1)

def calcula_porcentaje(X, Y, Theta):
    H = np.ravel(np.transpose(hipotesis(X, Theta)))
    aux = [fun(H[i], Y[i, 0]) for i in range(len(H))]
    return np.sum(aux)/len(H)
    
def pinta_frontera_recta(X, Y, theta, poly, lda):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    aux = np.where(Y == 1)
    plt.scatter(X[aux, 0], X[aux, 1], c="black", label="y = 1", marker="+")
    aux = np.where(Y == 0)
    plt.scatter(X[aux, 0], X[aux, 1], c="yellow", label="y = 0", marker="o")
    plt.legend()
    plt.title("lambda ="+ str(lda))
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.savefig("lambda="+ str(lda) + ".png")
    plt.close()
        
def print_data(X,Y):
    plt.figure()
    aux = np.where(Y == 1)
    plt.scatter(X[aux, 0], X[aux, 1], c="black", label="y = 1", marker="+")
    aux = np.where(Y == 0)
    plt.scatter(X[aux, 0], X[aux, 1], c="yellow", label="y = 0", marker="o")
    plt.legend()
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.savefig("data.png")


datos = carga_csv("ex2data2.csv")
poly = PolynomialFeatures(6) 
X = datos[:, :-1]
m = np.shape(X)[0]
Y = datos[:, -1:]

print_data(X,Y)

X = poly.fit_transform(X)
n = np.shape(X)[1]

porc_exito = np.array([])
pruebas = [0, 1, 10, 25, 50, 75, 100, 125, 150]
for lda in pruebas:
    Theta = np.zeros(n)
    result = opt.fmin_tnc(func=coste, x0=Theta, fprime=gradiente, args=(X, Y, lda))
    theta_opt = result[0]
    pinta_frontera_recta(X[:, 1:], Y, theta_opt, poly, lda)
    porc_exito = np.append(porc_exito, calcula_porcentaje(X,Y, theta_opt))

plt.figure()
plt.plot(pruebas, porc_exito, c="blue", label="Eficacia", linestyle='-')
plt.legend()
plt.xlabel("Lambda")
plt.ylabel("Porcentaje de éxito")
plt.savefig("eficacia.png")
plt.close()

print("FIN")