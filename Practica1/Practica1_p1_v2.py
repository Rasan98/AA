from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pandas.io.parsers import read_csv

def sum(a,b):
    return a+b

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
         

datos = carga_csv("ex1data1.csv")

X = datos[:, :-1]
m = np.shape(X)[0]

Y = datos[:, -1:]
n = np.shape(X)[1]

X = np.hstack([np.ones([m,1]), X])

alpha = 0.01

theta, ths, costes = descenso_gradiente(X, Y, alpha) 

H = hipotesis(X, theta)
X = datos[:, 0]
Y = datos[:, 1]

#plt.figure()
#plt.scatter(X, Y, c="red", label="Training cases", marker='x')
#plt.plot(X, H, c="blue", label="Regression line", linestyle='-')
#plt.legend()
#plt.xlabel("Population in 10.000s")
#plt.ylabel("Benefits in $10.000s")
#plt.savefig("line.png")
#rint("Finished line.png")

fig = plt.figure()
ax = fig.gca(projection='3d')

#Make data
#X = np.arange(-10, 10, 0.25)
#Y = np.arange(-1, 4, 0.25)

X, Z = np.meshgrid(ths[1], costes)
X, Y = np.meshgrid(ths[0], ths[1])



#Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth= 0, antialiased=False)

#Customize the z axis

ax.set_zlim(0, 700)
ax.zaxis.set_major_locator(LinearLocator(7))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

ax.set_xlabel('θ0')
ax.set_ylabel('θ1')

plt.savefig("3d.png")



