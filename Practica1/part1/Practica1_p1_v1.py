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

def descenso_gradiente(X, Y, alpha):
    theta= np.array([[0.], [0.]])
    ths0 = np.array([0.])
    ths1 = np.array([0.])
    costes = np.array([coste(X,Y, theta)])
    for i in np.arange(1500):
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

#thetas, costes = 
theta, ths, costes = descenso_gradiente(X, Y, alpha) 

H = hipotesis(X, theta)
X = datos[:, 0]
Y = datos[:, 1]

plt.figure()
plt.scatter(X, Y, c="red", label="Training cases", marker='x')
plt.plot(X, H, c="blue", label="Regression line", linestyle='-')
plt.legend()
plt.xlabel("Population in 10.000s")
plt.ylabel("Benefits in $10.000s")
plt.savefig("line.png")

fig = plt.figure()
ax = fig.gca(projection='3d')

#Make data

X = datos[:, :-1]
Y = datos[:, -1:]
X = np.hstack([np.ones([m,1]), X])
arrays = make_data([-10, 10], [-1, 4], X, Y)


#Plot the surface
surf = ax.plot_surface(arrays[0], arrays[1], arrays[2], cmap=cm.coolwarm, linewidth=0, antialiased=False)

#Customize the z and x axis
ax.set_zlim(0, 700)
ax.zaxis.set_major_locator(LinearLocator(8))
ax.xaxis.set_major_locator(LinearLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax.zaxis.set_major_formatter(FormatStrFormatter("%d"))
ax.view_init(elev=15, azim=230)
ax.set_xlabel('θ0')
ax.set_ylabel('θ1')

plt.savefig("3d.png")

fig = plt.figure()
plt.contour(arrays[0], arrays[1], arrays[2], np.logspace(-2, 3, 20), colors='blue')
plt.xlabel("θ0")
plt.ylabel("θ1")
plt.scatter(theta[0], theta[1], c="red", marker='x')
plt.savefig("contour.png")

print("FIN")