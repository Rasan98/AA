import numpy as np
import scipy.optimize as opt 
import matplotlib.pyplot as plt
from scipy.io import loadmat

def sigmoide(X):
    return 1/(1+np.exp(-X)) 

def hipotesis(X, Theta):
    return sigmoide(np.dot(X, np.transpose(np.array([Theta]))))
  
def coste(Theta, X, Y, reg):
    m = np.shape(X)[0] 
    H = hipotesis(X, Theta) 
    aux = Y*np.log(H + 1e-6) + (1-Y)*np.log(1 - H + 1e-6) 
    aux = -aux.sum()/m
    aux2 = np.sum((Theta ** 2)) 
    aux2 = (reg/(2*m))*aux2
    return aux + aux2 

def gradienteRecurs(Theta, X, Y, reg):
    m = np.shape(X)[0] # numero de filas de x
    #calculando los gradientes vectorizados y regularizados cuando j=0 y j>=1
    grad = np.ravel((1/m)*np.dot(np.transpose(X), (hipotesis(X,Theta) - Y))) + (reg/m)*Theta  
    grad[0] = (1/m)*np.sum((hipotesis(X,Theta) - Y) * X[:,0:1]) 
    return grad  


def divide_data(X, Y, fact):
    sep =  np.where(Y == 1)[1][0]
    newX = X[0:int(sep//fact), :]
    newY = Y[0, 0:int(sep//fact)]
    newX = np.vstack([newX,X[sep:sep + int((X.shape[0]-sep)//fact)]])
    newY = np.hstack([newY, np.ones(int((X.shape[0]-sep)//fact))])
    return newX, np.array([newY])

def entrenamiento(Xp, Yp, reg):
    n = np.shape(Xp)[1]# columnas 65537
    thetas = np.empty((0,n), float) #(0,65537)
    Y1 = np.copy(Yp) #neumonia
    Y0 = np.copy(Yp) #normal
    Theta = np.zeros(n) #(65537,)
    print("Posiciones neumonia:",np.where(Yp == 1),"\nPosiciones normal:",np.where(Yp == 0))
    neumonia = np.where(Yp == 1) 
    normal = np.where(Yp == 0)
    Y1[neumonia[0]]=1
    Y1[normal[0]]=0
    print("Y1: ",Y1)
    Y0[normal[0]]=1
    Y0[neumonia[0]]=0
    print("Y0: ",Y0)
    X = Xp #
    result1 = opt.fmin_tnc(func=coste, x0=Theta, fprime=gradienteRecurs, args=(X, Y1, reg)) #se entrena el clasificador  
    result0 = opt.fmin_tnc(func=coste, x0=Theta, fprime=gradienteRecurs, args=(X, Y0, reg)) #se entrena el clasificador 
    thetas = np.vstack((thetas, result1[0])) #agrega a theta una fila con los resultados obtenidos
    thetas = np.vstack((thetas, result0[0]))
    print("Resultado: thetas=")
    print(thetas)    
    return thetas

def efectividad(thetas, X, Y):
    r=1/(1+np.exp(-np.argmax(np.dot(thetas, X))))
    print("Hipotesis",r)
    if r < 0.5:
        r=0
    else:
        if r>= 0.5:
            r=1
    print("Y",Y) 
    print("Hipotesis",r)
    return r == Y #regresa el valor de true o false dependiendo del resultado de la comparacion


data= loadmat("data.mat")
#print("Columnas: \n")
#print(data.keys())
X = data['xtrain']
Y = data['ytrain']
X,Y= divide_data(X,  Y, 5)
Y = Y.astype(int)
Y=np.transpose(Y)
m = np.shape(X)[0] # filas
X = np.hstack([np.ones([m,1]), X])
thetas = entrenamiento(X, Y, 0.1)

aux = [efectividad(thetas, X[i], Y[i][0]) for i in range(m)] 
print("Porcentaje de éxito: -->", np.sum(aux)/m) 


