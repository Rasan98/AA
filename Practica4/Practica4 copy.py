import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat     
#import checkNNGradients


def sigmoide(X):
    return 1/(1+np.exp(-X))

def sigmoide_derivada(X): #NEW
    return sigmoide(X)*(1-sigmoide(X))

def pesosAleatorios(L_in, L_out): #NEW
    eini = np.sqrt(6)/np.sqrt(L_in + L_out)
    aux = np.random.uniform(-eini,eini,(L_in+1)*L_out)
    return np.reshape(aux, (L_out,L_in + 1))




def forwprop(theta1, theta2, X):
    a1 = X
    z2 = np.dot(theta1, np.transpose(a1))
    a2 = sigmoide(z2)
    a2 = np.vstack((np.ones(np.shape(a2)[1]), a2))
    z3 = np.dot(theta2, a2)
    a3 = sigmoide(z3)
    return a3.transpose()    

def coste(theta1, theta2, X, y, lda):
    H = hipotesis(theta1, theta2, X)
    aux = (-y*np.log((H + 1e-6))) - ((1-y)*np.log((1-H + 1e-6)))
    aux = (1 / (len(X))) * np.sum(aux)
    #aux2 = np.sum(theta1 ** 2) + np.sum(theta2 ** 2)
    #aux2 = (aux2*lda)/(2*len(X))
    return aux #+ aux2
    
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[: (num_ocultas * (num_entradas + 1))], (num_ocultas, (num_entradas+1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    m = X.shape[0]
    X = np.hstack([np.ones([m, 1]), X])
    return coste(theta1, theta2, X, y, reg)

def codificaY(Y, num_etiquetas):
    Yp = np.zeros((Y.shape[0], num_etiquetas + 1))
    Yp[[np.arange(Y.shape[0])], Y[:,0]] = 1
    Yp[:,0] = Yp[:,num_etiquetas]
    Yp = np.delete(Yp, 10, 1) 
    return Yp

data = loadmat("ex4data1.mat")
X = data['X']
Y = data['y']
Y = Y.astype(int)

num_etiquetas = 10
Y = codificaY(Y, num_etiquetas)
Y = Y.astype(int)

num_entradas = 400
num_ocultas = 25
params_num = num_ocultas*(num_entradas+1) + num_etiquetas*(num_ocultas+1) 
params_rn = np.random.uniform(-0.5,0.5,params_num)
reg = 1

weights = loadmat("ex4weights.mat")
theta1, theta2 = weights["Theta1"], weights["Theta2"]

print(backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg))

print("Fin" * 5)