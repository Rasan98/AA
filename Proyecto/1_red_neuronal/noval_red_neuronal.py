#0 normal, 1 neumonía

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat     
import scipy.optimize as opt

def sigmoide(X):
    #print(np.ravel(X)[np.argmax(X)])
    return 1/(1+np.exp(-X))

def pesosAleatorios(L_in, L_out):
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
    return a2.transpose(), a3.transpose()

def coste(theta1, theta2, m, y, lda, H):
    aux = (-y*np.log((H + 1e-10))) - ((1-y)*np.log((1-H + 1e-10)))
    aux = (1 / m) * np.sum(aux)
    aux2 = np.sum(theta1[:,1:] ** 2) + np.sum(theta2[:,1:] ** 2)
    aux2 = (aux2*lda)/(2*m)
    c = aux + aux2
    print(c)
    return c

def backprop_rec(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[: (num_ocultas * (num_entradas + 1))], (num_ocultas, (num_entradas+1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    
    m = X.shape[0]       
    a1 = np.hstack([np.ones([m, 1]), X])
    
    a2, h = forwprop(theta1, theta2, a1)       
    cost = coste(theta1, theta2, m, y, reg, h)       
    
    delta3 = h - y 
    delta2 = np.dot(theta2.transpose(), delta3.transpose()).transpose() * (a2 * (1-a2))
    delta2 = delta2[:,1:]
    inc1 = np.dot(delta2.transpose(), a1)
    inc2 = np.dot(delta3.transpose(), a2)
    D1 = inc1/m
    D1[:,1:] = D1[:,1:] + (reg/m)*theta1[:,1:]
    D2 = inc2/m
    D2[:,1:] = D2[:,1:] + (reg/m)*theta2[:,1:]
    #print(cost)
    return cost, np.concatenate((np.ravel(D1), np.ravel(D2)))

def fun(h, etiq):
    return np.argmax(h) == etiq

def calculate_precision(theta1, theta2, X, Y):
    a1 = np.hstack([np.ones([len(X), 1]), X])
    _ , h = forwprop(theta1, theta2, a1)
    aux = [fun(h[i], Y[i][0]) for i in range(len(X))]
    return np.sum(aux)/len(X)

def codificaY(Y, num_etiquetas):
    Yp = np.zeros((Y.shape[0], num_etiquetas))
    Yp[[np.arange(Y.shape[0])], Y[:,0]] = 1
    return Yp

def calc_norm(X):
    medias = np.mean(X,0)
    desv = np.std(X,0)
    Xnorm = (X-medias)/desv
    return Xnorm, medias, desv

def aplica_norm(X, medias, desv):
    Xnorm = (X-medias)/desv
    return Xnorm

def divide_data(X, Y, fact):
    sep =  np.where(Y == 1)[1][0]
    newX = X[0:int(sep//fact), :]
    newY = Y[0, 0:int(sep//fact)]
    newX = np.vstack([newX,X[sep:sep + int((X.shape[0]-sep)//fact)]])
    newY = np.hstack([newY, np.ones(int((X.shape[0]-sep)//fact))])
    return newX, np.array([newY])


print("Loading data")
data = loadmat("..\\data300.mat")
print("Data loaded")
Xtrain = data['xtrain']
Ytrain = data['ytrain']

Xtrain, Ytrain = divide_data(Xtrain, Ytrain, 2) #Mín 1.7 with data256

Ytrain = Ytrain.transpose()
Ytrain = Ytrain.astype(int)

print("Normalizing xtrain")
Xtrain, medias, desv = calc_norm(Xtrain)
print("xtrain normalized")


num_etiquetas = 2
Ytrain = codificaY(Ytrain,num_etiquetas)


num_entradas = Xtrain.shape[1]
num_ocultas = 100
params_1 = pesosAleatorios(num_entradas, num_ocultas)
params_2 = pesosAleatorios(num_ocultas, num_etiquetas)
params_rn = np.concatenate((np.ravel(params_1), np.ravel(params_2)))
reg = 1

print("Training init")
res = opt.minimize(fun=backprop_rec, x0=params_rn, args=(num_entradas, num_ocultas, num_etiquetas, Xtrain, Ytrain, reg),
                    method="TNC", jac = True, options={"maxiter":80})
print("Training end")

thetas = res.x
theta1 = np.reshape(thetas[:(num_ocultas * (num_entradas + 1))], (num_ocultas, (num_entradas+1)))
theta2 = np.reshape(thetas[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

dicti = {}
dicti["theta1"] = theta1
dicti["theta2"] = theta2

Xtest = data["xtest"]
Ytest = data["ytest"]

print("Normalizing xtest")
Xtest = aplica_norm(Xtest, medias, desv)
print("xtest normalized")

sep =  np.where(Ytest == 1)[1][0]
X_norm = Xtest[0:sep, :]
Y_norm = np.array([Ytest[0, 0:sep]])
X_pneum = Xtest[sep:, :]
Y_pneum = np.array([Ytest[0, sep:]])


Y_norm = Y_norm.transpose()
Y_norm = Y_norm.astype(int)
Y_norm = codificaY(Y_norm, num_etiquetas)

Y_pneum = Y_pneum.transpose()
Y_pneum = Y_pneum.astype(int)
Y_pneum = codificaY(Y_pneum, num_etiquetas)

Ytest = Ytest.transpose()
Ytest = Ytest.astype(int)
Ytest = codificaY(Ytest, num_etiquetas)


print("Starting test")
print(" Normal precision-->" + str(calculate_precision(theta1, theta2, X_norm, Y_norm)))
print(" Pneumonia precision-->" + str(calculate_precision(theta1, theta2, X_pneum, Y_pneum)))
print(" Full precision-->" + str(calculate_precision(theta1, theta2, Xtest, Ytest)))
print("Test finished")
#savemat("weights.mat", dicti)

print("Fin" * 5)