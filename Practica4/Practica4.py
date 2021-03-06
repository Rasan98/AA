import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat     
import checkNNGradients as chk
import scipy.optimize as opt

def sigmoide(X):
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
    return aux + aux2

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[: (num_ocultas * (num_entradas + 1))], (num_ocultas, (num_entradas+1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    m = X.shape[0]       
    a1 = np.hstack([np.ones([m, 1]), X])
    
    a2, h = forwprop(theta1, theta2, a1)       
    cost = coste(theta1, theta2, m, y, reg, h)       
    
    inc1 = np.zeros((theta1.shape[0], theta1.shape[1]))
    inc2 = np.zeros((theta2.shape[0], theta2.shape[1]))   
    for i in range(m):
        delta3 = h[i] - y[i] 
        delta3 = delta3[:,np.newaxis]
        delta2 = np.dot(theta2.transpose(), delta3) * ((a2[i] * (1-a2[i]))[:,np.newaxis])
        
        aux = a1[i]
        inc1 = inc1 + np.dot(delta2[1:], aux[np.newaxis,:])
        aux = a2[i]
        inc2 = inc2 + np.dot(delta3, aux[np.newaxis,:])
    D1 = inc1/m
    D1[:,1:] = D1[:,1:] + (reg/m)*theta1[:,1:]
    D2 = inc2/m
    D2[:,1:] = D2[:,1:] + (reg/m)*theta2[:,1:]
    #print(cost)
    return cost, np.concatenate((np.ravel(D1), np.ravel(D2)))

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

def codificaY(Y, num_etiquetas):
    Yp = np.zeros((Y.shape[0], num_etiquetas + 1))
    Yp[[np.arange(Y.shape[0])], Y[:,0]] = 1
    Yp = np.delete(Yp, 0, 1) 
    return Yp

def fun(h, etiq):
    return np.argmax(h) == etiq - 1

def calculate_precision(theta1, theta2, X, Y):
    a1 = np.hstack([np.ones([len(X), 1]), X])
    _ , h = forwprop(theta1, theta2, a1)
    aux = [fun(h[i], Y[i][0]) for i in range(len(X))]
    return np.sum(aux)/len(X)



data = loadmat("ex4data1.mat")
X = data['X']
Y = data['y']
Y = Y.astype(int)

num_etiquetas = 10
y = codificaY(Y, num_etiquetas)
y = y.astype(int)

num_entradas = 400
num_ocultas = 25
params_1 = pesosAleatorios(num_entradas, num_ocultas)
params_2 = pesosAleatorios(num_ocultas, num_etiquetas)
params_rn = np.concatenate((np.ravel(params_1), np.ravel(params_2)))
reg = 1

print(chk.checkNNGradients(backprop, 1))
#print(backprop(params_rn, num_entradas, num_ocultas, num_etiquetas,X, Y, reg))

res = opt.minimize(fun=backprop_rec, x0=params_rn, args=(num_entradas, num_ocultas, num_etiquetas, X, y, reg),
                    method="TNC", jac = True, options={"maxiter":70})

thetas = res.x
theta1 = np.reshape(thetas[:(num_ocultas * (num_entradas + 1))], (num_ocultas, (num_entradas+1)))
theta2 = np.reshape(thetas[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

#weights = loadmat("ex4weights.mat")
#theta1, theta2 = weights["Theta1"], weights["Theta2"]


print(calculate_precision(theta1, theta2, X, Y))


print("Fin" * 5)