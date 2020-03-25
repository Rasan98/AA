import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat     

def sigmoide(X):
    return 1/(1+np.exp(-X))

def fun(a3, etiq):
    return np.argmax(a3) + 1 == etiq

data = loadmat("ex3data1.mat")
X = data['X']
Y = data['y']
Y = Y.astype(int)
m = np.shape(X)[0]
X = np.hstack([np.ones([m,1]), X])


weights = loadmat("ex3weights.mat")
theta1, theta2 = weights["Theta1"], weights["Theta2"]

a1 = X

z2 = np.dot(theta1, np.transpose(a1))
a2 = sigmoide(z2)
a2 = np.vstack((np.ones(np.shape(a2)[1]), a2))

z3 = np.dot(theta2, a2)
a3 = sigmoide(z3)
a3 = a3.transpose()


aux = [fun(a3[i], Y[i][0]) for i in range(m)]

print("Sol -->", np.sum(aux)/m)

print("Fin")