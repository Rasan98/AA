import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat     
from sklearn import svm

def train_linear_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "linear", C=c)
    svmAux.fit(X,y.ravel())
    return svmAux


def train_rbf_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "rbf", C=c, gamma= 1/(2 * sigma**2))
    svmAux.fit(X,y.ravel())
    return svmAux

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

print("Loading")
data = loadmat("..\\data300_L.mat")
Xtrain = data['xtrain']
Ytrain = data['ytrain']
print("Loaded")

Xtrain, Ytrain = divide_data(Xtrain, Ytrain, 1.7) #Mín 1.7 with data256

Ytrain = Ytrain.ravel()

print("Training")
rbf = train_rbf_svm(Xtrain, Ytrain, 0.01, 0.01)
print("  rbf")
linear = train_linear_svm(Xtrain, Ytrain, 0.01, 0.01)
print("  linear")
print("Trained")

Xtest = data['xtest']
Ytest = data['ytest'].ravel()

sep =  np.where(Ytest == 1)[0][0]
X_norm = Xtest[0:sep, :]
Y_norm = Ytest[0:sep]
X_pneum = Xtest[sep:, :]
Y_pneum = Ytest[sep:]

print("Testing")
Hrbf = rbf.predict(Xtest)
Hrbf_norm = rbf.predict(X_norm)
Hrbf_pneum = rbf.predict(X_pneum)
print("  rbf")
Hlinear = linear.predict(Xtest)
Hlinear_norm = linear.predict(X_norm)
Hlinear_pneum = linear.predict(X_pneum)
print("  linear")
print("Tested")

print("----------------Porcentages------------")
print("   rbf:")
print("      Total-->", np.sum(Hrbf==Ytest)/len(Ytest))
print("      Normal-->", np.sum(Hrbf_norm==Y_norm)/len(Y_norm))
print("      Pneumonia-->", np.sum(Hrbf_pneum==Y_pneum)/len(Y_pneum))
print("   linear:")
print("      Total-->", np.sum(Hlinear==Ytest)/len(Ytest))
print("      Normal-->", np.sum(Hlinear_norm==Y_norm)/len(Y_norm))
print("      Pneumonia-->", np.sum(Hlinear_pneum==Y_pneum)/len(Y_pneum))
