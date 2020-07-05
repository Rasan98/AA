import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat     
from sklearn import svm

def train_linear_svm(X, y, c):
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
data = loadmat("..\\60_20_20_data300.mat")
Xtrain = data['xtrain']
Ytrain = data['ytrain']
print("Loaded")

Xtrain, Ytrain = divide_data(Xtrain, Ytrain, 3) #Mín 1.7 with data256

Ytrain = Ytrain.ravel()

values = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

print("Training")
Xval = data["xval"]
Yval = data["yval"].ravel()


#-----------------------------------------
pairs = np.empty((64,2))
errors = np.empty(64)
my_dict = {}
for i in range(8):
    for j in range(8):
        print("   Train(rbf) -->", 8*i + j)
        new = np.array([values[i],values[j]])
        pairs[8*i + j] = np.array([values[i],values[j]])
        aux = train_rbf_svm(Xtrain, Ytrain, new[0], new[1])
        print("   Predict(rbf) -->", 8*i + j)
        H = aux.predict(Xval)
        error = np.sum((H - Yval)**2)*(1/(2*len(Yval)))
        errors[8*i + j] = error
#-----------------------------------------
#cs = np.empty(8)
#errors = np.empty(8)
#for i in range(8):
#    print("   Train(linear) -->", i)
#    new = values[i]
#    cs[i] = values[i]
#    aux = train_linear_svm(Xtrain, Ytrain, new)
#    H = aux.predict(Xval)
#    error = np.sum((H - Yval)**2)*(1/(2*len(Yval)))
#    errors[i] = error
#-----------------------------------------


print("Trained")


#-----------------------------------------
opt = pairs[np.argmin(errors)]
#-----------------------------------------
#opt = cs[np.argmin(errors)]
#-----------------------------------------


Xtest = data['xtest']
Ytest = data['ytest'].ravel()

sep =  np.where(Ytest == 1)[0][0]
X_norm = Xtest[0:sep, :]
Y_norm = Ytest[0:sep]
X_pneum = Xtest[sep:, :]
Y_pneum = Ytest[sep:]


#-----------------------------------------
aux = train_rbf_svm(Xtrain, Ytrain, opt[0], opt[1])
#-----------------------------------------
#aux = train_linear_svm(Xtrain, Ytrain, opt)
#-----------------------------------------


print("Testing")
#-----------------------------------------
Hrbf = aux.predict(Xtest)
Hrbf_norm = aux.predict(X_norm)
Hrbf_pneum = aux.predict(X_pneum)
#print("  rbf")
#-----------------------------------------
#Hlinear = aux.predict(Xtest)
#Hlinear_norm = aux.predict(X_norm)
#Hlinear_pneum = aux.predict(X_pneum)
#print("  linear")
#-----------------------------------------
print("Tested")


print("----------------Porcentages------------")
#-----------------------------------------
print("Optimum hyperparameters: C-->", opt[0], "Sigma--> ", opt[1])
print("   rbf:")
print("      Total-->", np.sum(Hrbf==Ytest)/len(Ytest))
print("      Normal-->", np.sum(Hrbf_norm==Y_norm)/len(Y_norm))
print("      Pneumonia-->", np.sum(Hrbf_pneum==Y_pneum)/len(Y_pneum))
#-----------------------------------------
#print("Optimum hyperparameters: C-->", opt)
#print("   linear:")
#print("      Total-->", np.sum(Hlinear==Ytest)/len(Ytest))
#print("      Normal-->", np.sum(Hlinear_norm==Y_norm)/len(Y_norm))
#print("      Pneumonia-->", np.sum(Hlinear_pneum==Y_pneum)/len(Y_pneum))
#-----------------------------------------