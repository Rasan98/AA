import numpy as np
from scipy.io import loadmat
from scipy.io import savemat 

def red_data(X, Y, fact):
    newX = X[0:int(len(X)//fact), :]
    newY = Y[0, 0:int(len(X)//fact)]
    return newX, np.array([newY])

print("Load data")
data = loadmat("pneumonia_train300.mat")
X = data["xtrain"]
Y = data["ytrain"]
print("End load")

nX, nY = red_data(X,Y,1.7)

dicti = {"xtrain": nX, "ytrain": nY}

print("Init saving")
savemat("1,7pneumonia_train300.mat", dicti)
print("Saved")