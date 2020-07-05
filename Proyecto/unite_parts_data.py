import numpy as np
from scipy.io import loadmat
from scipy.io import savemat 

print("Loading")
train = loadmat("train300.mat")
print("   train")
val = loadmat("val300.mat")
print("   val")
test = loadmat("test300.mat")
print("Loaded")

train["xval"] = val["xval"]
train["xtest"] = test["xtest"]
train["yval"] = val["yval"]
train["ytest"] = test["ytest"]

print("Saving")
savemat("data300.mat", train)
print("Saved")

#print("Init")
#norm = loadmat("ntrain300.mat")
#pneum = loadmat("ptrain300.mat")
#print("Loaded")
#
#Xn = norm["xtrain"]
#Yn = norm["ytrain"]
#Xp = pneum["xtrain"]
#Yp = pneum["ytrain"]
#
#nX = np.vstack([Xn, Xp])
#nY = np.array([np.concatenate([Yn[0], Yp[0]])])
#
#dicti = {"xtrain": nX, "ytrain":nY}
#
#print("Saving")
#savemat("train300.mat", dicti)
#print("Saved")