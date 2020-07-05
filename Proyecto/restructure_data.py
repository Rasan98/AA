import numpy as np
from scipy.io import loadmat
from scipy.io import savemat 

def unite_data(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
    septrain = np.where(Ytrain == 1)[1][0]
    sepval = np.where(Yval == 1)[1][0]
    septest = np.where(Ytest == 1)[1][0]
    normal = np.vstack([Xtrain[0:septrain, :], Xval[0:sepval, :], Xtest[0:septest, :]])
    pneum = np.vstack([Xtrain[septrain:, :], Xval[sepval:, :], Xtest[septest:, :]])
    return normal, pneum



print("Loading data")
data = loadmat("data300.mat")
print("Loaded")

Xtrain = data["xtrain"]
Ytrain = data["ytrain"]
Xval = data["xval"]
Yval = data["yval"]
Xtest = data["xtest"]
Ytest = data["ytest"]


print("Uniting data")
normal, pneum = unite_data(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
print("United")

print("Restructuring data")

ntr = int(60*len(normal)/100)
ptr = int(60*len(pneum)/100)
nvt = int(20*len(normal)/100)
pvt = int(20*len(pneum)/100)


newXtr = np.vstack([normal[0:ntr,:], pneum[0:ptr,:]])
newYtr = np.array([np.concatenate([np.zeros(ntr), np.ones(ptr)])])
newXval = np.vstack([normal[ntr:ntr+nvt,:], pneum[ptr:ptr+pvt,:]])
newYval = np.array([np.concatenate([np.zeros(nvt), np.ones(pvt)])])
newXte = np.vstack([normal[ntr+nvt:,:], pneum[ptr+pvt:,:]])
newYte = np.array([np.concatenate([np.zeros(len(normal[ntr+nvt:,:])), np.ones(len(pneum[ptr+pvt:,:]))])])
print("Restructured data")

dicti = {"xtrain": newXtr, "ytrain": newYtr, "xval": newXval, "yval": newYval, "xtest": newXte, "ytest": newYte}

print("Saving data")
savemat("60_20_20_data300.mat", dicti)
print("Saved")