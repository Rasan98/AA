import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat     
import scipy.optimize as opt
from sklearn import svm
import matplotlib.gridspec as gridspec

def train_linear_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "linear", C=c)
    svmAux.fit(X,y.ravel())
    return svmAux
def train_rbf_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "rbf", C=c, gamma= 1/(2 * sigma**2))
    svmAux.fit(X,y.ravel())
    return svmAux
def create_dictionary():
    return {"linear":train_linear_svm, "rbf":train_rbf_svm}

def visualize_boundary_mod(X, y, svm, fig):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    fig.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    fig.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    fig.contour(x1, x2, yp)

def print_versions(X, y, file_name, C, kern, sigma=None):
    dic = create_dictionary()
    fig = plt.figure(figsize=(10,20))
    gs = gridspec.GridSpec(nrows=5, ncols=2)
    for i in range(C.shape[0]):
        aux = fig.add_subplot(gs[i//2, i%2])
        aux.set_title("C = " + str(C[i]))
        svmAux = dic[kern](X, y, C[i], sigma)
        visualize_boundary_mod(X, y, svmAux, aux)
    fig.savefig(file_name)

def visualize_boundary(X, y, svm, file_name):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(
    X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.close()

C = np.linspace(0,100,11,dtype=int)
C = np.arange(0, 50, 5)
C[-1] = 100
C[0] = 1

data1 = loadmat("ex6data1.mat")
X1 = data1['X']
Y1 = data1['y']
print_versions(X1, Y1, "data1", C, "linear")


data2 = loadmat("ex6data2.mat")
X2 = data2['X']
Y2 = data2['y']
print_versions(X2, Y2, "data2", C, "rbf", 0.1)

values = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
data3 = loadmat("ex6data3.mat")
X3 = data3['X']
Y3 = data3['y']
X3val = data3['Xval']
Y3val = data3['yval']
m = X3val.shape[0]

pairs = np.empty(64, dtype=tuple)
errors = np.empty(64)
for i in range(8):
    for j in range(8):
        new = (values[i],values[j])
        pairs = np.concatenate(pairs, np.array([new]))
        aux = train_rbf_svm(X3, Y3, new[0], new[1])
        H = aux.predict(X3val.ravel())
        error = np.sum((H - Y3val.ravel())**2)*(1/(2*m))
        errors = np.concatenate(errors, np.array([error]))

opt = pairs[np.argmin(errors)]
aux = train_rbf_svm(X3, Y3, opt[0], opt[1])
visualize_boundary(X3, Y3, aux, "data3")


