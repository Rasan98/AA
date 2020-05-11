import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat     
from sklearn import svm
import get_vocab_dict
import process_email

def train_linear_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "linear", C=c)
    svmAux.fit(X,y.ravel())
    return svmAux
def train_rbf_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "rbf", C=c, gamma= 1/(2 * sigma**2))
    svmAux.fit(X,y.ravel())
    return svmAux

def transform_data():
    easy = np.array([])
    hard = np.array([])
    no_spam = np.array([])
    return easy, hard, no_spam

def distribute_data(easy, hard, no_spam, data_divisor=1):
    #train 60%
    Xt = np.array([])
    Yt = np.array([])
    #calculate_hyperparameters 20%
    Xval = np.array([])
    Yval = np.array([])
    #calculate_error 20%
    Xtest = np.array([])
    Ytest= np.array([])
    return (Xt, Yt), (Xval, Yval), (Xtest, Ytest)


easy, hard, no_spam = transform_data()

train, cross_val, test = distribute_data(easy, hard, no_spam, 4)

values = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

pairs = np.empty((64,2))
errors = np.empty(64)
for i in range(8):
    for j in range(8):
        new = np.array([values[i],values[j]])
        pairs[8*i + j] = new
        aux = train_rbf_svm(train[0], train[1], new[0], new[1])
        H = aux.predict(cross_val[0])
        error = np.sum((H - cross_val[1])**2)*(1/(2*cross_val[1].shape[0])) #pylint: disable=unsubscriptable-object
        errors[8*i + j] = error

opt = pairs[np.argmin(errors)]
print("optimum hyperparameters -->", opt)


aux = train_rbf_svm(train[0], train[1], opt[0], opt[1])
H = aux.predict(test[0])
error = np.sum((H - test[1])**2)*(1/(2*test[1].shape[0]))#pylint: disable=unsubscriptable-object