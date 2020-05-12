import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat     
from sklearn import svm
import get_vocab_dict as gvd
import process_email as pm
import codecs

def train_linear_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "linear", C=c)
    svmAux.fit(X,y.ravel())
    return svmAux


def train_rbf_svm(X, y, c, sigma):
    svmAux = svm.SVC(kernel= "rbf", C=c, gamma= 1/(2 * sigma**2))
    svmAux.fit(X,y.ravel())
    return svmAux


def process_email(name, dicti):
    print(name)
    if name == "spam/0340.txt":
        return np.zeros(1899)
    email_contents = codecs.open(name, 'r', encoding='utf-8', errors='ignore').read()
    email = np.array(pm.email2TokenList(email_contents))
    index = np.vectorize(dicti.get)(email, -1)
    index = index[index!=-1]
    index = index-1
    vect = np.zeros(1899)
    vect[index] = 1
    return vect


def transform_data(): #used in console to save the email's vectors in a .mat file
    easy = np.empty((0, 1899))
    hard = np.empty((0, 1899))
    spam = np.empty((0, 1899))
    dicti = gvd.getVocabDict()
    for i in np.arange(250):
        hard = np.vstack((hard, process_email("hard_ham/{0:04d}.txt".format(i+1), dicti)))
        spam = np.vstack((spam, process_email("spam/{0:04d}.txt".format(i+1), dicti)))
        easy = np.vstack((easy,process_email("easy_ham/{0:04d}.txt".format(i+1), dicti)))
    for i in np.arange(250,500):
        spam = np.vstack((spam, process_email("spam/{0:04d}.txt".format(i+1), dicti)))
        easy = np.vstack((easy,process_email("easy_ham/{0:04d}.txt".format(i+1), dicti)))
    for i in np.arange(500,2551):
        easy = np.vstack((easy,process_email("easy_ham/{0:04d}.txt".format(i+1), dicti)))
    spam = np.delete(spam, 339,0)#email spam/0340.txt doesn't work
    my_dict = {"easy":easy, "hard":hard, "spam":spam}
    savemat("email_vectors.mat", my_dict)


def distribute_data(easy, hard, spam, porcent, data_divisor=1): #1--> spam; 0 --> not spam
    #train 60%
    teasy = int(np.ceil((porcent[0]*easy.shape[0]/100)/data_divisor))
    thard = int(np.ceil((porcent[0]*hard.shape[0]/100)/data_divisor))
    tspam = int(np.ceil((porcent[0]*spam.shape[0]/100)/data_divisor))
    aux = np.vstack((easy[:teasy,:], hard[:thard,:]))
    Xt = np.vstack((aux, spam[:tspam,:]))
    Yt = np.concatenate((np.zeros(teasy+thard), np.ones(tspam)))
    #calculate_hyperparameters 20%
    veasy = int(np.ceil((porcent[1]*easy.shape[0]/100)/data_divisor))
    vhard = int(np.ceil((porcent[1]*hard.shape[0]/100)/data_divisor))
    vspam = int(np.ceil((porcent[1]*spam.shape[0]/100)/data_divisor))
    aux = np.vstack((easy[teasy:(veasy+teasy),:], hard[thard:(vhard+thard),:]))
    Xval = np.vstack((aux, spam[tspam:(vspam+tspam),:]))
    Yval = np.concatenate((np.zeros(veasy+vhard), np.ones(vspam)))
    #calculate_error 20%
    aux = np.vstack((easy[(teasy+veasy):,:], hard[(thard+vhard):,:]))
    Xtest = np.vstack((aux, spam[(tspam+vspam):,:]))
    aux1 = (easy.shape[0] - (teasy + veasy)) + (hard.shape[0] - (thard + vhard))
    aux2 =  spam.shape[0] - (tspam + vspam)
    Ytest = np.concatenate((np.zeros(aux1), np.ones(aux2)))
    return (Xt, Yt), (Xval, Yval), (Xtest, Ytest)

data = loadmat("email_vectors.mat")
easy = data["easy"]
hard = data["hard"]
spam = data["spam"]

train, cross_val, test = distribute_data(easy, hard, spam, [60, 20, 20])

values = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

pairs = np.empty((64,2))
errors = np.empty(64)
my_dict = {}
for i in range(8):
    for j in range(8):
        print("Train -->", 8*i + j)
        new = np.array([values[i],values[j]])
        pairs[8*i + j] = new
        aux = train_rbf_svm(train[0], train[1], new[0], new[1])
        H = aux.predict(cross_val[0])
        error = np.sum((H - cross_val[1])**2)*(1/(2*cross_val[1].shape[0])) #pylint: disable=unsubscriptable-object
        errors[8*i + j] = error
        my_dict[str(8*i + j)] = (new,error)

savemat("hyper.mat", my_dict)

opt = pairs[np.argmin(errors)]
print("optimum hyperparameters -->", opt)




aux = train_rbf_svm(train[0], train[1], opt[0], opt[1])
H = aux.predict(test[0])
error = np.sum((H - test[1])**2)*(1/(2*test[1].shape[0]))#pylint: disable=unsubscriptable-object

print(error)

print("Precision -->", np.sum(H==test[1])/test[1].shape[0])#pylint: disable=unsubscriptable-object