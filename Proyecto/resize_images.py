import os
import numpy as np
from PIL import Image as img
from scipy.io import loadmat
from scipy.io import savemat     
from unicodedata import normalize

def resize_image(path, name, dest):
    im = img.open(path+name)
    new = im.resize((256,256))
    if(new.mode != "L"):
        new = new.convert("L")
    new.save(dest+name, "JPEG")
    return np.asarray(new).ravel()

image_dest = os.getcwd()+"\\resized\\"
root = os.getcwd() + "\\"
partes = [("train","xtrain","ytrain"),("val","xval","yval"),("test","xtest","ytest")]
dicti = {}
for i in range(3):
    dest = image_dest + partes[i][0] + "\\"
    orig = root + partes[i][0] + "\\"
    Xs = np.empty((0, 256*256))
    Ys = np.array([], int)
    for dir in os.listdir(orig):
        aux1 = dest + dir + "\\"
        aux2 = orig + dir + "\\"
        for image in os.listdir(aux2):
            x = resize_image(aux2, image, aux1)
            Xs = np.vstack((Xs, x))
            if dir == "NORMAL":
                Ys = np.concatenate([Ys, [0]])
            else:
                Ys = np.concatenate([Ys, [1]])
        print(partes[i][0]+"----->"+dir+" OK")
    dicti[partes[i][1]] = Xs
    dicti[partes[i][2]] = Ys
    

savemat("data.mat", dicti)

print("Fin"*5)