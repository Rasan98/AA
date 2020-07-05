import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat   

print("Loading data")
data = loadmat("data300.mat")
print("Data loaded")

y1 = data["ytrain"]
y2 = data["ytest"]
y3 = data["yval"]


pneum = y1.sum() + y2.sum() + y3.sum()
norm = y1.shape[1] + y2.shape[1] + y3.shape[1] - pneum


plt.figure()
plt.bar(["Neumonia", "Normal"], [pneum, norm], width=0.5) 
plt.ylabel("Número de casos")
plt.savefig("bar_chart.png")