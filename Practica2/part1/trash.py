plt.figure()
aux = np.where(Y == 1)
plt.scatter(X[aux, 1], X[aux, 2], c="black", label="admitted", marker="+")
aux = np.where(Y == 0)
plt.scatter(X[aux, 1], X[aux, 2], c="yellow", label="not admitted", marker="o")
#plt.plot(np.random.uniform(30, 100, 100), hipotesis(X, theta_opt))
plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.savefig("data.png")