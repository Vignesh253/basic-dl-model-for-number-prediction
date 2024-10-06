import numpy as np
import matplotlib.pyplot as plt
import random
X_train = np.loadtxt(r'C:\Users\vigne\Downloads\numclas\Neural-Network---MultiClass-Classifcation-with-Softmax\train_X.csv', delimiter=',').T
Y_train = np.loadtxt(r'C:\Users\vigne\Downloads\numclas\Neural-Network---MultiClass-Classifcation-with-Softmax\train_label.csv', delimiter=',').T

X_test = np.loadtxt(r'C:\Users\vigne\Downloads\numclas\Neural-Network---MultiClass-Classifcation-with-Softmax\test_X.csv', delimiter=',').T
Y_test = np.loadtxt(r'C:\Users\vigne\Downloads\numclas\Neural-Network---MultiClass-Classifcation-with-Softmax\test_label.csv', delimiter=',').T

print("shape of X_train :",X_train.shape)
print("shape of Y_train:",Y_train.shape)

print("shape of X_test :",X_test.shape)
print("shape of X_test :",Y_test.shape)
index = random.randrange(0,X_train.shape[1])
plt.imshow( X_train [:,index].reshape(28,28))
plt.show()
