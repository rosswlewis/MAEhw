import os
os.chdir('C:/Users/rossw/Documents/MAE Program/Q2/Machine Learning 425/Homework 4/PartII_svm')
import numpy as np
import download_data as dl
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn import metrics
from conf_matrix import func_confusion_matrix

## step 1: load data from csv file. 
data = dl.download_data('crab.csv').values

n = 200
#split data 
S = np.random.permutation(n)
#100 training samples
Xtr = data[S[:100], :6]
Ytr = data[S[:100], 6:]
# 100 testing samples
X_test = data[S[100:], :6]
Y_test = data[S[100:], 6:].ravel()

## step 2 randomly split Xtr/Ytr into two even subsets: use one for training, another for validation.
#############placeholder 1: training/validation #######################
n2 = len(Xtr)
S2 = np.random.permutation(n2)

# subsets for training models
x_train = Xtr[S2[:int(n2/2)]]
y_train = Ytr[S2[:int(n2/2)]]
# subsets for validation
x_validation = Xtr[S2[int(n2/2):]]
y_validation = Ytr[S2[int(n2/2):]]
#############placeholder end #######################

## step 3 Model selection over validation set
# consider the parameters C, kernel types (linear, RBF etc.) and kernel
# parameters if applicable. 


# 3.1 Plot the validation errors while using different values of C ( with other hyperparameters fixed) 
#  keeping kernel = "linear"
#############placeholder 2: Figure 1#######################
c_range = range(1,11) #
svm_c_error = []
for c_value in c_range:
    model = svm.SVC(kernel='linear', C=c_value)
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validation, y_validation)
    svm_c_error.append(error)
print(svm_c_error)
plt.plot(c_range, svm_c_error)
plt.title('Linear SVM')
plt.xlabel('c values')
plt.ylabel('error')
#plt.xticks(c_range)
plt.show()
#############placeholder end #######################


# 3.2 Plot the validation errors while using linear, RBF kernel, or Polynomial kernel ( with other hyperparameters fixed) 
#############placeholder 3: Figure 2#######################
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
for kernel_value in kernel_types:
    # your own codes
    model = svm.SVC(kernel=kernel_value, C=5)
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validation, y_validation)
    svm_kernel_error.append(error)

plt.plot(kernel_types, svm_kernel_error)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()
#############placeholder end #######################


## step 4 Select the best model and apply it over the testing subset 
#############placeholder 4:testing  #######################

best_kernel = 'linear'
best_c = 5 # poly had many that were the "best"
model = svm.SVC(kernel=best_kernel, C=best_c)
model.fit(X=x_train, y=y_train)

error = 1. - model.score(X_test, Y_test)
print(error)

#############placeholder end #######################


## step 5 evaluate your results in terms of accuracy, real, or precision. 

#############placeholder 5: metrics #######################
# func_confusion_matrix is not included
# You might re-use this function for the Part I. 
y_pred = model.predict(X_test)
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(Y_test, y_pred)

print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))

#############placeholder end #######################

#############placeholder 6: success and failure examples #######################
# Success samples: samples for which you model can correctly predict their labels
# Failure samples: samples for which you model can not correctly predict their labels

dif = []
for i in range(len(Y_test)):
    if(Y_test[i] != y_pred[i]):
        dif.append(i)
        
for i in range(len(dif)):
    print("predicted",y_pred[dif[i]])
    print('actual',Y_test[dif[i]])
    print('data',X_test[dif[i]])
    
sam = []
for i in range(len(Y_test)):
    if(Y_test[i] == y_pred[i]):
        sam.append(i)
        
for i in range(5):
    print("predicted",y_pred[sam[i]])
    print('actual',Y_test[sam[i]])
    print('data',X_test[sam[i]])

#############placeholder end #######################



