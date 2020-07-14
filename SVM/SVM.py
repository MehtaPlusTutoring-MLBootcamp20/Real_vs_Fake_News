#comment
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import numpy as np

news = datasets._____() #add dataset file name in blank

X = news.data  
y = news.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=100) #split data for traininga

clf = svm.SVC(kernel = #"linear"?) #SVM Model, default C-value
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test) #accuracy
print("This is the accuracy " + accuracy)

C_values = [0.001, 0.01, 0.1, 1] #test various C-values to determine which gives the best accuraccy
for i in C_values:
  clf = svm.SVC(kernel = "linear", C=i)
  clf.fit(X_train, Y_train)
  print(i, clf.score(X_test, Y_test))

kernels = ['linear', 'rbf', 'poly','sigmoid', 'precomputed']
for kernel in kernels:
  svc = svm.SVC(kernel=kernel).fit(X_train, Y_train)
  print(kernel, clf.score(X_test, Y_test))

gammas = [0.1, 1, 10, 100]
for gamma in gammas:
  svc = svm.SVC(kernel='rbf', gamma=gamma).fit(X_train, Y_train)
  print(gamma, clf.score(X_test, Y_test))

cs = [0.1, 1, 10, 100, 1000]
for c in cs:
  svc = svm.SVC(kernel='rbf', C=c).fit(X_train, Y_train)
  print(c, clf.score(X_test, Y_test))

degrees = [0, 1, 2, 3, 4, 5, 6]
for degree in degrees:
  svc = svm.SVC(kernel='poly', degree=degree).fit(X_train, Y_train)
  print(degree, clf.score(X_test, Y_test))