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