                                #Iris Classification (KNN)
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv("9-document.csv", names=names, header=0)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10)
classifier = KNeighborsClassifier(n_neighbors=5).fit(Xtrain, ytrain)
ypred = classifier.predict(Xtest)

print("\n-------------------------------------------------------------------------")
print('%-25s %-25s %-25s' % ('Original Label', 'Predicted Label', 'Correct/Wrong'))
print("-------------------------------------------------------------------------")

for i, label in enumerate(ytest):
    print('%-25s %-25s' % (label, ypred[i]), end="")
    if label == ypred[i]:
        print(' %-25s' % ('Correct'))
    else:
        print(' %-25s' % ('Wrong'))

print("-------------------------------------------------------------------------")
print("\nConfusion Matrix:\n", metrics.confusion_matrix(ytest, ypred))
print("-------------------------------------------------------------------------")
print("\nClassification Report:\n", metrics.classification_report(ytest, ypred))
print("-------------------------------------------------------------------------")
print('Accuracy of the classifier is %0.2f' % metrics.accuracy_score(ytest, ypred))
print("-------------------------------------------------------------------------")

