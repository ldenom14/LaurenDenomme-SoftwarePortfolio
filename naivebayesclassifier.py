from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#load dataset

iris = load_iris()

#print(iris)


X = iris.data #contains feature data: sepal length, sepal width, petal length, and petal width
y = iris.target #contains species information

#print(X)
#print(y)

#reate and print likelihood table 

likelihood_table = {}

for class_label in np.unique(y):
    class_mask = (y == class_label)
    class_X = X[class_mask]
    class_mean = np.mean(class_X, axis=0)
    class_standard_dev= np.std(class_X, axis= 0)
    likelihood_table[class_label] = {'mean': class_mean, 'std': class_standard_dev}

for class_label, class_info in likelihood_table.items():
    print(f"Class Label: {class_label}")
    for feature_index, (mean, standard_dev) in enumerate(zip(class_info['mean'], class_info['std'])):
        print(f"Feature {feature_index}: Mean= {mean}, Standard Dev= {standard_dev}")

#split the training data and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#initialize and train NB classifier

gnb = GaussianNB(var_smoothing=1e-9) #Laplacian correction
gnb.fit(X_train, y_train)

#posterior probability
y_prob = gnb.predict_proba(X_test)

print("Posterior Probabilities:")
for i in range(5): #print for the first 5 samples
    print(f"Sample {i+1}: {y_prob[i]}")


