## importing the tools I deemed to be necessary.........
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


## getting our data available and running some simple checks......

heart_disease = pd.read_csv('heart_disease.csv')
heart_disease.isna().sum()
heart_disease.head()

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

np.random.seed(0)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


#print(classification_report(y_test, y_pred))
#print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



