import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors

#### Pre-processing the dataset ####
file_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(file_path, "./resources/dataset/breast-cancer-wisconsin.data")

df = pd.read_csv(data_path)
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#### End Pre-processing the dataset ####

#### Building and Training Classifier ####
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
#### End Building and Training Classifier ####

#### Testing and Accuracy ####
accuracy = clf.score(X_test, y_test)
print(accuracy)
#### End Testing and Accuracy ####





