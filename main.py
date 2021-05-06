
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv('dates.csv', delimiter=",")

X = dataset.iloc[0:13].values

y = dataset.iloc[:, -1].values

X, y = make_classification(n_samples=100, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regr = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

performance = regr.predict([[32.2,32.3,12,23.2,3.3,2.34,23,23,23,24,32.2,32.3,12,23.2,3.3,2.34,23,23,23,24]])

print (performance[0])

y_pred = regr.predict(X_test)

print(regr.score(X_test, y_test))
