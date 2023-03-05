from sklearn import datasets
wine = datasets.load_wine()
print('Features: ', wine.feature_names)
print('Labels: ', wine.target_names)
from sklearn.model_selection import train_test_split
X = wine.data
y = wine.target
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 109)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(xTrain, yTrain)
yPred = gnb.predict(xTest)
from sklearn import metrics
print('Accuracy: ', metrics.accuracy_score(yTest, yPred))