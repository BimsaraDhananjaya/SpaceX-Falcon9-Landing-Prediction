# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import asyncio
import aiohttp
import requests

import io
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
# Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standardize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


def plot_confusion_matrix(y, y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'land'])
    ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.show()


URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
response1 = requests.get(URL1)
text1 = io.BytesIO(response1.content)
data = pd.read_csv(text1)


async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            return data


async def main():
    URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
    resp2 = await fetch_data(URL2)
    text2 = io.BytesIO(resp2)
    X = pd.read_csv(text2)
    return X

# Define a synchronous function to run the asynchronous main function


def run_main():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
    return result


# Call the run_main function and display the first 100 rows
result = run_main()
result.head(100)
print(result.head(100))

Y = data['Class'].to_numpy()
Y.dtype

transform = preprocessing.StandardScaler()
X = transform.fit_transform(result)
X

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

print('Train set:')
print('X-train= ', X_train.shape, 'Y-train= ', Y_train.shape)
print('Test set:')
print('X-test= ', X_test.shape,  'Y-test= ', Y_test.shape)

parameters = {'C': [0.01, 0.1, 1],
              'penalty': ['l2'],
              'solver': ['lbfgs']}

parameters = {"C": [0.01, 0.1, 1], 'penalty': [
    'l2'], 'solver': ['lbfgs']}  # l1 lasso l2 ridge
# Create a logistic regression object
lr = LogisticRegression()

# Create a GridSearchCV object logreg_cv
logreg_cv = GridSearchCV(lr, parameters, cv=10)

# Fit the training data into the GridSearch object
logreg_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
print("accuracy :", logreg_cv.best_score_)

print("Logistic Regression test data accuracy :",
      logreg_cv.score(X_test, Y_test))
yhat = logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)
plt.show()

parameters = {'kernel': ('linear', 'rbf', 'poly', 'rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma': np.logspace(-3, 3, 5)}
# Create a support vector machine
svm = SVC()
# create a GridSearchCV object svm_cv with cv = 10
svm_cv = GridSearchCV(svm, parameters, cv=10)

# Fit the training data into the GridSearch object
svm_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ", svm_cv.best_params_)
print("accuracy :", svm_cv.best_score_)

print("SVM test data accuracy :", svm_cv.score(X_test, Y_test))

yhat = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)
plt.show()

parameters = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [2*n for n in range(1, 10)],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10]}

# Create a decision tree classifier object
tree = DecisionTreeClassifier()

# create a GridSearchCV object tree_cv with cv = 10
tree_cv = GridSearchCV(tree, parameters, cv=10)

# Fit the training data into the GridSearch object
tree_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ", tree_cv.best_params_)
print("accuracy :", tree_cv.best_score_)

print("Decision Tree accuracy on test set :", tree_cv.score(X_test, Y_test))
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)
plt.show()

parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1, 2]}

# Create a KNN classifier object
KNN = KNeighborsClassifier()

# create a GridSearchCV object knn_cv with cv = 10
knn_cv = GridSearchCV(KNN, parameters, cv=10)

# Fit the training data into the GridSearch object
knn_cv.fit(X_train, Y_train)
print("tuned hpyerparameters :(best parameters) ", knn_cv.best_params_)
print("accuracy :", knn_cv.best_score_)
knn_cv.score(X_test, Y_test)
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)
plt.show()

Report = pd.DataFrame({'Method': ['Test Data Accuracy']})

knn_accuracy = knn_cv.score(X_test, Y_test)
Decision_tree_accuracy = tree_cv.score(X_test, Y_test)
SVM_accuracy = svm_cv.score(X_test, Y_test)
Logistic_Regression = logreg_cv.score(X_test, Y_test)

Report['Logistic_Reg'] = [Logistic_Regression]
Report['SVM'] = [SVM_accuracy]
Report['Decision Tree'] = [Decision_tree_accuracy]
Report['KNN'] = [knn_accuracy]


Report.transpose()
print(Report.transpose())
