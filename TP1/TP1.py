from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
import pandas as pd


dataframe = pd.read_csv("creditcard.csv")

# print(database.head)
# print(database.shape)
# print(database.info)
# print(database.describe)
# print(database.head)
# print(database.value)

# del(dataframe["Time"])

print(dataframe.head)

train,test = train_test_split(dataframe)


x_train = train
y_train = x_train["Class"]

x_test = test
y_test = x_test["Class"]

print(x_train.shape)
print(y_train.shape)

classifierTree = tree.DecisionTreeClassifier("entropy")
classifierTree.fit(x_train,y_train)

y_pred_tree = classifierTree.predict(x_test)

print("===== TREE =====")
print("r2 {}".format(r2_score(y_pred_tree,y_test)))
print("MAE {}".format(mean_absolute_error(y_pred_tree,y_test)))
print("MSE {}".format(mean_squared_error(y_pred_tree,y_test)))
print("Score {}".format(classifierTree.score(x_test,y=y_test)))
print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_tree)))
print("================")

RFC = RandomForestClassifier(n_estimators=10)
RFC.fit(x_train,y_train)

y_pred_rfc = RFC.predict(X=x_test)

print("===== RFC =====")
print("r2 {}".format(r2_score(y_pred_rfc,y_test)))
print("MAE {}".format(mean_absolute_error(y_pred_rfc,y_test)))
print("MSE {}".format(mean_squared_error(y_pred_rfc,y_test)))
print("Score {}".format(RFC.score(x_test,y_test)))
print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_rfc)))
print("================")

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(x_train,y_train)

y_pred_knn = KNN.predict(x_test)

print("===== KNN =====")
print("r2 {}".format(r2_score(y_pred_knn,y_test)))
print("MAE {}".format(mean_absolute_error(y_pred_knn,y_test)))
print("MSE {}".format(mean_squared_error(y_pred_knn,y_test)))
print("Score {}".format(KNN.score(x_test,y_test)))
print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_knn)))
print("================")

mlpc = MLPClassifier()
mlpc.fit(x_train,y_train)

y_pred_mlpc = mlpc.predict(x_test)

print("===== MLPC =====")
print("r2 {}".format(r2_score(y_pred_mlpc,y_test)))
print("MAE {}".format(mean_absolute_error(y_pred_mlpc,y_test)))
print("MSE {}".format(mean_squared_error(y_pred_mlpc,y_test)))
print("Score {}".format(mlpc.score(x_test,y_test)))
print("Confusion Matrix \n {}".format(confusion_matrix(y_test,y_pred_mlpc)))
print("================")
