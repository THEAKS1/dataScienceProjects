import pandas as pd

data = pd.read_csv("mushrooms.csv")

from sklearn.preprocessing import LabelEncoder

for i in range(len(data.columns)):
    labelEncoder = LabelEncoder()
    data.iloc[:,i] = labelEncoder.fit_transform(data.iloc[:,i])
    
x = data.iloc[:,1:]
y = data.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# ------ Logistic Regression ----------
from sklearn.linear_model import LogisticRegression
logRegressor = LogisticRegression()
logRegressor.fit(x_train, y_train)

logPredict = logRegressor.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, logPredict)


# --------------- KNN ---------------- 
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier()
knnClassifier.fit(x_train, y_train)

knnPredict = knnClassifier.predict(x_test)

cm_knn = confusion_matrix(y_test, knnPredict)


# -------------- kernel SVM -----------------
from sklearn.svm import SVC
KsvmClassifier = SVC(kernel="rbf")
KsvmClassifier.fit(x_train, y_train)

KsvmPredict = KsvmClassifier.predict(x_test)

Ksvm_cm = confusion_matrix(y_test, KsvmPredict)


# -------------- SVM -----------------
from sklearn.svm import SVC
svmClassifier = SVC(kernel = "linear")
svmClassifier.fit(x_train, y_train)

svmPredict = svmClassifier.predict(x_test)

svm_cm = confusion_matrix(y_test, svmPredict)


# ---------- Naive Bayes --------------
from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(x_train, y_train)

NBPredict = NBclassifier.predict(x_test)

NBcm = confusion_matrix(y_test, NBPredict)


# -------- Decision Tree ------------
from sklearn.tree import DecisionTreeClassifier
DTreeClassifier = DecisionTreeClassifier(criterion="entropy")
DTreeClassifier.fit(x_train, y_train)

DTreePredict = DTreeClassifier.predict(x_test)

DTreecm = confusion_matrix(y_test, DTreePredict)


# ---------- Random Forest ----------------
from sklearn.ensemble import RandomForestClassifier
RFClassifier = RandomForestClassifier(criterion="entropy")
RFClassifier.fit(x_train, y_train)

RFPredict = RFClassifier.predict(x_test)

RFcm = confusion_matrix(y_test, RFPredict)

predicted_values = [logPredict, knnPredict, KsvmPredict, svmPredict, NBPredict, DTreePredict, RFPredict]

from sklearn.metrics import accuracy_score
accuracyScores = [accuracy_score(y_test, x) for x in predicted_values]