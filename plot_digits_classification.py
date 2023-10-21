from sklearn import datasets
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=42)

from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
production_pred = clf.predict(X_test)
production_acc = clf.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [10, 20, 30, None]}
dtc = DecisionTreeClassifier()
clf = GridSearchCV(dtc, parameters)
clf.fit(X_train, y_train)
candidate_pred = clf.predict(X_test)
candidate_acc = clf.score(X_test, y_test)

from sklearn.metrics import confusion_matrix, f1_score
conf_matrix_prod_cand = confusion_matrix(production_pred, candidate_pred)
conf_matrix_correct_incorrect = confusion_matrix((y_test == production_pred).astype(int), (y_test == candidate_pred).astype(int))
f1_macro = f1_score(y_test, production_pred, average='macro')
