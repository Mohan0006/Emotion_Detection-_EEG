from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier


def rf_classifier(x_train,y_train,x_test,y_test):
      print("\n\n******************************** RF CLASSIFIER *********************************\n")
      clf = RandomForestClassifier(n_estimators=1000,random_state=42)
      clf = MultiOutputClassifier(clf, n_jobs=3)
      clf.fit(x_train, y_train)
      y_pred = clf.predict(x_train)
      print("Training Accuracy: "+ str(accuracy_score(y_train, y_pred)*100))
      y_pred = clf.predict(x_test)
      print("Testing Accuracy: "+ str(accuracy_score(y_test, y_pred)*100) +"\n")
      