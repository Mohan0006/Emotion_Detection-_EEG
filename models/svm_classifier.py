import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from sklearn.multioutput import MultiOutputClassifier


def svm_classifier(x_train,y_train,x_test,y_test):
      print("\n\n******************************** SVM CLASSIFIER *********************************\n")
      svm = SVC()
      clf = MultiOutputClassifier(svm, n_jobs=3)
      clf.fit(x_train, y_train)
      y_pred = clf.predict(x_train)
      print("Training Accuracy: "+ str(accuracy_score(y_train, y_pred)*100))
      y_pred = clf.predict(x_test)
      print("Testing Accuracy: "+ str(accuracy_score(y_test, y_pred)*100) +"\n")
      
      