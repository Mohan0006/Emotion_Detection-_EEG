import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneOut

def svm_classifier_leaveoneout(x, y):
      print("\n\n******************************** SVM CLASSIFIER (Leave One Out) *********************************\n")
      svm = SVC()
      clf = MultiOutputClassifier(svm, n_jobs=3)
      loo = LeaveOneOut()
      train_scores = []
      test_scores = []
      for train_index, test_index in loo.split(x):
          x_train, x_test = x[train_index], x[test_index]
          y_train, y_test = y[train_index], y[test_index]
          clf.fit(x_train, y_train)
          y_train_pred = clf.predict(x_train)
          train_scores.append(accuracy_score(y_train, y_train_pred))
          y_test_pred = clf.predict(x_test)
          test_scores.append(accuracy_score(y_test, y_test_pred))
      
      print("Training Accuracy: "+ str(np.mean(train_scores)*100))
      print("Testing Accuracy: "+ str(np.mean(test_scores)*100) +"\n")
