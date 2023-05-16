from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

def knn_classifier(x_train,y_train,x_test,y_test):
      print("\n\n******************************** KNN CLASSIFIER *********************************\n")
      for k in [1,2,3,4,5,6,7,8,9,10]:
       print("=> K =",k,"\n")     
       clf = KNeighborsClassifier(n_neighbors=k)
       clf = MultiOutputClassifier(clf, n_jobs=3)
       clf.fit(x_train, y_train)
       y_pred = clf.predict(x_train)
       print("Training Accuracy: "+ str(accuracy_score(y_train, y_pred)*100))
       y_pred = clf.predict(x_test)
       print("Testing Accuracy: "+ str(accuracy_score(y_test, y_pred)*100) +"\n\n")