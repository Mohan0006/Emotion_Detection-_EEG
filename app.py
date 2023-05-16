import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.svm_classifier import svm_classifier
from models.dt_classifier import dt_classifier
from models.rf_classifier import rf_classifier
from models.knn_classifier import knn_classifier
from models.svm_classifier_leaveoneout import svm_classifier_leaveoneout
from models.visualize import visualize_channel,visualize_activity,visualize_mobility,visualize_complexity
from models.svm_kfold_classifier import svm_kfold
def main():
    df = pd.read_csv("./features/features_normalized.csv",header = None)
    data = np.array(df)
    att = data[:,0:4]
    labels = data[:,4:6]
    
    x_train, x_test, y_train, y_test = train_test_split(att, labels, test_size=0.1, random_state=42)
    
    # print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)
    
    svm_classifier(x_train,y_train,x_test,y_test)
    dt_classifier(x_train,y_train,x_test,y_test)
    rf_classifier(x_train,y_train,x_test,y_test)
    knn_classifier(x_train,y_train,x_test,y_test)
    svm_classifier_leaveoneout(att,labels)
    svm_kfold(att,labels)
    
    # visualize_channel()
    # visualize_activity(x_train,y_train)
    # visualize_mobility(x_train,y_train)
    # visualize_complexity(x_train,y_train)
    
if __name__ == '__main__':
    main();    