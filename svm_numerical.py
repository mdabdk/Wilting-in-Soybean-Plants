import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn import preprocessing

from sklearn.metrics import precision_recall_fscore_support

# get weather data

train_data = pd.read_csv('TrainData-C2/labels/ExtraCredit_Train.csv')

test_data = pd.read_csv('TrainData-C2/labels/ExtraCredit_Test.csv')

# extract training data

X_train = train_data.iloc[:,2:].to_numpy()

y_train = train_data.iloc[:,1].to_numpy()

# scale training data to range [-1,1]

X_train = preprocessing.minmax_scale(X_train,feature_range = (-1,1))

# extract test data

X_test = test_data.iloc[:,2:].to_numpy()

y_test = test_data.iloc[:,1].to_numpy()

# scale test data to range [-1,1]

X_test = preprocessing.minmax_scale(X_test,feature_range = (-1,1))

# initialize polynomial SVM classifier

clf = SVC(C = 1,
          kernel = 'poly',
          degree = 3,
          coef0 = 0,
          gamma = 'scale',
          tol = 1e-5,
          class_weight = 'balanced',
          random_state = 42)

# start training

clf.fit(X_train, y_train)

# get predictions on test data

y_hat = clf.predict(X_test)

# compute precision, recall, and F_1 scores

(precision,
recall,
f1_score,
support) = precision_recall_fscore_support(y_test,
                                           y_hat,
                                           beta = 1.0,
                                           average = None)

# compute testing accuracy
                                           
test_acc = np.mean(np.equal(y_test,y_hat))