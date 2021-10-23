import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import Dense,BatchNormalization,LeakyReLU
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
def specific(confusion_matrix):
    '''recall = TP / (Tp + FN)'''
    specific = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    return specific

def train_model(x, y):
    skf = StratifiedKFold(shuffle=True, n_splits=5)
    accs = []
    sps = []
    sns = []
    mccs = []
    pres = []
    f1s = []
    count = 0
    under_sample = RandomUnderSampler(sampling_strategy={0:1050000})
    over_sample = SMOTE(sampling_strategy={1: 1050000})
    for train_index, test_index in skf.split(x, y):
        print(count)
        X_train, y_train = x[train_index], y[train_index]
        X_train, y_train = under_sample.fit_resample(X_train, y_train)
        X_train, y_train = over_sample.fit_resample(X_train, y_train)
        X_test, y_test = x[test_index], y[test_index]
        svclf = svm.SVC(C=0.9)
        LGBclf = lgb.LGBMClassifier()
        DNN1 = MLPClassifier(max_iter=32,hidden_layer_sizes=(64,16))
        DNN2 = MLPClassifier(max_iter=64,hidden_layer_sizes=(128,64,32))
        rf1 = XGBClassifier()
        rf = StackingCVClassifier(
            classifiers=[svclf, LGBclf, DNN1,DNN2,rf1], meta_classifier=lrclf, cv=5)
        rf.fit(X_train, y_train)
        prediction = rf.predict(X_test)
        cm = confusion_matrix(y_test, prediction)
        acc = accuracy_score(y_test, prediction)
        sp = specific(cm)
        sn = recall_score(y_test, prediction)
        mcc = matthews_corrcoef(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        accs.append(acc)
        sps.append(sp)
        sns.append(sn)
        mccs.append(mcc)
        pres.append(pre)
        f1s.append(f1)
        count += 1
    print("acc: ", np.mean(accs))
    print("sp: ", np.mean(sps))
    print("sn: ", np.mean(sns))
    print("mcc: ", np.mean(mccs))
    print("pre: ", np.mean(pres))
    print("f1: ", np.mean(f1s))
def load_data():
    features = np.load("ONEHOT_file.npy")
    label_1=np.zeros((2923874,))
    label_2=np.ones((386773,))
    y=np.concatenate((label_1,label_2),axis=0)
    return features, y

if __name__ == "__main__":
    x, y = load_data()
    train_model(x, y)