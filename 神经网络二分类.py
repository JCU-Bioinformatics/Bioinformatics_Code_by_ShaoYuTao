import scipy
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    '''

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n

def Coverage(y_hat, y):
    '''
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    '''
    import numpy as np
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])
    return sorce_k / n

def Accuracy(y_hat, y):
    '''
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
    '''
    import numpy as np
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n

def AbsoluteTrue(y_hat, y):
    '''
    错误一个即为零
    '''
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n

def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''
    n, m = y_hat.shape
    sorce_k = 0
    union = 0
    intersection = 0
    for v in range(n):
        if y_hat[v].all == y[v].all:
            union += 1
        for h in range(m):
            if y_hat[v,h] == y[v,h]:
                intersection += 1
                break
    print(n)
    print(intersection)
    return (n - intersection)/n

def recall(confusion_matrix):
    '''recall = TP / (Tp + FN)'''
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    return recall

def MCC(confusion_matrix):
    '''MCC = (TP*TN - FP*FN) / ((TP + FP)(TP + FN)(TN + FP)(TN)) ** 0.5'''
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    MCC = (TP*TN - FP*FN) / ((TP + FP)*(TP + FN)*(TN + FP)*(TN+FN)) ** 0.5
    return MCC

def specific(confusion_matrix):
    '''sp = TN / (TN + FP)'''
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    sp = TN / (TN + FP)
    return sp


data, meta = scipy.io.arff.loadarff("2932序列30特征 +元胞自动机.csv.arff")
df = pd.DataFrame(data)
X = df.iloc[:,0:30].values
Y = df.iloc[:,30:31].values
results = {}
confusion_matrixs = {}
kf = KFold(n_splits=5, shuffle=True)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras import backend as K
K.clear_session()
tf.reset_default_graph()
counter = 1
for train,test in kf.split(X,Y):
    location_confusion_matrixs = {}
    # aims = []
    # covs = []
    # accs = []
    # absts = []
    # absfs = []
    accs = []
    mccs= []
    sns = []
    sps = []
    X_train = X[train]
    Y_train = Y[train]
    X_test = X[test]
    Y_test = Y[test]
    Y_train = Y_train.astype(np.float64)
    Y_test = Y_test.astype(np.float64).flatten()
    # aim, cov, acc, abst, absf=train_deep(X_train,Y_train,X_test,Y_test)

    feature_dim = X_train.shape[1]
    label_dim = Y_train.shape[1]
    model = Sequential()
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    model.add(Dense(128, activation='relu', input_dim=feature_dim))
    # model.add(BatchNormalization(trainable=True))
    # model.add(Dense(150, activation='relu'))
    # model.add(BatchNormalization(trainable=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(BatchNormalization(trainable=True))
    # model.add(Dense(50, activation='relu'))
    # model.add(BatchNormalization(trainable=True))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model = deep_model(feature_dim,label_dim)
    model.summary()
    model.fit(X_train,Y_train,batch_size=8, epochs=128,validation_data=(X_test,Y_test))
    prediction = model.predict(X_test)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] =0
    # print(prediction)
    cm = confusion_matrix(Y_test, prediction)
    acc = accuracy_score(Y_test, prediction)
    sp = specific(cm)
    sn = recall(cm)
    mcc = MCC(cm)
    accs.append(acc)
    sns.append(sn)
    mccs.append(mcc)
    sps.append(sp)
print("acc:\t", np.mean(accs))
print("s p:\t", np.mean(sps))
print("s n:\t", np.mean(sns))
print("mcc:\t", np.mean(mccs))

    # y_hat=model.predict_proba(X_test)
    # y_hat[y_hat >= 0.5] = 1
    # y_hat[y_hat < 0.5] =0

    # aim = Aiming(y_hat, Y_test)
    # cov = Coverage(y_hat, Y_test)
    # acc = Accuracy(y_hat, Y_test)
    # abst = AbsoluteTrue(y_hat, Y_test)
    # absf = AbsoluteFalse(y_hat, Y_test)
    # aims.append(aim)
    # covs.append(cov)
    # accs.append(acc)
    # absts.append(abst)
    # absfs.append(absf)
    # for location in range(Y.shape[1]):
    #     location_confusion_matrix = confusion_matrix(y_hat[:,location], Y_test[:,location])
    #     location_confusion_matrixs[str(location)] = location_confusion_matrix
    # confusion_matrixs[str(counter)] = location_confusion_matrixs
    # counter += 1
    # print(aims)
    # print(covs)
    # print(accs)
    # print(absts)
    # print(absfs)·
# print("aim:",np.mean(aims))
# print("cov:",np.mean(covs))
# print("acc:",np.mean(accs))
# print("abst:",np.mean(absts))
# print("absf:",np.mean(absfs))
# np.save('matrixs_3.npy', confusion_matrixs)

# import numpy as np
# dic = np.load('results.npy',allow_pickle=True).item()
# print(dic)
# matrixs = np.load('matrixs_3.npy', allow_pickle=True).item()
# Acc = {}
# SN = {}
# SP = {}
# MCC = {}
# a = []
# count = 0
# for i in range(22):
#     acc= 0
#     sn = 0
#     sp = 0
#     mcc = 0
#     for j in range(1, 6):
#         confusion_matrix =matrixs[str(j)][str(i)]
#         a.append(confusion_matrix.shape)
#         TP = confusion_matrix[1, 1]
#         FP = confusion_matrix[0, 1]
#         FN = confusion_matrix[1, 0]
#         TN = confusion_matrix[0, 0]
#         acc += (TP + TN) / (TP + TN + FP + FN)
#         sn += TP / (TP + FN)
#         sp += TN / (TN + FP)
#         mcc += ((TP * TN) - (FP * FN)) / (np.sqrt((TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)))
#     Acc[str(i)] = acc / 5
#     SN[str(i)] = sn / 5
#     SP[str(i)] = sp / 5
#     MCC[str(i)] = mcc / 5
# print(Acc)
# print(SN)
# print(SP)
# print(MCC)