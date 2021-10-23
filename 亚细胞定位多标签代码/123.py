import scipy
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
import numpy as np
data, meta = scipy.io.arff.loadarff("yeast-train.arff")
df = pd.DataFrame(data)
X = df.iloc[:,0:27].values
y = df.iloc[:,27:49].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(int)
def deep_model(feature_dim,label_dim):
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    model.add(Dense(500, activation='relu', input_dim=feature_dim))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(label_dim, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
def train_deep(X_train,y_train,X_test,y_test):
    feature_dim = X_train.shape[1]
    label_dim = y_train.shape[1]
    model = deep_model(feature_dim,label_dim)
    model.summary()
    model.fit(X_train,y_train,batch_size=25, epochs=200,validation_data=(X_test,y_test))
    # model.save('E:\\项目\\亚细胞定位\\yeast\\my_model.h5')#将整个模型保存为HDF5文件
    # model.save_weights('E:\\项目\\亚细胞定位\\yeast\\my_checkpoint') # 保存权重
    y_hat=model.predict_proba(X_test)
    y_hat[y_hat > 0.48] = 1
    y_hat[y_hat <=0.48] =0
    aim = Aiming(y_hat, y_test)
    cov = Coverage(y_hat, y_test)
    acc = Accuracy(y_hat, y_test)
    abst = AbsoluteTrue(y_hat, y_test)
    absf = AbsoluteFalse(y_hat, y_test)
    print("aim:",aim)
    print("cov:",cov)
    print("acc:",acc)
    print("abst:",abst)
    print("absf:",absf)
    return y_hat


def Aiming(y_hat, y):
    '''
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
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
    import numpy as np
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
    import numpy as np
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v,h] == 1 or y[v,h] == 1:
                union += 1
            if y_hat[v,h] == 1 and y[v,h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n


train_deep(X_train,y_train,X_test,y_test)

