import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

class PseAAC:
    path = "AAindex.xlsx"

    def __init__(self, k=3, w=0.05):
        '''
        :param k: lambda，默认为3，生成20+lambda特征向量
        :param w: 权重系数,默认0.05
        '''
        self.k = k
        self.w = w

    def __read_index(self):
        '''
        :return:
            acid_names:伪氨基酸名称
            aaindex:物化数值
        '''
        data = pd.read_excel(self.path)
        acid_names = list(data.columns)
        acid_names.remove('Amino acid')
        aaindex = data.iloc[0:, 1:].values
        return acid_names, aaindex

    def __encode_sequence(self, seq):
        '''
        :param seq:输入序列
        :return:
            encoded_seq: shape(6, len(seq))
        '''
        acid_names, aaindex = self.__read_index()
        seq = np.reshape(np.array(list(seq)), (-1, 1))
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(np.reshape(acid_names, (-1, 1)))
        onehot_meta_sequence = onehot_encoder.transform(seq)
        encoded_seq = np.dot(aaindex, onehot_meta_sequence.T)
        return encoded_seq, np.mean(aaindex, axis=1), np.std(aaindex, axis=1)

    def __get_AAc(self, seq):
        '''
        :param seq: 输入序列
        :return: AAC特征
        '''
        chars = list(seq)
        list_aac = []
        acid = "#ACDEFGHIKLMNPQRSTVWY"
        acid = list(acid)
        counter = dict(Counter(chars).items())
        for i in acid:
            try:
                list_aac.append(counter[i] / len(chars))
            except:
                list_aac.append(0)
        aac_feature = np.reshape(np.array(list_aac), (1, -1))
        return aac_feature.flatten()

    def get_single_feature(self, seq):
        '''
        :param seq: 输入序列
        :return: theta特征
        '''
        encoded_seq, mean, std = self.__encode_sequence(seq)
        aac_feature = self.__get_AAc(seq)
        length = len(seq)
        L = int(length / 3)
        theta = []
        x = []
        for j in range(self.k):
            theta_u = []
            for i in range(encoded_seq.shape[1] - j - 1):
                A_i = encoded_seq[:, i]
                A_j = encoded_seq[:, i+j+1]
                H_j = (A_j - mean) / std
                H_i = (A_i - mean) / std
                x_i = np.sum(np.power(H_j - H_i, 2)) / mean.shape[0]
                theta_u.append(x_i)
            theta.append(sum(x) / L - j - 1)
        for u in range(20):
            x_u = aac_feature[u] / (np.sum(aac_feature) + self.w * sum(theta))
            x.append(x_u)
        for u in range(self.k):
            x_u = theta[u] / (np.sum(aac_feature) + self.w * sum(theta))
            x.append(x_u)
        return x

    def get_features(self, sequences):
        '''
        :param sequences: 序列， DataFrame格式，序列的列名必须为"seq"
        :return: 特征矩阵
        '''
        columns_name = sequences.columns.tolist()
        index_seq = columns_name.index("seq")
        fetures = None
        for i in range(len(sequences)):
            print(i)
            single_sequence = sequences.iloc[i, index_seq]
            single_feature = self.get_single_feature(single_sequence)
            single_feature = np.reshape(single_feature, (1, -1))
            if i == 0:
                fetures = single_feature
            else:
                fetures = np.r_[fetures, single_feature]
        np.save("features.npy", fetures)
        return fetures


if __name__ == "__main__":
    data = pd.read_excel("data.xlsx")
    aa = PseAAC(k=5, w=0.05)
    aa.get_features(data)