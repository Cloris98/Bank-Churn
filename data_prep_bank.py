import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler


class Preprocessing:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def OneHotEncoding(self, df, enc, categories):
        transformed = pd.DataFrame(enc.transform(df[categories]).toarray(), columns=enc.get_feature_names(categories))
        return pd.concat([df.reset_index(drop=True), transformed], axis=1).drop(categories, axis=1)

    def split_data(self, X, Y):
        train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=1)
        return train_x, test_x, train_y, test_y

    def under_sampling(self, train_x, train_y):
        rus = RandomUnderSampler(random_state=0)
        train_x_rus, train_y_rus = rus.fit_resample(train_x, train_y)
        print('Training sample size after underSampling:', sorted(Counter(train_y_rus).items()))
        return train_x_rus, train_y_rus

    def over_sampling(self, train_x, train_y):
        ros = RandomOverSampler(random_state=0)
        train_x_ros, train_y_ros = ros.fit_resample(train_x, train_y)
        print('Training sample size after OverSamplingL: ', sorted(Counter(train_y_ros).items()))
        return train_x_ros, train_y_ros

    def smote(self, train_x, train_y):
        smote = SMOTE(random_state=0)
        train_x_smote, train_y_smote = smote.fit_resample(train_x, train_y)
        print('Training sample size after SMOTE: ', sorted(Counter(train_y_smote).items()))
        return train_x_smote, train_y_smote

    def standardize_data(self, train_x, test_x, num_cols):
        scaler = StandardScaler()
        scaler.fit(train_x[num_cols])
        train_x[num_cols] = scaler.transform(train_x[num_cols])
        test_x[num_cols] = scaler.transform(test_x[num_cols])
        return train_x, test_x

    def process(self, mode):
        Y = self.df['Exited']

        # drop useless column
        to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
        X = self.df.drop(to_drop, axis=1)

        data = self.split_data(X, Y)
        train_x = data[0]
        train_y = data[2]
        test_x = data[1]
        test_y = data[3]
        print('Original training sample size: ', sorted(Counter(train_y).items()))

        # process categorical feature
        categories = ['Geography']
        enc_ohe = OneHotEncoder()
        enc_ohe.fit(train_x[categories])
        train_x = self.OneHotEncoding(train_x, enc_ohe, categories)
        test_x = self.OneHotEncoding(test_x, enc_ohe, categories)

        cate_2 = ['Gender']
        enc_oe = OrdinalEncoder()
        enc_oe.fit(train_x[cate_2])
        train_x[cate_2] = enc_oe.transform(train_x[cate_2])
        test_x[cate_2] = enc_oe.transform(test_x[cate_2])

        if mode == 'org':
            train_x, train_y = train_x, train_y
        if mode == 'rus':
            train_data = self.under_sampling(train_x, train_y)
            train_x, train_y = train_data[0], train_data[1]
        if mode == 'ros':
            train_data = self.over_sampling(train_x, train_y)
            train_x, train_y = train_data[0], train_data[1]
        if mode == 'smote':
            train_data = self.smote(train_x, train_y)
            train_x, train_y = train_data[0], train_data[1]
        # standardize/ normalize numerical data
        num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')]
        train_x, test_x = self.standardize_data(train_x, test_x, num_cols)[0], self.standardize_data(train_x, test_x, num_cols)[1]

        return train_x, train_y, test_x, test_y

