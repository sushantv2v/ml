import pandas as pd
from sklearn import preprocessing



class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type,handle_na =False):

        """

        :param df: pandas dataframe
        :param categorical_features: list of cartegories
        :param encoding_type: binary , ohe , label
        """



        self.df = df

        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna('-99999')
        self.output_df = self.df.copy(deep=True)


    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] =  lbl
            return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c,axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f'__bin__{j}'
                self.output_df[new_col_name] = val[:,j]
            self.binary_encoders[c] = lbl
        return self.output_df


    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return  ohe.transform(self.df[self.cat_feats].values)





    def fit_transform(self):

        if self.enc_type == 'label':
            return self._label_encoding()
        elif self.enc_type =='binary':
                return self._label_binarization()
        elif self.enc_type =='ohe':
                return self._one_hot()
        else:
            raise Exception('Encoding type not understood')







if __name__ == '__main__':

    df = pd.read_csv("../input/train.csv")

    df_test = pd.read_csv("../input/test.csv")

    sample = pd.read_csv('../input/sample_submission.csv')

    train_len = len(df['id'])


    df_test['target'] = -1

    full_data = pd.concat([df,df_test])

    cols = ['bin_0','bin_1','bin_2']

    cat_feats = CategoricalFeatures(full_data,
                                    categorical_features=cols,
                                    encoding_type = 'ohe'
                                    ,handle_na=True)

    full_data_transformed = cat_feats.fit_transform()

    #print(full_data_transformed.head())

    train_df = full_data_transformed[:train_len,:]
    test_df = full_data_transformed[train_len:,:]

    print(train_df.shape)
    print(test_df)

    from sklearn import linear_model

    X = full_data_transformed[:train_len, :]
    X_test = full_data_transformed[train_len:, :]

    clf = linear_model.LogisticRegression()
    clf.fit(X, df.target.values)
    preds = clf.predict_proba(X_test)[:, 1]

    print(preds)

    sample.loc[:, "target"] = preds
    sample.to_csv("submission.csv", index=False)