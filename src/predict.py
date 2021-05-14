from sklearn import preprocessing
import pandas as pd
import os
import numpy as np
from sklearn import  ensemble
from sklearn import  metrics
import joblib

from . import dispatcher



TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get("MODEL")

#print(dispatcher.MODELS[MODEL])

#FOLD = 1
#TRAINING_DATA = '/Users/sushantverma/PycharmProjects/abhishek_thakur/input/train_folds.csv'



def predict():

    df = pd.read_csv(TEST_DATA)
    test_idx  = df['id'].values
    predictions = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join(f'models/{MODEL}_{FOLD}_label_encoder.pkl'))
        cols = joblib.load(os.path.join(f'models/{MODEL}_{FOLD}_columns.pkl'))
        for c in cols:
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist())

        #data is ready to train

        #clf = ensemble.RandomForestClassifier(n_estimators=1000 ,n_jobs=-1,verbose=2)
        clf = joblib.load(os.path.join(f'models/{MODEL}_{FOLD}.pkl'))
        df = df[cols]

        preds = clf.predict_proba(df)[:,1]

        if FOLD ==0:
            predictions = preds
        else:
            predictions += preds

    predictions = predictions/5
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    sub.id = sub.id.astype(int)
    return sub

if __name__ == '__main__':
    submission = predict()
    submission.to_csv(f'models/{MODEL}.csv',index =False)