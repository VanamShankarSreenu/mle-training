from mle_training import ingest_data,score,train
import pandas as pd
import pytest
import pickle
import os
#test data is split into test and train and saved
def test_ingest_data():
    obj = ingest_data.ingest_data()
    obj.stratifiedsplit()
    obj.transform_train_test()
    dir = obj.args.ingest_data_path
    #df = pd.read_csv('/data/raw/housing.csv')
    df_train = pd.read_csv(dir+'/train.csv')
    df_test = pd.read_csv(dir+'/test.csv')
    #assert df.empty is False
    assert df_train.empty is False
    assert df_test.empty is False

def test_traing():
    obj = train.Model()
    obj.Linear_Model_train()
    assert obj.saved is 1
    obj.DesTree_Model_Train()
    assert obj.saved is 1
    obj.RanFor_Model_Train()
    assert obj.saved is 1
    path = obj.args.output_model_path
    models = []
    for filename in os.scandir(path):
        models.append(filename)
        with open(filename , 'rb') as f:
            pickle.load(f)

def test_score():
    obj = score.scores()
    obj.results()
    assert obj.rmse is not None
    assert obj.r2 is not None
