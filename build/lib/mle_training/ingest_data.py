from ast import arg
from gettext import install
import os,tarfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import argparse
import csv
from os.path import dirname as up
import logging
import argparse

class ingest_data:
    def __init__(self):
        self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        self.HOUSING_PATH =  'data/raw/'
        self.HOUSING_URL = self.DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
        self.strat_train_set =  None
        self.strat_test_set = None
        self.housing =  None
        self.args = self.parse_args()
        self.logger = self.logconfig(self.args)
        self.logger.debug('initialised the variables in ingest_data')

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--ingest_data_path", type = str,default="data/processed/",help="specify the output folder/file path")
        parser.add_argument("--log_level", type = str,default='DEBUG',help="specifiy level of debug")
        parser.add_argument("--log_path", type = str,help="specify the path where to save log file")
        parser.add_argument("--no_console_log", type = str,default=None,help="specify to log on console or not")
        args = parser.parse_args()
        self.args = args
        return args

    def logconfig(self,args):
        #getting logger object
        logger = logging.getLogger(__name__)
        #specifing the format we want to log
        formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(funcName)s - %(lineno)d - %(message)s')
        #default setting to Debug level
        level = logging.DEBUG
        logger.setLevel(level)

        if args.log_level:
            if args.log_level == 'DEBUG':
                level=logging.DEBUG
            
            elif args.log_level == 'INFO':
                level=logging.INFO
                
            elif args.log_level == 'ERROR':
                level=logging.ERROR
            
            elif args.log_level == 'WARNING':
                level=logging.WARNING
            else:
                level=logging.CRITICAL
            logger.setLevel(level)

        if args.log_path:
            file_handler = logging.FileHandler(args.log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if not args.no_console_log:
            stream_handler = logging.StreamHandler()
            #set the console format
            stream_handler.setFormatter(formatter)
            #add console handler to logger
            logger.addHandler(stream_handler)
        
        logger.debug('INSIDE LOG')
        return logger


    def fetch_housing_data(self):
        os.makedirs('data/raw/', exist_ok=True)
        tgz_path = os.path.join('data/raw/', "housing.tgz")
        urllib.request.urlretrieve(self.HOUSING_URL, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path='data/raw/')
        self.logger.debug('extracted dataset to  path [%s] ',tgz_path)
        housing_tgz.close()


    def load_housing_data(self,housing_path):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)
    

    def load_data(self):
        self.fetch_housing_data()
        housing = self.load_housing_data(self.HOUSING_PATH)
        self.housing=housing
        self.logger.debug('Data is loaded')
        return housing
    
    def stratifiedsplit(self):
        housing  = self.load_data()
        #stratified split
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        #remove INCOME CAT column from test and train
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set
        self.logger.debug('Applied Stratified spliting')

    def DataTransformation(self,housing,data):
        #storring shape
        shape_before  = housing.shape
        self.logger.debug('%s data Applying Data transformation',data)
        #pipeline fill null values,encode catagorical data
        housing_labels = housing["median_house_value"].copy()
        housing = housing.drop("median_house_value", axis=1)
        housing_num = housing.drop("ocean_proximity",axis=1)
        housing_cat = housing[["ocean_proximity"]]
        self.logger.debug('%s Data passing through pipelines ',data)
        num_pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy="median")),
            ('std_scaler',StandardScaler())
        ])
        num_attribs = list(housing_num.columns)
        cat_attribs = ["ocean_proximity"]

        full_pipeline = ColumnTransformer([
            ('num',num_pipeline,num_attribs),
            ("cat",OneHotEncoder(),cat_attribs)
        ])
        housing_prepared = full_pipeline.fit_transform(housing)
        self.logger.debug('%s Data is prepared',data)
        housing_prepared = np.c_[housing_prepared,housing_labels]
        cat_encoder = full_pipeline.named_transformers_["cat"]
        cat_1hot_attrib = list(cat_encoder.categories_[0])
        att = num_attribs+cat_1hot_attrib+['median_house_value']
        df = pd.DataFrame(data=housing_prepared,columns=att)
        #shape after 
        shape_after = df.shape
        if shape_after !=shape_before:
            self.logger.warning('shape of dataset is altered')
        return df


    def transform_train_test(self):
        self.df_train = self.DataTransformation(self.strat_train_set,"train")
        self.df_test =  self.DataTransformation(self.strat_test_set,"test")
        #get path of folder from command line argumentsSS
        parser = argparse.ArgumentParser()
        parser.add_argument("--ingest_data_path", type = str,help="the output folder/file path")
        args = parser.parse_args()
        # if args are not empty save in specified
        if args.ingest_data_path:
            self.logger.debug('user provide path to save files')
            os.makedirs(args.ingest_data_path, exist_ok=True)
            self.df_train.to_csv(args.ingest_data_path+'/train.csv')
            self.df_test.to_csv(args.ingest_data_path+'/test.csv')
            self.logger.debug('saving at path %s',args.ingest_data_path+'/train.csv')
            self.logger.debug('saving at path %s',args.ingest_data_path+'/test.csv')
        # save in default
        else:
            self.logger.debug('user didnt provide path to save files')
            #dir = up(os.path.realpath(os.getcwdb()))
            #dir = dir.decode('utf-8')
            dir = self.args.ingest_data_path
            self.df_train.to_csv(dir+'/train.csv')
            self.df_test.to_csv(dir+'/test.csv')
            self.logger.debug('saving at default path %s',dir+'/train.csv')
            self.logger.debug('saving at default path %s',dir+'/test.csv')
            

obj = ingest_data()
obj.stratifiedsplit()
obj.transform_train_test()