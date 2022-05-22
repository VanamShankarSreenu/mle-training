from pandas import read_csv
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import argparse
import logging
import os
from os.path import dirname as up

class parse_log:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset",default='data/processed/train.csv', type = str,help="the dataset  path you like to run on model")
        parser.add_argument("--output_model_path",default='artifacts/models/',type = str,help="the output path you like to save your model")
        parser.add_argument("--log_level", type = str,help="specifiy level of debug")
        parser.add_argument("--log_path", type = str,help="specify the path where to save log file")
        parser.add_argument("--no_console_log", type = str,help="specify to log on console or not")
        args = parser.parse_args()
        self.args = args
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
        self.logger = logger

class Model:
    def __init__(self):
        obj = parse_log()
        self.args = obj.args
        self.logger = obj.logger

    def Linear_Model_train(self):
       self.saved =0
       lr = LinearRegression()
       self.logger.debug('running linear regression model')
       self.df = read_csv(self.args.dataset)
       self.Y = self.df[["median_house_value"]]
       self.X = self.df.drop("median_house_value",axis=1)
       lr.fit(self.X.values,self.Y.values)
       os.makedirs(self.args.output_model_path ,exist_ok=True)
       path = self.args.output_model_path
       path = os.path.join(path, "linearmodel.pkl")
       self.logger.debug('saving at  path provided in args %s',path)
       with open(path, "wb") as file:
           pickle.dump(lr, file)
           self.saved = 1
           file.close()
       self.lr = lr
               
    def DesTree_Model_Train(self):
        self.saved = 0
        dr = DecisionTreeRegressor()
        self.logger.debug('running decision regression model')
        self.df = read_csv(self.args.dataset)
        self.Y = self.df[["median_house_value"]]
        self.X = self.df.drop("median_house_value",axis=1)
        dr.fit(self.X,self.Y)
        os.makedirs(self.args.output_model_path ,exist_ok=True)
        path = self.args.output_model_path
        path = os.path.join(path, "desmodel.pkl")
        self.logger.debug('saving at  path provided in args %s',path)
        with open(path, "wb") as file:
            pickle.dump(dr, file)
            self.saved = 1
            file.close()
        self.destree = dr

    def RanFor_Model_Train(self):
        self.saved = 0
        rg = RandomForestRegressor()
        self.logger.debug('running Random forest regression model')
        self.df = read_csv(self.args.dataset)
        self.Y = self.df[["median_house_value"]]
        self.X = self.df.drop("median_house_value",axis=1)
        rg.fit(self.X.values,self.Y.values)
        os.makedirs(self.args.output_model_path ,exist_ok=True)
        path = self.args.output_model_path
        path = os.path.join(path, "regmodel.pkl")
        self.logger.debug('saving at  path provided in args %s',path)
        with open(path, "wb") as file:
            pickle.dump(rg, file)
            self.saved = 1
            file.close()
        self.rg = rg

obj =  Model()
obj.Linear_Model_train()
obj.DesTree_Model_Train()
obj.RanFor_Model_Train()