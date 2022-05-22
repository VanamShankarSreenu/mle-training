from mle_training import ingest_data
import unittest

class Test_train(unittest.TestCase):
    def test_parse_args(self):
        method = ingest_data.ingest_data()
        #check if stored in default paths
        self.assertEqual(method.args.ingest_data_path,"data/processed/")
        self.assertEqual(method.args.log_level,'DEBUG')


    def test_load_data(self):
        #checking loaded dataset is not empty
        method = ingest_data.ingest_data()
        df = method.load_data()
        self.assertEqual(df.empty,False)


    def test_stratifiedsplit(self):
        method = ingest_data.ingest_data()
        method.stratifiedsplit()
        assert "income_cat" not in method.strat_train_set
        assert "income_cat" not in method.strat_test_set
    
    def test_DataTransformation(self):
        method = ingest_data.ingest_data()
        method.stratifiedsplit()
        method.transform_train_test()
        assert "ocean_proximity" not in method.df_train
        assert "ocean_proximity" not in method.df_test
        X1,Y1 = method.df_train.shape
        X2,Y2 = method.df_test.shape
        #TESTING FEATURES DIMESIONS
        self.assertEqual(Y1,Y2)



if __name__ == '__main__':
    unittest.main()