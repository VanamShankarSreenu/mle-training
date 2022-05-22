from mle_training import train
import unittest

class Test_train(unittest.TestCase):
    def test_parse_args(self):
        method = train.Model()
        self.assertEqual(method.args.dataset,"data/processed/train.csv")
        self.assertEqual(method.args.output_model_path,'artifacts/models/')


    def test_linear_model(self):
        method = train.Model()
        method.Linear_Model_train()
        assert "median_house_value" not in method.X
        #checking loaded dataset is not empty
        self.assertEqual(method.df.empty,False)
        i,j =method.Y.shape
        #Y should be i*1 dims
        self.assertEqual(j,1)
        #check model is saved or not
        self.assertEqual(method.saved,1)


    def test_Decison_model(self):
        method = train.Model()
        method.DesTree_Model_Train()
        assert "median_house_value" not in method.X
        #checking loaded dataset is not empty
        self.assertEqual(method.df.empty,False)
        i,j =method.Y.shape
        #Y should be i*1 dims
        self.assertEqual(j,1)
        #check model is saved or not
        self.assertEqual(method.saved,1)


    def test_Decison_model(self):
        method = train.Model()
        method.DesTree_Model_Train()
        assert "median_house_value" not in method.X
        #checking loaded dataset is not empty
        self.assertEqual(method.df.empty,False)
        i,j =method.Y.shape
        #Y should be i*1 dims
        self.assertEqual(j,1)
        #check model is saved or not
        self.assertEqual(method.saved,1)

    
    def test_Randforest_Model(self):
        method = train.Model()
        method.RanFor_Model_Train()
        assert "median_house_value" not in method.X
        #checking loaded dataset is not empty
        self.assertEqual(method.df.empty,False)
        i,j =method.Y.shape
        #Y should be i*1 dims
        self.assertEqual(j,1)
        #check model is saved or not
        self.assertEqual(method.saved,1)



if __name__ == '__main__':
    unittest.main()