import unittest
import Model.LEM as LEM 
import pandas as pd

class TestPythonModule(unittest.TestCase):
    """Example test"""

    def test_obtain_data(self):
        
        #inputs = {'section':'LEM/historical'}
        #param = {"id":1, "Nombre": "Abusu"}
        inputs = {'section':'subcatchment/1'}
        #response = LEM.LEM().obtain_data(inputs,param)
        response = LEM.LEM().obtain_data(inputs)
        df = pd.DataFrame(list(response.items())) 

        print(df)
        #print (pd.DataFrame(response))
        self.assertIsNotNone(response)
        #self.assertEqual(type(response), dict)
        


if __name__ == '__main__':
    unittest.main()