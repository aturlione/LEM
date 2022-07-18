import unittest
import Model.LEM as LEM 


class TestPythonModule(unittest.TestCase):
    """Example test"""

    def test_LEM_simulation(self):
        
        inputs = {'section':'subcatchment/1',
                  'initial date': '21-10-01 00:00:00',
                  'final date': '21-10-10 00:00:00'}
        param = {"id":1, "Nombre": "Abusu"}
        response = LEM.LEM().LEM_simulation(inputs,param)
        
        print(response)
        
        self.assertIsNotNone(response)
        #self.assertEqual(type(response), dict)
        


if __name__ == '__main__':
    unittest.main()