import unittest
import neuralnetworks.cactivation as act


class MyTestCase(unittest.TestCase):
    def test_act_function_pos(self):
        leaky = act.LeakyReLU()
        self.assertEqual(leaky.f(10), 10)

    def test_act_function_neg(self):
        leaky = act.LeakyReLU()
        self.assertEqual(leaky.f(-10), -1)

    def test_act_function_neg_custa(self):
        leaky = act.LeakyReLU(0.3)
        self.assertEqual(leaky.f(-10), -3)



if __name__ == '__main__':
    unittest.main()
