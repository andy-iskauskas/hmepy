import unittest
import pandas as pd
import numpy as np
from hmepy import *

mat1 = pd.DataFrame(data = {'a': [1.9, 2.1, 3.4], 'b': [0.1, -0.1, 0.4]})
mat2 = pd.DataFrame(data = {'a': [1.8, 2.4, 3.2], 'b': [0.5, 0, -0.5]})

class testExpSq(unittest.TestCase):
    def test_1d_equal(self):
        tmat = pd.DataFrame(data = {'a': [1]})
        calc = expSq(tmat, tmat, {'theta': 0.1})
        self.assertEqual(calc[0,0], 1)
    def test_2d_equal(self):
        tmat = pd.DataFrame(data = {'a': [1,2,3], 'b': [0.1, 0.4, 0.3]})
        calc = expSq(tmat, tmat, {'theta': 0.2})
        self.assertTrue(all(np.diag(calc) == 1))
    def test_1d(self):
        d1 = pd.DataFrame(data = {'x': [1]})
        d2 = pd.DataFrame(data = {'x': [2]})
        calc = expSq(d1, d2, {'theta': 0.1})
        self.assertAlmostEqual(calc[0,0], 3.720076e-44, 7)
    def test_1d_multi(self):
        d1 = pd.DataFrame(data = {'a': [1,2]})
        d2 = pd.DataFrame(data = {'a': [1.1, 2.9]})
        calc = expSq(d1, d2, {'theta': 0.4})
        target = np.transpose(np.array([[9.394131e-01, 1.589391e-10],[0.006329715, 0.006329715]]))
        self.assertIsNone(np.testing.assert_allclose(target, calc))
    def test_2d_multi(self):
        calc = expSq(mat1, mat2, {'theta': 1})
        target = np.transpose(np.array([
            [0.8436648, 0.7710516, 0.1287349],
            [0.6376282, 0.9048374, 0.2541070],
            [0.07653555, 0.31348618, 0.42741493]
        ]))
        self.assertIsNone(np.testing.assert_allclose(target, calc, rtol = 1e-5))
    def test_both_bigger_than_one(self):
        d2 = pd.DataFrame(data = {'a': [1.8, 2.4], 'b': [0.5, 0]})
        calc = expSq(mat1, d2, {'theta': 1})
        self.assertEqual(np.shape(calc)[0], 2)
        self.assertEqual(np.shape(calc)[1], 3)

    def test_single_column_or_row(self):
        d1 = pd.DataFrame(data = {'a': [1.8], 'b': [0.5]})
        self.assertEqual(np.shape(expSq(d1, mat1, {'theta': 1}))[1], 1)
        self.assertEqual(np.shape(expSq(mat1, d1, {'theta': 1}))[0], 1)
    def test_different_theta(self):
        thet = {'theta': np.array([1, 0.5])}
        calc = expSq(mat1, mat2, thet)
        target = np.array([
            [0.522046, 0.216536, 0.074274],
            [0.748264, 0.878095, 0.19398],
            [0.043718, 0.157237, 0.037628]
        ])
        self.assertIsNone(np.testing.assert_allclose(target, calc, rtol = 1e-5))
    def test_same_points(self):
        calc = expSq(mat1, mat1, {'theta': 0.2})
        self.assertIsNone(np.testing.assert_array_equal(calc, calc.T))
    def test_no_theta(self):
        with self.assertRaises(ValueError):
            expSq(mat1, mat1)
    def test_no_df(self):
        with self.assertRaises(TypeError):
            expSq(mat1, hp = {'theta': 0.1})

class testExpSqDiff(unittest.TestCase):
    def test_self_zero(self):
        mat = pd.DataFrame(data = {'a': [1.9, 2.1, 3.4], 'b': [0.1, -0.1, 0.4]})
        calc = expSqDiff(mat, mat, {'theta': 1}, 'a')
        self.assertTrue(all(np.diag(calc) == 0))
    def test_one_dim(self):
        m1 = pd.DataFrame(data = {'a': [1]})
        m2 = pd.DataFrame(data = {'a': [2]})
        calc = expSqDiff(m1, m2, {'theta': 0.1}, 'a')
        self.assertAlmostEqual(calc[0,0], 7.44015195e-42)
    def test_three_dim(self):
        m1 = pd.DataFrame(data = {'a': [1], 'b': [2], 'c': [-1]})
        m2 = pd.DataFrame(data = {'a': [1.5], 'b': [2.9], 'c': [0.7]})
        calc = expSqDiff(m1, m2, {'theta': 0.2}, 'c')
        self.assertAlmostEqual(calc[0,0], 4.899197e-12)
    def test_two_dim_two_deriv(self):
        calc = expSqDiff(mat1, mat2, {'theta': 1}, 'a', 'b')
        result = np.array([
            [0.1349864, 0.4590923, 0.04898275],
            [0.1542103, -0.1085805, -0.50157789],
            [0.4016529, 0.4472282, -0.30773875]
        ])
        self.assertIsNone(np.testing.assert_allclose(result, calc, rtol = 1e-6))
    def test_fail_no_deriv(self):
        with self.assertRaises(TypeError):
            expSqDiff(mat1, mat2, {'theta': 0.1})

class testMatern(unittest.TestCase):
    def test_matern_basic(self):
        calc = matern(pd.DataFrame(data = {'a': [1]}), pd.DataFrame(data = {'a': [1]}), {'nu': 1.5, 'theta': 0.1})
        self.assertEqual(calc[0,0], 1)
    def test_matern_basic_2d(self):
        mat = pd.DataFrame(data = {'a': [1,2,3], 'b': [0.1, 0.4, 0.3]})
        calc = matern(mat, mat, {'theta': 0.2, 'nu': 1.5})
        self.assertTrue(all(np.diag(calc) == 1))
    def test_matern_1d(self):
        calc = matern(pd.DataFrame(data = {'a': [1,2]}), pd.DataFrame(data = {'a': [1.1, 2.9]}), {'theta': 0.4, 'nu': 2.5})
        expect = np.array([
            [0.9509599, 0.09449877],
            [0.0012006, 0.09449877]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc, rtol = 1e-4))
    def test_matern_3d_single(self):
        v1 = pd.DataFrame(data = {'a': [1], 'b': [2], 'c': [-1]})
        v2 = pd.DataFrame(data = {'a': [1.5], 'b': [2.9], 'c': [-0.7]})
        calc = matern(v1, v2, {'theta': 0.2, 'nu': 0.5})
        self.assertAlmostEqual(calc[0,0], 0.0046919704)
    def test_matern_2d_multi(self):
        calc = matern(mat1, mat2, {'theta': 1, 'nu': 1.5})
        expect = np.array([
            [0.8392642, 0.6764411, 0.2350773],
            [0.7786323, 0.8949942, 0.4436402],
            [0.2914432, 0.3986634, 0.5259420]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc, rtol = 1e-6))
    def test_dim_one(self):
        d1 = pd.DataFrame(data = {'a': [1.8, 2.4], 'b': [0.5, 0]})
        calc = matern(mat1, d1, {'theta': 1, 'nu': 1.5})
        self.assertEqual(np.shape(calc), (2,3))
    def test_dim_two(self):
        d1 = pd.DataFrame(data = {'a': [1.8], 'b': [0.5]})
        thishp = {'theta': 1, 'nu': 1.5}
        calc1 = matern(mat1, d1, thishp)
        calc2 = matern(d1, mat1, thishp)
        self.assertEqual(np.shape(calc1), (1,3))
        self.assertEqual(np.shape(calc2), (3,1))
        self.assertIsNone(np.testing.assert_equal(calc1, calc2.T))
    def test_symmetry(self):
        calc = matern(mat1, mat1, {'theta': 0.2, 'nu': 1.5})
        self.assertIsNone(np.testing.assert_equal(calc, calc.T))
    def test_no_hp(self):
        with self.assertRaises(ValueError):
            matern(mat1, mat1)
        with self.assertRaises(ValueError):
            matern(mat1, mat1, {'nu': 0.5})
        with self.assertRaises(ValueError):
            matern(mat1, mat1, {'theta': 1})
    def test_bad_nu(self):
        with self.assertRaises(ArithmeticError):
            matern(mat1, mat1, {'theta': 0.1, 'nu': 1})

class testMaternDiff(unittest.TestCase):
    def test_self_corr_zero(self):
        df = pd.DataFrame(data = {'a': [1]})
        calc = maternDiff(df, df, {'theta': 0.1, 'nu': 1.5}, 'a')
        self.assertIsNone(np.testing.assert_equal(calc, np.zeros((1,1))))
    def test_self_corr_zero_3d(self):
        df = pd.DataFrame(data = {'a': [1,2,3], 'b': [0.1, 0.4, 0.3]})
        calc = maternDiff(df, df, {'theta': 0.1, 'nu': 1.5}, 'a')
        self.assertIsNone(np.testing.assert_equal(np.diag(calc), np.zeros(3)))
    def test_1d_1point(self):
        df1 = pd.DataFrame(data = {'a': [1]})
        df2 = pd.DataFrame(data = {'a': [2]})
        calc = maternDiff(df1, df2, {'theta': 0.1, 'nu': 1.5}, 'a')
        self.assertAlmostEqual(calc[0,0], -9.01405e-6)
    def test_1d_multi(self):
        df1 = pd.DataFrame(data = {'a': [1,2]})
        df2 = pd.DataFrame(data = {'a': [1.1,2.9]})
        hp = {'theta': 0.4, 'nu': 2.5}
        calc = maternDiff(df1, df2, hp, 'a')
        expect = np.array([
            [-0.928542145, 0.3692918],
            [-0.005609912, -0.3692918]
        ])
        self.assertIsNone(np.testing.assert_allclose(calc, expect))
    def test_3d_single(self):
        df1 = pd.DataFrame(data = {'a': [1], 'b': [2], 'c': [-1]})
        df2 = pd.DataFrame(data = {'a': [1.5], 'b': [2.9], 'c': [-0.7]})
        calc = maternDiff(df1, df2, {'theta': 0.2, 'nu': 1.5}, 'c')
        self.assertAlmostEqual(calc[0,0], -0.00208377)
    def test_3d_multi(self):
        calc = maternDiff(mat1, mat2, {'theta': 1, 'nu': 2.5}, 'a', 'b')
        expect = np.array([
            [0.132580306, 0.334695240, 0.036993702],
            [0.133234551, -0.123267173, -0.2998889029],
            [0.264540759, 0.267678812, -0.190884317]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc, rtol = 1e-5))
    def test_fail_no_dir(self):
        with self.assertRaises(TypeError):
            maternDiff(mat1, mat2, {'theta': 0.1, 'nu': 2.5})
    def test_fail_not_diff(self):
        with self.assertRaises(ArithmeticError):
            maternDiff(mat1, mat1, {'theta': 0.1, 'nu': 0.5}, 'a')
            maternDiff(mat1, mat1, {'theta': 0.1, 'nu': 1.5}, 'a', 'b')

class testOrn(unittest.TestCase):
    def test_self_corr(self):
        df = pd.DataFrame(data = {'a': [1]})
        calc = ornUhl(df, df, {'theta':0.2})
        self.assertEqual(calc[0,0], 1)
    def test_self_multi(self):
        df = pd.DataFrame(data = {'a': [1,2,3], 'b': [0.1, 0.4, 0.3]})
        calc = ornUhl(df, df, {'theta': 0.2})
        self.assertIsNone(np.testing.assert_equal(np.ones(3), np.diag(calc)))
    def test_one_point_1d(self):
        df1 = pd.DataFrame(data = {'a': [1]})
        df2 = pd.DataFrame(data = {'a': [2]})
        calc = ornUhl(df1, df2, {'theta': 0.1})
        self.assertAlmostEqual(calc[0,0], 4.539993e-5)
    def test_two_point_1d(self):
        df1 = pd.DataFrame(data = {'a': [1,2]})
        df2 = pd.DataFrame(data = {'a': [1.1, 2.9]})
        calc = ornUhl(df1, df2, {'theta': 0.4})
        target = np.array([
            [0.778800783, 0.1053992],
            [0.008651695, 0.1053992]
        ])
        self.assertIsNone(np.testing.assert_allclose(calc, target, rtol = 1e-6))
    def test_one_point_3d(self):
        df1 = pd.DataFrame(data = {'a': [1], 'b': [2], 'c': [-1]})
        df2 = pd.DataFrame(data = {'a': [1.5], 'b': [2.9], 'c': [-0.7]})
        calc = ornUhl(df1, df2, {'theta': 0.2})
        self.assertAlmostEqual(calc[0,0], 0.00469197)
    def test_multi_point_3d(self):
        calc = ornUhl(mat1, mat2, {'theta': 1})
        expect = np.array([
            [0.6621186, 0.5112889, 0.2012672],
            [0.6005545, 0.7288934, 0.3406046],
            [0.2388828, 0.3102211, 0.3977409]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc, rtol = 1e-6))
    def test_dim_check(self):
        df1 = pd.DataFrame(data = {'a': [1.8], 'b': [0.5]})
        self.assertEqual(np.shape(ornUhl(mat1, mat2.iloc[[1,2],:], {'theta': 1})), (2, 3))
        self.assertEqual(np.shape(ornUhl(df1, mat1, {'theta': 1})), (3, 1))
        self.assertEqual(np.shape(ornUhl(mat1, df1, {'theta': 1})), (1, 3))
    def test_symm(self):
        calc = ornUhl(mat1, mat1, {'theta': 1})
        self.assertIsNone(np.testing.assert_equal(calc, calc.T))
    def test_fail_no_param(self):
        with self.assertRaises(ValueError):
            ornUhl(mat1, mat2)
        with self.assertRaises(ValueError):
            ornUhl(mat1, mat2, {'nu': 0.5})
    def test_match_matern(self):
        df1 = pd.DataFrame(data = {'a': np.random.uniform(0, 1, 10),
                                   'b': np.random.uniform(-1, 1, 10)})
        df2 = pd.DataFrame(data = {'a': np.random.uniform(0, 1, 5),
                                   'b': np.random.uniform(-1, 1, 5)})
        self.assertIsNone(np.testing.assert_allclose(
            matern(df1, df2, {'theta': 0.2, 'nu': 0.5}),
            ornUhl(df1, df2, {'theta': 0.2})
        ))

class testGamma(unittest.TestCase):
    def test_self_corr(self):
        df = pd.DataFrame(data = {'a': [1]})
        self.assertEqual(gammaExp(df, df, {'theta': 0.1, 'gamma': 1})[0,0], 1)
    def test_self_multi(self):
        df = pd.DataFrame(data = {'a': [1,2,3], 'b': [0.1, 0.4, 0.3]})
        self.assertIsNone(np.testing.assert_equal(
            np.diag(gammaExp(df, df, {'theta': 0.2, 'gamma': 1.5})),
            np.ones(3)
        ))
    def test_single_1d(self):
        df1 = pd.DataFrame(data = {'a': [1]})
        df2 = pd.DataFrame(data = {'a': [2]})
        calc = gammaExp(df1, df2, {'theta': 0.1, 'gamma': 1.2})
        self.assertAlmostEqual(calc[0,0], 1.3088694e-7)
    def test_multi_1d(self):
        df1 = pd.DataFrame(data = {'a': [1,2]})
        df2 = pd.DataFrame(data = {'a': [1.1, 2.9]})
        calc = gammaExp(df1, df2, {'theta': 0.4, 'gamma': 0.6})
        expect = np.array([
            [0.6470865, 0.196575705],
            [0.07832213, 0.196575705]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc))
    def test_single_3d(self):
        df1 = pd.DataFrame(data = {'a': [1], 'b': [2], 'c': [-1]})
        df2 = pd.DataFrame(data = {'a': [1.5], 'b': [2.9], 'c': [-0.7]})
        calc = gammaExp(df1, df2, {'theta': 0.2, 'gamma': 1.1})
        self.assertAlmostEqual(calc[0,0], 0.0017601453)
    def test_multi_3d(self):
        calc = gammaExp(mat1, mat2, {'theta': 1, 'gamma': 0.7})
        expect = np.array([
            [0.5840054, 0.4694570, 0.24870724],
            [0.5357538, 0.6397463, 0.34877791],
            [0.2764781, 0.3274293, 0.38879390]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc))
    def test_dim_check(self):
        df1 = pd.DataFrame(data = {'a': [1.8], 'b': [0.5]})
        self.assertEqual(np.shape(gammaExp(mat1, mat2.iloc[[1,2],:], {'theta': 1, 'gamma': 1.4})), (2, 3))
        self.assertEqual(np.shape(gammaExp(df1, mat1, {'theta': 1, 'gamma': 1.4})), (3, 1))
        self.assertEqual(np.shape(gammaExp(mat1, df1, {'theta': 1, 'gamma': 1.4})), (1, 3))
    def test_symm(self):
        calc = gammaExp(mat1, mat1, {'theta': 1, 'gamma': 0.5})
        self.assertIsNone(np.testing.assert_equal(calc, calc.T))
    def test_fail_param(self):
        with self.assertRaises(ValueError):
            gammaExp(mat1, mat1)
        with self.assertRaises(ValueError):
            gammaExp(mat1, mat1, {'theta': 0.1})
        with self.assertRaises(ValueError):
            gammaExp(mat1, mat1, {'gamma': 0.4})
    def test_fail_gamma_misspec(self):
        with self.assertRaises(ArithmeticError):
            gammaExp(mat1, mat1, {'theta': 0.1, 'gamma': 0})
        with self.assertRaises(ArithmeticError):
            gammaExp(mat1, mat1, {'theta': 0.1, 'gamma': 2.01})

class testRatQuad(unittest.TestCase):
    def test_self_corr(self):
        df = pd.DataFrame(data = {'a': [1]})
        self.assertEqual(ratQuad(df, df, {'theta': 0.1, 'alpha': 1.5})[0,0], 1)
    def test_self_multi(self):
        df = pd.DataFrame(data = {'a': [1,2,3], 'b': [0.1, 0.4, 0.3]})
        self.assertIsNone(np.testing.assert_equal(
            np.diag(ratQuad(df, df, {'theta': 0.2, 'alpha': 1.5})),
            np.ones(3)
        ))
    def test_single_1d(self):
        df1 = pd.DataFrame(data = {'a': [1]})
        df2 = pd.DataFrame(data = {'a': [2]})
        calc = ratQuad(df1, df2, {'theta': 0.1, 'alpha': 2.5})
        self.assertAlmostEqual(calc[0,0], 0.00049482515)
    def test_multi_1d(self):
        df1 = pd.DataFrame(data = {'a': [1,2]})
        df2 = pd.DataFrame(data = {'a': [1.1, 2.9]})
        calc = ratQuad(df1, df2, {'theta': 0.4, 'alpha': 2.5})
        expect = np.array([
            [0.96942099, 0.1740445],
            [0.01401614, 0.1740445]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc, rtol = 1e-6))
    def test_single_3d(self):
        df1 = pd.DataFrame(data = {'a': [1], 'b': [2], 'c': [-1]})
        df2 = pd.DataFrame(data = {'a': [1.5], 'b': [2.9], 'c': [-0.7]})
        calc = ratQuad(df1, df2, {'theta': 0.2, 'alpha': 0.5})
        self.assertAlmostEqual(calc[0,0], 0.1833397)
    def test_multi_3d(self):
        calc = ratQuad(mat1, mat2, {'theta': 1, 'alpha': 1.5})
        expect = np.array([
            [0.9206467, 0.8108737, 0.3952748],
            [0.8827861, 0.9520052, 0.6124095],
            [0.4578728, 0.5688002, 0.6878453]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc))
    def test_dim_check(self):
        df1 = pd.DataFrame(data = {'a': [1.8], 'b': [0.5]})
        self.assertEqual(np.shape(ratQuad(mat1, mat2.iloc[[1,2],:], {'theta': 1, 'alpha': 1.4})), (2, 3))
        self.assertEqual(np.shape(ratQuad(df1, mat1, {'theta': 1, 'alpha': 1.4})), (3, 1))
        self.assertEqual(np.shape(ratQuad(mat1, df1, {'theta': 1, 'alpha': 1.4})), (1, 3))
    def test_symm(self):
        calc = ratQuad(mat1, mat1, {'theta': 1, 'alpha': 1.1})
        self.assertIsNone(np.testing.assert_equal(calc, calc.T))
    def test_fail_param(self):
        with self.assertRaises(ValueError):
            ratQuad(mat1, mat1)
        with self.assertRaises(ValueError):
            ratQuad(mat1, mat1, {'theta': 0.1})
        with self.assertRaises(ValueError):
            ratQuad(mat1, mat1, {'alpha': 0.4})

class testRatQuadDiff(unittest.TestCase):
    def test_self_corr_zero_1d(self):
        df = pd.DataFrame(data = {'a': [1]})
        self.assertEqual(ratQuadDiff(df, df, {'theta': 0.2, 'alpha': 1.1}, 'a')[0,0], 0)
    def test_self_corr_zero_3d(self):
        df = pd.DataFrame(data = {'a': [1,2,3], 'b': [0.1, 0.4, 0.3]})
        calc = ratQuadDiff(df, df, {'theta': 0.2, 'alpha': 1.5}, 'b')
        self.assertIsNone(np.testing.assert_equal(np.zeros(3), np.diag(calc)))
    def test_multi_1d(self):
        df1 = pd.DataFrame(data = {'a': [1,2]})
        df2 = pd.DataFrame(data = {'a': [1.1, 2.9]})
        calc = ratQuadDiff(df1, df2, {'theta': 0.4, 'alpha': 2.5}, 'a')
        expect = np.array([
            [-0.5984080, 0.4864598],
            [-0.0301935, -0.4864598]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc, rtol = 1e-6))
    def test_single_3d(self):
        df1 = pd.DataFrame(data = {'a': [1], 'b': [2], 'c': [-1]})
        df2 = pd.DataFrame(data = {'a': [1.5], 'b': [2.9], 'c': [-0.7]})
        calc = ratQuadDiff(df1, df2, {'theta': 0.2, 'alpha': 1.5}, 'c')
        self.assertAlmostEqual(calc[0,0], -0.0205828295)
    def test_multi_3d(self):
        calc = ratQuadDiff(mat1, mat2, {'theta': 1, 'alpha': 2.5}, 'a', 'b')
        expect = np.array([
            [0.04817765, 0.17099417, 0.03464827],
            [0.05572207, -0.03841922, -0.21899785],
            [0.23266799, 0.20716591, -0.12432663]
        ])
        self.assertIsNone(np.testing.assert_allclose(expect, calc))
    def test_no_deriv_fails(self):
        with self.assertRaises(TypeError):
            ratQuadDiff(mat1, mat1)

if __name__ == '__main__':
    unittest.main()
