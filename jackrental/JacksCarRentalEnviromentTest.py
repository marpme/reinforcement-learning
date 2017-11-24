import unittest
import jackrental.JacksCarRentalEnviroment as env
import numpy as np

class TestJacksCarRentalEnvironment(unittest.TestCase):
    # there are a lot of magic numbers in this test
    # these numbers depending on the env parameters!

    def setUp(self):
        self.env = env.JacksCarRentalEnvironment()

    def _avg_mean(self, cars, action, t_mean, max_diff=1.):
        env = self.env
        n = 10000
        r_mean = 0
        for i in range(n):
            env.reset(cars)
            _, r, _, _ = env.step(action)
            r_mean += r
        r_mean = r_mean / n
        diff = np.abs(r_mean - t_mean)
        self.assertTrue(diff < max_diff)

    def test_rent0(self):
        self._avg_mean(20, 0, 70)

    def test_rent1(self):
        self._avg_mean(10, 0, 70)

    def test_nightly_moves(self):
        env = self.env
        env.reset(0)
        _, r, _, _ = env.step(5)
        self.assertTrue(r == -10.)

    def _avg_nb_cars(self, a_desired, b_desired,
                     cars, action):

        env = self.env
        n = 10000
        a_mean = 0
        b_mean = 0
        for i in range(n):
            env.reset(cars)
            ab, _, _, _ = env.step(action)
            a_mean += ab[0]
            b_mean += ab[1]
        a_diff = np.abs((a_mean / n) - a_desired)
        b_diff = np.abs((b_mean / n) - b_desired)
        self.assertTrue(a_diff < 0.5)
        self.assertTrue(b_diff < 0.5)

    def test_nb_cars_10_0(self):
        a_desired = 10
        b_desired = 8
        cars = 10
        action = 0
        self._avg_nb_cars(a_desired, b_desired,
                          cars, action)

    def test_nb_cars_10_3(self):
        a_desired = 7
        b_desired = 11
        cars = 10
        action = 3
        self._avg_nb_cars(a_desired, b_desired,
                          cars, action)

    def test_nb_cars_10_m2(self):
        a_desired = 12
        b_desired = 6
        cars = 10
        action = -2
        self._avg_nb_cars(a_desired, b_desired,
                          cars, action)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
