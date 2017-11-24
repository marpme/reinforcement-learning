import jackrental.CarRental as env
import numpy as np


class JacksCarRentalEnvironment:
    def __init__(self):
        self.t = 1
        self.cost_car_exchange = 2.
        self.RENTAL_INCOME = 10
        self.TRANSFER_COST = 2.
        self.done = False
        self.car_rental_a = env.CarRental(3, 3, np.random.randint(10))
        self.car_rental_b = env.CarRental(4, 2, np.random.randint(10))
        self.REQUEST_RATE = [3, 4]
        self.RETURN_RATE = [4, 2]
        self.MAX_CAPACITY = 20

    def reset(self, numbers=None):
        self.done = False
        self.t = 1
        if numbers == None:
            self.car_rental_a = env.CarRental(3, 3, np.random.randint(10))
            self.car_rental_b = env.CarRental(4, 2, np.random.randint(10))
        else:
            self.car_rental_a = env.CarRental(3, 3, numbers)
            self.car_rental_b = env.CarRental(4, 2, numbers)

        return ([self.car_rental_a.getCurrentState(), self.car_rental_b.getCurrentState()])

    def step(self, action):
        reward = 0

        self.exchangeCars(action)
        reward -= action * self.cost_car_exchange

        if self.car_rental_a.carRentalIsDead() or self.car_rental_b.carRentalIsDead():
            self.done = True

        if self.done == False:
            reward += self.car_rental_a.nightlyMove()
            reward += self.car_rental_b.nightlyMove()

        self.t += 1
        return ([self.car_rental_a.getCurrentState(), self.car_rental_b.getCurrentState()], reward, self.done, '')

    def exchangeCars(self, amount):
        if amount <= 5 and amount >= -5:
            self.car_rental_a.transact_cars(-amount)
            self.car_rental_b.transact_cars(amount)
        elif amount < -5:
            self.car_rental_a.transact_cars(-(-5))
            self.car_rental_b.transact_cars(-5)
        elif amount > 5:
            self.car_rental_a.transact_cars(-5)
            self.car_rental_b.transact_cars(5)