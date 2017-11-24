import numpy as np

class CarRental:
    def __init__(self, req_mean, ret_mean, startCars):
        self.sell_income = 10.
        self.min_cars = 1
        self.current_cars = startCars
        self.req_mean = req_mean
        self.ret_mean = ret_mean

    def nightlyMove(self):
        returned_cars = self.getReturnedCars()
        requested_cars = self.getRequestedCars()

        self.addCars(returned_cars - requested_cars)

        return requested_cars * self.sell_income

    def getCurrentState(self):
        return self.current_cars

    def addCars(self, cars):
        if self.current_cars + cars > 20:
            self.current_cars = 20
        else:
            self.current_cars += cars

    def getReturnedCars(self):
        return np.random.poisson(self.ret_mean)

    def getRequestedCars(self):
        return np.random.poisson(self.req_mean)

    def carRentalIsDead(self):
        return self.current_cars < self.min_cars

    def s(self, amount):
        self.current_cars += amount
        return self.current_cars

    def transact_cars(self, amount):
        self.current_cars += amount
