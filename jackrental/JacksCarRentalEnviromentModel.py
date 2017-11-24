import jackrental.JacksCarRentalEnviroment as env
import numpy as np
import scipy.stats
import logging

logger = logging.getLogger('enviroment')

class JacksCarRentalEnvironmentModel(env.JacksCarRentalEnvironment):

    def get_transition_probabilities_and_expected_reward(self, state, action):
        """
            Compute the $p(s', r\mid s,a)$
            Parameters
            ----------
            old_state: tuple of two ints
                the state (cars_at_A, cars_at_B)
            action: int
                nigthly movements of the cars as a int between -5 to 5, e.g.:
                action +3: move three cars from A to B.
                action -2: move two cars from B to A.

            Returns
            -------
            numpy array (2d - float): mapping from (new) states to probabilities
                index first dimension: cars at A
                index second dimension: cars at B
                value: probability
            float:  expected reward for the state-action pair
        """
        # assert type(action) == int
        assert np.abs(action) <= 5
        # first we move the cars in the night
        num_states_for_a_location = self.MAX_CAPACITY + 1

        state = self._nightly_moves(state, action)

        expected_reward = - self.TRANSFER_COST * np.abs(action)
        expected_reward += self._expected_reward_rent(state)

        transition_probabilities = self._rent_transition_probabilities(state)
        transition_probabilities = self._returns_transition_probabilities(transition_probabilities)
        return transition_probabilities, expected_reward

    def _nightly_moves(self, state, action):

        cars_at_A = state[0]
        cars_at_B = state[1]
        if action > 0:
            cars_moved = min(action, cars_at_A)
        else:
            cars_moved = max(action, -cars_at_B)

        cars_at_A = min(cars_at_A - cars_moved, self.MAX_CAPACITY)
        cars_at_B = min(cars_at_B + cars_moved, self.MAX_CAPACITY)
        return [cars_at_A, cars_at_B]

    def _expected_reward_rent(self, state):
        expected_reward_rent = 0.
        m = self.MAX_CAPACITY + 1
        request_mu = self.REQUEST_RATE
        for i in (0, 1):
            cars_at_loc = state[i]
            rv = scipy.stats.poisson(request_mu[i])
            rent_prob = (rv.pmf(range(m)))
            logger.debug(rent_prob)
            rent_prob[cars_at_loc] = rent_prob[cars_at_loc:].sum()
            rent_prob[cars_at_loc + 1:] = 0.
            logger.debug(rent_prob)
            expected_reward_rent += np.dot(np.arange(len(rent_prob)), rent_prob) * self.RENTAL_INCOME
        return expected_reward_rent

    def _rent_transition_probabilities(self, state):

        num_states_for_a_location = self.MAX_CAPACITY + 1
        m = 15
        n = num_states_for_a_location + 2 * m
        p_ = [np.zeros(n), np.zeros(n)]
        request_mu = self.REQUEST_RATE

        for i in (0, 1):
            rv = scipy.stats.poisson(request_mu[i])
            cars_at_loc = state[i]
            x = cars_at_loc + m + 1
            rent_prob = (rv.pmf(range(x)))
            assert state[i] - x + m + 1 == 0
            p_[i][0:cars_at_loc + m + 1] = rent_prob[::-1]
            p_[i][m] = p_[i][:m + 1].sum()
            p_[i] = p_[i][m:-m]
        return p_

    def _returns_transition_probabilities(self, state_probalility):

        num_states_for_a_location = self.MAX_CAPACITY + 1
        m = 11
        n = num_states_for_a_location + 2 * m
        returns_mu = self.RETURN_RATE
        p_ = [np.zeros(num_states_for_a_location), np.zeros(num_states_for_a_location)]
        for i in (0, 1):
            rv = scipy.stats.poisson(returns_mu[i])
            logger.debug(len(state_probalility[i]))
            for cars_at_loc in range(len(state_probalility[i])):
                p = np.zeros(n)
                logger.debug(p.shape)
                x = num_states_for_a_location - cars_at_loc + m - 1
                return_prob = (rv.pmf(range(x)))
                logger.debug(p[cars_at_loc + m:-1].shape)
                p[cars_at_loc + m:-1] = return_prob
                logger.debug(return_prob)
                p[num_states_for_a_location + m - 1] = p[num_states_for_a_location + m - 1:].sum()
                p = p[m:-m]
                logger.debug(p)
                logger.debug(p.sum())
                p_[i] += p * state_probalility[i][cars_at_loc]
        return p_
