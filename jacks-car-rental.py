import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import scipy.stats
import logging
import itertools
import jackrental.JacksCarRentalEnviromentModel as env


def plot_policy(policy):
    MAX_CAPACITY = 20
    A = np.arange(0, MAX_CAPACITY+1)
    B = np.arange(0, MAX_CAPACITY+1)
    A, B = np.meshgrid(A, B)
    Po = policy.reshape(MAX_CAPACITY+1,-1)
    levels = range(-5,6,1)
    plt.figure(figsize=(7,6))
    CS = plt.contourf(A, B, Po, levels)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('actions')
    #plt.clabfrom mpl_toolkits.mplot3d import Axes3D



def plot3d_over_states(f, zlabel="", ):
    MAX_CAPACITY = 20
    A = np.arange(0, MAX_CAPACITY + 1)
    B = np.arange(0, MAX_CAPACITY + 1)
    # B, A !!!
    B, A = np.meshgrid(B, A)
    V = f.reshape(MAX_CAPACITY + 1, -1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(A, B, V, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.scatter(A, B, V, c='b', marker='.')
    ax.set_xlabel("cars at A")
    ax.set_ylabel("cars at B")
    ax.set_zlabel(zlabel)

    # ax.view_init(elev=10., azim=10)

    plt.show()


"""
1. Initialization 
    Value for State  Policy for State 
    V (s) ∈ R and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
    Repeat
        ∆ ← 0
        For each s ∈ S:
            v ← V (s)
            V (s) ← Bellman Eq.
            ∆ ← max(∆, |v − V (s)|)
    until ∆ < θ (a small positive number)

3. Policy Improvement
    policy-stable ← true
    For each s ∈ S:
        old-action ← π(s)
        π(s) ← ArgMax Bellman Eq.
        If old-action != π(s), then policy-stable ← f alse
    If policy-stable, then stop and return V ≈ v∗ and π ≈ π∗; else go to 2
"""

# all possible states
states = []
for i in range(0, 20 + 1):
    for j in range(0, 20 + 1):
        states.append([i, j])


def oneStepAhead(transition_probabilities, reward, Vs):
    return np.sum(transition_probabilities[0] * transition_probabilities[1] * (reward + 1 * Vs))


def policyImprovement(Vs, policy, env):
    while True:
        # Will be set to false if we make any changes to the policy
        improvePolicy = True
        newPolicy = np.zeros((21, 21), dtype=np.int)
        actions = np.arange(-5, 6)

        for s1 in range(21):
            for s2 in range(21):
                actionReturns = []
                for action in actions:
                    (probabilities, reward) = env.get_transition_probabilities_and_expected_reward((s1, s2), action)
                    actionReturns.append(oneStepAhead(probabilities, reward, Vs))
                bestAction = np.argmax(actionReturns)
                newPolicy[s1, s2] = actions[bestAction]

        # if policy is stable
        policyChanges = np.sum(newPolicy != policy)
        print('Policy for', policyChanges, 'states changed')
        print("Policy: ", newPolicy)
        return newPolicy


def policyEvaluation(Vs, policy, env):
    """
        # start policy evaluation
        for i, j in states:
            newStateValue[i, j] = expectedReturn([i, j], policy[i, j], stateValue)
        if np.sum(np.abs(newStateValue - stateValue)) < 1e-4:
            stateValue[:] = newStateValue
            improvePolicy = True
            continue
        stateValue[:] = newStateValue
    """
    while True:
        v = np.zeros((21, 21))
        for a in range(21):
            for b in range(21):
                (trans_probs, reward) = env.get_transition_probabilities_and_expected_reward((a, b), policy[a][b])
                v[a][b] = oneStepAhead(trans_probs, reward, Vs)
        print("delta: ", np.sum(np.abs(v - Vs)))
        if np.sum(np.abs(v - Vs)) < 1e-4:
            print('Values: ', Vs)
            Vs = v
            return v
        else:
            Vs = v


def policyIteration(iterations=5, env=env.JacksCarRentalEnvironmentModel()):
    Vs = np.zeros((21, 21))
    policy = np.zeros((21, 21), dtype=np.int)

    for i in range(iterations):
        Vs = policyEvaluation(Vs, policy, env)
        policy = policyImprovement(Vs, policy, env)

    return (policy, Vs)

(policy, V) = policyIteration()
plot3d_over_states(V, 'v')
plot_policy(policy)
plot3d_over_states(V)