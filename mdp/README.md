# MDP Solvers:

Implement multiple solvers for MDPs staring from model-based (Dynamic Programming) to model-free (Monte Carlo, SARSA, Q-Learning)

# Implemented Algorithms:
- **Dynamic Programming**: Policy Iteration, Value Iteration, Linear Programming
- **Monte Carlo Methods**:
    - Implement sampling a fixed no. of episodes from a given MDP given a policy.
    - Implement policy improvement (epsilon policy improvement)
    - **Running Notes**:
        - Initialise Q values to -ve inf at start to handle -ve rewards.
        - Better to have an epsilon min to ensure some exploration happens even after many updates.
        - Still lot of issues with on policy MC, not many state action pairs seen, parallelisation still issues
        - Important to have many samples across episodes, preferentially to end in termination to have the most accurate return values.
        - Need to show a lot of episodes to each thread to have accurate Q-values
        - Further, it is better to have the threaded voting since works better and have shorter execution times.
- **Temporal Difference(0) Methods**:
    - Implement TD(0) policy evaluation
    - **Running Notes**:
        - Implement 3 modes for lr: fixed, adagrad and rmsprop
        - Continuing TestCase (1M iters, alpha 0.01):
            - *Fixed*: inf_norm 0.087454
            - *Adagrad*: inf_norm 0.008525
            - *RmsProp (beta 0.99)*: inf_norm 0.100381
        - Episodic TestCase (1M iters, alpha 0.01):
            - *Fixed*: inf_norm 0.073291
            - *Adagrad*: inf_norm 0.009131
            - *RmsProp (beta 0.99)*: inf_norm 0.082586