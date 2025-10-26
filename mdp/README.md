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