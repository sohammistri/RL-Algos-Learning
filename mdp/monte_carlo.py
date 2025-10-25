import argparse
import numpy as np

from sample_episodic_mdps import EpisodicMDP

def evaluate_policy(mdp, policy, episodes, mc_type="every", mean_type="mean", alpha=None, verbose=False):
    """Monte Carlo policy evaluation for episodic MDP."""
    num_states = mdp.num_states
    V_pi = np.zeros(num_states)
    state_visit_cnt = np.zeros(num_states)

    if mean_type == "weighted" and alpha is None:
        raise ValueError("Weighted mean requires alpha")

    gamma = mdp.discount

    for episode in episodes:
        # Compute returns from end to start
        G = 0.0
        returns = np.zeros(len(episode))
        for i in reversed(range(len(episode))):
            s, a, r = episode[i]
            G = gamma * G + r
            returns[i] = G

        # First-visit bookkeeping
        first_visit_idx = {}
        if mc_type == "first":
            for t, (s, _, _) in enumerate(episode):
                s = int(s)
                if s not in first_visit_idx:
                    first_visit_idx[s] = t

        # Update values
        for t, (s, a, r) in enumerate(episode):
            s = int(s)
            if mc_type == "every" or (mc_type == "first" and first_visit_idx[s] == t):
                state_visit_cnt[s] += 1
                if mean_type == "mean":
                    V_pi[s] += (1 / state_visit_cnt[s]) * (returns[t] - V_pi[s])
                else:  # weighted
                    V_pi[s] += alpha * (returns[t] - V_pi[s])

    if verbose:
        print("\nEstimated State Values (V_pi):")
        for s, v in enumerate(V_pi):
            print(f"{v:.6f} {policy[s]}")

    return V_pi

def main():
    parser = argparse.ArgumentParser(description='Sample episodes from an episodic MDP')
    parser.add_argument('--mode', type=str, required=True, help='Mode to run MC algorithm (eval or ctrl)', choices=['eval', 'ctrl'], default='ctrl')
    parser.add_argument('--mdp', type=str, required=True, help='Path to MDP file')
    parser.add_argument('--policy', type=str, default=None, help='Path to policy file (optional)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to sample')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mean_type', type=str, choices=['mean', 'weighted'], default='mean', help='Type of mean to use (mean or weighted)')
    parser.add_argument('--mc_type', type=str, choices=['first', 'every'], default='every', help='Monte Carlo type (first-visit or every-visit)')
    parser.add_argument('--alpha', type=float, default=None, help='Learning rate for weighted mean (required if mean_type is weighted)')
    parser.add_argument('--verbose', action='store_true', default=True, help='Show verbose output (default: True)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Disable verbose output')

    
    args = parser.parse_args()

    print(args)
    
    # Set random seed
    np.random.seed(args.seed)

    # Load MDP
    mdp = EpisodicMDP(args.mdp)

    # First, implement eval:
    if args.mode == "eval":
        try:
            assert args.policy is not None
        except:
            raise ValueError("Cannot evaluate a policy not provided")
        
        # Load policy
        policy = mdp.load_policy(args.policy)

        # Sample episodes
        episodes = mdp.sample_multiple_episodes(args.episodes, policy, args.max_steps)

        # create value functions
        V_pi = evaluate_policy(mdp, policy, episodes, args.mc_type, args.mean_type, args.alpha, args.verbose)
    

if __name__ == "__main__":
    main()