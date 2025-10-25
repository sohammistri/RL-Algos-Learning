import argparse
import numpy as np

from sample_episodic_mdps import EpisodicMDP

def evaluate_policy(mdp, policy, episodes, V_pi, mc_type, mean_type, alpha, verbose):
    state_visit_cnt = np.zeros(mdp.num_states)

    # check if mean_type is mean or weighted, if weighted, check if alpha exists.
    if mean_type == " weighted":
        try:
            assert alpha is not None
        except:
            raise("Weighted mean not possible without weight alpha")

    for episode in episodes:
        G = 0
        if mc_type == "first":
            # mantain where state was first visited
            state_visit_id = np.zeros(mdp.num_states) - 1 # idxes made -1
            for i, t in enumerate(episode):
                s, _, _ = t
                s = int(s)
                if state_visit_id[s] == -1:
                    state_visit_id[s] = len(episode) - 1 - i

        # next evaluate
        for i, t in enumerate(episode[::-1]):
            s, a, r = t
            s, a, r = int(s), int(a), float(r)
            G = mdp.discount * G + r
            # print(s, a, r, G)
            if mc_type == "every":
                state_visit_cnt[s] += 1
                if mean_type == "mean":
                    V_pi[s] += ((1 / state_visit_cnt[s]) * (G - V_pi[s]))
                else:
                    V_pi[s] += (alpha * (G - V_pi[s]))
            elif mc_type == "first" and i == state_visit_id[s]:
                state_visit_cnt[s] += 1
                if mean_type == "mean":
                    V_pi[s] += ((1 / state_visit_cnt[s]) * (G - V_pi[s]))
                else:
                    V_pi[s] += (alpha * (G - V_pi[s]))
                

    if verbose:
        v_pi_list = V_pi.tolist()
        for v, a in zip(v_pi_list, policy):
            print(f"{v:.6f}", a)



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
            raise("Cannot evaluate a policy not provided")
        
        # Load policy
        policy = mdp.load_policy(args.policy)

        # Sample episodes
        episodes = mdp.sample_multiple_episodes(args.episodes, policy, args.max_steps)

        # create value functions
        V_pi = np.zeros(mdp.num_states, dtype=float)
        evaluate_policy(mdp, policy, episodes, V_pi, args.mc_type, args.mean_type, args.alpha, args.verbose)
    

if __name__ == "__main__":
    main()