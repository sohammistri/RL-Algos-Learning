import argparse
import numpy as np
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed

from sample_episodic_mdps import EpisodicMDP

def evaluate_policy_worker(mdp, policy, max_steps, n_episodes, seed=None):
    rng = np.random.default_rng(seed)
    total_returns = np.zeros(mdp.num_states, dtype=float)
    state_visit_cnts = np.zeros(mdp.num_states, dtype=int)
    gamma = mdp.discount

    for _ in range(n_episodes):
        episode = mdp.sample_episode_thread_safe(policy, max_steps=max_steps, rng=rng)
        G = 0.0
        for (s, a, r) in reversed(episode):
            G = gamma * G + r
            s = int(s)
            state_visit_cnts[s] += 1
            total_returns[s] += G

    return state_visit_cnts, total_returns

def on_policy_monte_carlo_ctrl_worker(mdp, max_steps, n_episodes, epsilon_min=0.01, seed=None):
    """
    Monte Carlo Control using ɛ-greedy policy improvement (on-policy). Single worker for the same
    """
    rng = np.random.default_rng(seed)
    # Step 1: Initialise Q table and random policy
    num_states, num_actions, gamma = mdp.num_states, mdp.num_actions, mdp.discount
    Q_values = np.full((num_states, num_actions), -np.inf)
    policy = np.full((num_states, num_actions), 1 / num_actions)
    assert np.allclose(policy.sum(axis=1), 1.0), "Policy rows must sum to 1.0"
    
    state_action_visit_cnt = np.zeros((num_states, num_actions))    # Step 2: Loop for some steps
    for k in range(n_episodes):
        # Step 2.1: sample one episode from this MDP as per policy
        episode = mdp.sample_episode_thread_safe(policy, max_steps=max_steps, rng=rng)

        # Step 2.2: update Q values
        G = 0.0
        for i in reversed(range(len(episode))):
            s, a, r = episode[i]
            G = gamma * G + r
            state_action_visit_cnt[s][a] += 1
            if Q_values[s][a] == -np.inf:
                Q_values[s][a] = 0
            Q_values[s][a] += ((1 / state_action_visit_cnt[s][a]) * (G - Q_values[s][a]))

        # Step 2.3: Update policy
        epsilon = max(epsilon_min, 1 / (k + 1))
        # epsilon = 1
        for s in range(num_states):
            greedy_action = np.argmax(Q_values[s])
            policy[s] = np.full(num_actions, epsilon / num_actions)
            policy[s][greedy_action] += (1 - epsilon)
        
        # Assert policy sums to 1.0 across all rows after update
        assert np.allclose(policy.sum(axis=1), 1.0), "Policy rows must sum to 1.0"

    optimal_policy = [np.argmax(Q_values[s]) for s in range(num_states)]
    optimal_values = [Q_values[s][a] for s, a in enumerate(optimal_policy)]

    return optimal_values, optimal_policy


def main():
    parser = argparse.ArgumentParser(description='Sample episodes from an episodic MDP')
    parser.add_argument('--mode', type=str, required=True, help='Mode to run MC algorithm (eval or ctrl)', choices=['eval', 'ctrl'], default='ctrl')
    parser.add_argument('--mdp', type=str, required=True, help='Path to MDP file')
    parser.add_argument('--policy', type=str, default=None, help='Path to policy file (optional)')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to sample')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--num_threads', type=int, default=16, help='Number of threads')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mean_type', type=str, choices=['mean', 'weighted'], default='mean', help='Type of mean to use (mean or weighted)')
    parser.add_argument('--mc_type', type=str, choices=['first', 'every'], default='every', help='Monte Carlo type (first-visit or every-visit)')
    parser.add_argument('--alpha', type=float, default=None, help='Learning rate for weighted mean (required if mean_type is weighted)')
    parser.add_argument('--min_epsilon', type=float, default=0.01, help='Minimum epsilon in case of epsilon greedy policy optimisation')
    parser.add_argument('--verbose', action='store_true', default=True, help='Show verbose output (default: True)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Disable verbose output')

    
    args = parser.parse_args()

    # print(args)
    
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

        num_workers = args.num_threads
        episodes = args.num_episodes
        per_worker = episodes // num_workers
        leftovers = episodes % num_workers

        state_visit_cnts = np.zeros(mdp.num_states, dtype=int)
        total_returns = np.zeros(mdp.num_states, dtype=float)

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = []
            for i in range(num_workers):
                n = per_worker + (1 if i < leftovers else 0)
                if n == 0:
                    continue
                seed = secrets.randbits(64)
                futures.append(ex.submit(evaluate_policy_worker, mdp, policy, args.max_steps, n, seed + i))

            # collect results as you do now
            for fut in as_completed(futures):
                cnts, tots = fut.result()
                state_visit_cnts += cnts
                total_returns += tots
        
        state_visit_cnts = state_visit_cnts.astype(float)
        V_pi = np.where(state_visit_cnts > 0, total_returns / state_visit_cnts, 0)

        if args.verbose:
            for s, v in enumerate(V_pi):
                print(f"{v:.6f} {np.argmax(policy[s])}")

    # Next ctrl
    elif args.mode == "ctrl":
        num_workers = args.num_threads
        episodes = args.num_episodes
        per_worker = episodes // num_workers
        leftovers = episodes % num_workers

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = []
            for i in range(num_workers):
                n = per_worker + (1 if i < leftovers else 0)
                if n == 0:
                    continue
                seed = secrets.randbits(64)
                futures.append(ex.submit(on_policy_monte_carlo_ctrl_worker, mdp, args.max_steps, n, args.min_epsilon, seed + i))
            # Aggregate worker results: each future returns (optimal_values, optimal_policy)
            # We will take a vote per state from each worker's policy. If multiple actions tie on votes,
            # break ties by selecting the action with the larger corresponding value (averaged across voters).
            worker_results = [f.result() for f in futures]

        # worker_results is a list of tuples: (optimal_values, optimal_policy)
        if len(worker_results) == 0:
            print("No worker produced results")
            return

        num_states = mdp.num_states
        num_actions = mdp.num_actions

        # votes[s][a] = number of workers that picked action a for state s
        votes = np.zeros((num_states, num_actions), dtype=int)
        # value_sums[s][a] = sum of reported values for action a at state s (from workers that voted for a)
        value_sums = np.full((num_states, num_actions), -np.inf, dtype=float)
        value_counts = np.zeros((num_states, num_actions), dtype=int)

        for vals, policy in worker_results:
            # vals: list of length num_states containing value for chosen action at each state
            # policy: list of length num_states containing chosen action for each state
            for s in range(num_states):
                a = int(policy[s])
                votes[s][a] += 1
                v = float(vals[s])
                if value_sums[s][a] == -np.inf:
                    value_sums[s][a] = v
                else:
                    value_sums[s][a] += v
                value_counts[s][a] += 1

        # Compute final policy and values with vote-majority and tie-break by larger average value
        final_policy = [0] * num_states
        final_values = [0.0] * num_states

        for s in range(num_states):
            # find actions with maximum votes
            max_votes = votes[s].max()
            candidate_actions = np.where(votes[s] == max_votes)[0]
            if candidate_actions.size == 1:
                chosen = int(candidate_actions[0])
            else:
                # tie: compute average reported value for each candidate action (fallback to -inf if no reports)
                avg_vals = []
                for a in candidate_actions:
                    if value_counts[s][a] > 0:
                        avg = value_sums[s][a] / value_counts[s][a]
                    else:
                        avg = -np.inf
                    avg_vals.append(avg)
                # pick the candidate with largest average value; if still tie, pick lowest action index
                max_avg = max(avg_vals)
                # find indices in candidate_actions matching max_avg
                best_idxs = [i for i, av in enumerate(avg_vals) if av == max_avg]
                chosen = int(candidate_actions[best_idxs[0]])

            final_policy[s] = chosen
            # compute final value: average of reported values for chosen action if available, else 0
            if value_counts[s][chosen] > 0:
                final_values[s] = float(value_sums[s][chosen] / value_counts[s][chosen])
            else:
                final_values[s] = 0.0

        if args.verbose:
            for s in range(num_states):
                print(f"{final_values[s]:.6f} {final_policy[s]}")


if __name__ == "__main__":
    main()