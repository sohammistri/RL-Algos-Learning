import argparse
from multiprocessing import Value
from typing import Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed # <-- Change this
import numba
from sample_episodic_mdps import MDP
from tqdm import tqdm, trange

def td0_policy_evaluation(
    mdp: MDP,
    policy: Optional[object] = None,
    num_iters: int = 10,
    alpha: float = 0.1,
    lr: str = "fixed",
    beta: float = 0.99,
    seed: int = None,
):
    """
    TD(0) policy evaluation on an MDP.
    """
    rng = np.random.default_rng(seed)
    num_states, num_actions, gamma = mdp.num_states, mdp.num_actions, mdp.discount
    values = np.zeros(num_states, dtype=float)
    # Per-state accumulators for adaptive step sizes
    adagrad_accum = np.zeros(num_states, dtype=float)
    rmsprop_accum = np.zeros(num_states, dtype=float)

    # Accept either deterministic (shape: [num_states], entries are action indices)
    # or stochastic (shape: [num_states, num_actions]) policies.
    if policy is None:
        raise ValueError("Policy needs to be provided in order to be evaluated")
    else:
        policy_arr = np.asarray(policy)
        if policy_arr.ndim == 1:
            # deterministic actions -> convert to one-hot distribution
            policy_dist = np.zeros((num_states, num_actions), dtype=float)
            for s in range(num_states):
                a = int(policy_arr[s])
                policy_dist[s, a] = 1.0
        elif policy_arr.ndim == 2:
            policy_dist = policy_arr.astype(float, copy=False)
        else:
            raise ValueError("Policy must be 1D (actions) or 2D (distributions).")

    # Validate that policy rows sum to 1.0
    row_sums = policy_dist.sum(axis=1)
    for s in range(num_states):
        if not np.isclose(row_sums[s], 1.0):
            raise ValueError(f"Policy row for state {s} does not sum to 1.0 (sum: {row_sums[s]})")

    terminal_states = set(mdp.terminal_states)

    for _ in trange(num_iters, desc="TD(0) Evaluation"):
        # Sample a non-terminal state to avoid degenerate self-loops
        # If all states are terminal (degenerate MDP), break.
        if len(terminal_states) == num_states:
            break
        while True:
            s1 = int(rng.integers(0, num_states))
            if s1 not in terminal_states:
                break

        a1 = int(rng.choice(num_actions, p=policy_dist[s1, :]))
        probs = mdp.P[s1, a1, :].astype(float, copy=False)
        prob_sum = probs.sum()
        if not np.isclose(prob_sum, 1.0):
            raise ValueError("MDP is not correct, probs don't add to 1.0")

        s2 = int(rng.choice(num_states, p=probs))
        r1 = float(mdp.R[s1, a1, s2])
        td_error = r1 + gamma * values[s2] - values[s1]

        # Learning rate schedule
        if lr == "fixed":
            alpha_updated = alpha
        elif lr == "adagrad":
            adagrad_accum[s1] += td_error * td_error
            alpha_updated = alpha / np.sqrt(adagrad_accum[s1] + 1e-8)
        elif lr == "rmsprop":
            rmsprop_accum[s1] = beta * rmsprop_accum[s1] + (1.0 - beta) * (td_error * td_error)
            alpha_updated = alpha / np.sqrt(rmsprop_accum[s1] + 1e-8)
        else:
            raise ValueError("Unknown lr mode. Use one of: fixed, adagrad, rmsprop.")

        values[s1] = values[s1] + alpha_updated * td_error

    return values

@numba.jit(nopython=True) # Add this decorator
def on_policy_td0_control_sarsa_worker_numba(
    # NOTE: Numba works best with simple types.
    # The MDP object would need to be refactored to pass its underlying
    # NumPy arrays (P, R) and scalar values (num_states, etc.) directly.
    P,
    R,
    num_states,
    num_actions,
    gamma,
    terminal_states,
    num_iters: int = 10,
    alpha: float = 0.1,
    # For simplicity, this example only shows the 'fixed' lr.
    # Supporting dynamic strings like 'rmsprop' requires more advanced Numba usage.
    # lr: str = "fixed",
    # beta: float = 0.99,
    epsilon_min: float = 0.01,
    seed: int = None
):
    """
    Numba-compatible JIT compiled version of the SARSA worker.
    """
    # Numba requires the random number generator to be initialized inside the function.
    if seed is not None:
        np.random.seed(seed)

    Q_values = np.zeros((num_states, num_actions), dtype=np.float64)

    for k in range(num_iters):
        epsilon = max(epsilon_min, 1 / (k + 1))
        a_star = np.argmax(Q_values, axis=1)
        
        # Epsilon-greedy policy generation
        policy = np.full((num_states, num_actions), epsilon / num_actions)
        for s in range(num_states): # Numba is good at optimizing loops
            policy[s, a_star[s]] += (1 - epsilon)

        # Sample initial state
        while True:
            s1 = np.random.randint(0, num_states)
            # Numba works with NumPy arrays for checking membership
            if s1 not in terminal_states:
                break

        a1 = np.searchsorted(np.cumsum(policy[s1, :]), np.random.rand())

        probs = P[s1, a1, :]
        s2 = np.searchsorted(np.cumsum(probs), np.random.rand())
        r1 = R[s1, a1, s2]

        a2 = np.searchsorted(np.cumsum(policy[s2, :]), np.random.rand())

        td_error = r1 + gamma * Q_values[s2, a2] - Q_values[s1, a1]
        Q_values[s1, a1] += alpha * td_error

    optimal_policy = np.argmax(Q_values, axis=1)
    
    # Manually compute the max along axis=1
    optimal_values = np.zeros(num_states, dtype=np.float64)
    for s in range(num_states):
        optimal_values[s] = np.max(Q_values[s, :])
    
    return optimal_values, optimal_policy

def on_policy_td0_control_sarsa(
    mdp: MDP,
    num_iters: int = 10,
    alpha: float = 0.1,
    lr: str = "fixed",
    beta: float = 0.99,
    epsilon_min: float = 0.01,
    num_workers: int = 16,
    seed: int = None
):
    """
    Multithreaded on-policy SARSA algorithm for optimal policy on an MDP.
    Runs multiple workers in parallel with different seeds, then aggregates results.
    """
    num_states, num_actions, gamma = mdp.num_states, mdp.num_actions, mdp.discount
    P, R = mdp.P.copy(), mdp.R.copy()
    terminal_states = np.array(mdp.terminal_states)
    
    # Run workers in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            worker_seed = seed + i if seed is not None else None
            future = executor.submit(
                on_policy_td0_control_sarsa_worker_numba,
                P,
                R,
                num_states,
                num_actions,
                gamma,
                terminal_states,
                num_iters,
                alpha,
                epsilon_min,
                worker_seed
            )
            futures.append(future)
        
        # Collect results from all workers
        all_optimal_values = []
        all_optimal_policies = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="SARSA workers", unit="worker"):
            optimal_values, optimal_policy = future.result()
            all_optimal_values.append(optimal_values)
            all_optimal_policies.append(optimal_policy)
    
    # Aggregate policies: for each state, take the most occurring action
    # Break ties by choosing the action with larger average Q-values
    aggregated_policy = np.zeros(num_states, dtype=int)
    
    for s in range(num_states):
        # Count occurrences of each action
        action_counts = {}
        action_values = {}
        
        for worker_policy, worker_values in zip(all_optimal_policies, all_optimal_values):
            action = int(worker_policy[s])
            if action not in action_counts:
                action_counts[action] = 0
                action_values[action] = []
            action_counts[action] += 1
            action_values[action].append(worker_values[s])
        
        # Find actions with maximum count
        max_count = max(action_counts.values())
        max_actions = [a for a, count in action_counts.items() if count == max_count]
        
        # If tie, break by choosing action with larger average value
        if len(max_actions) == 1:
            aggregated_policy[s] = max_actions[0]
        else:
            # Calculate average values for tied actions
            avg_values = {a: np.mean(action_values[a]) for a in max_actions}
            aggregated_policy[s] = max(max_actions, key=lambda a: avg_values[a])
    
    # For values, take the average across all workers
    aggregated_values = np.mean(all_optimal_values, axis=0)
    
    return aggregated_values, aggregated_policy

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TD(0) boilerplate for MDPs")
    parser.add_argument('--mode', type=str, required=True, help='Mode to run MC algorithm (eval or ctrl)', choices=['eval', 'ctrl'], default='ctrl')
    parser.add_argument("--mdp", type=str, required=True, help="Path to MDP file")
    parser.add_argument('--policy', type=str, default=None, help='Path to policy file (optional)')
    parser.add_argument('--eval-sol', type=str, default=None, help='Path to true value-action solution file for infinity-norm evaluation')
    parser.add_argument('--ctrl-sol', type=str, default=None, help='Path to true value-policy solution file for infinity-norm evaluation in control mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--num_iters", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--alpha", type=float, default=0.1, help="TD(0) learning rate alpha")
    parser.add_argument("--lr", type=str, choices=["fixed", "adagrad", "rmsprop"], default="fixed", help="Learning rate schedule")
    parser.add_argument("--beta", type=float, default=0.99, help="Beta for RMSProp accumulator")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon for exploration")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of parallel workers for SARSA control")
    parser.add_argument("--verbose", action="store_true", default=True, help="Show verbose output (default: True)")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose output")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load MDP (episodic MDP placeholder; adjust as needed for your MDP type)
    mdp = MDP(args.mdp)

    # First, implement eval:
    if args.mode == "eval":
        try:
            assert args.policy is not None
        except:
            raise ValueError("Cannot evaluate a policy not provided")
        
        # Load policy
        policy = mdp.load_policy(args.policy)
        # Basic sanity: ensure policy rows form distributions
        if not np.allclose(policy.sum(axis=1), 1.0):
            raise ValueError("Not a valid policy provided")

        # Solve the policy
        value_functions = td0_policy_evaluation(
            mdp,
            policy,
            args.num_iters,
            args.alpha,
            args.lr,
            args.beta,
            args.seed,
        )

        # Optional: evaluate infinity norm vs provided solution file
        if args.eval_sol:
            true_vals = []
            with open(args.eval_sol, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Expect: value action
                    true_vals.append(float(parts[0]))
            true_vals = np.asarray(true_vals, dtype=float)
            diff = np.abs(value_functions - true_vals)
            inf_norm = float(np.max(diff)) if diff.size > 0 else 0.0
            print(f"inf_norm {inf_norm:.6f}")

        # Print value functions
        if args.verbose:
            for s, v in enumerate(value_functions):
                print(f"{v:.6f} {np.argmax(policy[s])}")
    elif args.mode == "ctrl":
        optimal_values, optimal_policy = on_policy_td0_control_sarsa(
            mdp,
            args.num_iters,
            args.alpha,
            args.lr,
            args.beta,
            args.epsilon_min,
            args.num_workers,
            args.seed,
        )

        # Optional: evaluate infinity norm vs provided solution file
        if args.ctrl_sol:
            true_vals = []
            true_policy = []
            with open(args.ctrl_sol, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Expect: value policy
                    true_vals.append(float(parts[0]))
                    true_policy.append(int(parts[1]))
            true_vals = np.asarray(true_vals, dtype=float)
            true_policy = np.asarray(true_policy, dtype=int)
            
            # Compare value functions
            val_diff = np.abs(optimal_values - true_vals)
            val_inf_norm = float(np.max(val_diff)) if val_diff.size > 0 else 0.0
            print(f"value_inf_norm {val_inf_norm:.6f}")
            
            # Compare policies (check if any actions differ)
            policy_matches = np.array_equal(optimal_policy, true_policy)
            if policy_matches:
                print("policy_match True")
            else:
                print("policy_match False")

        if args.verbose:
            for v, a in zip(optimal_values, optimal_policy):
                print(f"{v:.6f} {a}")

if __name__ == "__main__":
    main()


