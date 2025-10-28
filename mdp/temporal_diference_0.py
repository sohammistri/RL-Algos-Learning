import argparse
from typing import Optional
import numpy as np

from sample_episodic_mdps import MDP


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
        # fallback: uniform random policy
        policy_dist = np.full((num_states, num_actions), 1.0 / num_actions, dtype=float)
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

    # Validate/repair policy rows so each sums to 1.0; if a row is all zeros, use uniform.
    row_sums = policy_dist.sum(axis=1, keepdims=True)
    for s in range(num_states):
        if not np.isfinite(row_sums[s, 0]) or row_sums[s, 0] <= 0:
            policy_dist[s, :] = 1.0 / num_actions
        else:
            policy_dist[s, :] = policy_dist[s, :] / row_sums[s, 0]

    terminal_states = set(mdp.terminal_states)

    for _ in range(num_iters):
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
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                # fallback: uniform over states if transition row is invalid
                probs = np.full(num_states, 1.0 / num_states, dtype=float)

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



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TD(0) boilerplate for MDPs")
    parser.add_argument('--mode', type=str, required=True, help='Mode to run MC algorithm (eval or ctrl)', choices=['eval', 'ctrl'], default='ctrl')
    parser.add_argument("--mdp", type=str, required=True, help="Path to MDP file")
    parser.add_argument('--policy', type=str, default=None, help='Path to policy file (optional)')
    parser.add_argument('--eval-sol', type=str, default=None, help='Path to true value-action solution file for infinity-norm evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--num_iters", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--alpha", type=float, default=0.1, help="TD(0) learning rate alpha")
    parser.add_argument("--lr", type=str, choices=["fixed", "adagrad", "rmsprop"], default="fixed", help="Learning rate schedule")
    parser.add_argument("--beta", type=float, default=0.99, help="Beta for RMSProp accumulator")
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
            # normalize defensively
            row_sums = policy.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            policy = policy / row_sums

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
        pass

if __name__ == "__main__":
    main()


