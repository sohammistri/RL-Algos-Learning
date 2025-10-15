#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
from pulp import *

def parse_arguments():
    """
    Parse command line arguments for the MDP planner.
    
    Returns:
        argparse.Namespace: Parsed arguments containing mdp, algorithm, and policy
    """
    parser = argparse.ArgumentParser(
        description='MDP Planner - Solve Markov Decision Processes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python planner.py --mdp data/mdp/episodic-mdp-2-2.txt
  python planner.py --mdp data/mdp/episodic-mdp-2-2.txt --algorithm hpi
  python planner.py --mdp data/mdp/episodic-mdp-2-2.txt --algorithm lp --policy data/mdp/policy-episodic-mdp-10-5.txt
        """
    )
    
    # Required MDP file argument
    parser.add_argument(
        '--mdp',
        type=str,
        required=True,
        help='Path to the input MDP file (required)'
    )
    
    # Optional algorithm argument with default value
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['hpi', 'lp'],
        default='hpi',
        required=False,
        help='Algorithm to use: hpi (Howard Policy Iteration) or lp (Linear Programming). Default: hpi'
    )
    
    # Optional policy file argument
    parser.add_argument(
        '--policy',
        type=str,
        required=False,
        help='Path to policy file for value function evaluation. Each line contains a single action integer for the corresponding state.'
    )
    
    return parser.parse_args()

def validate_arguments(args):
    """
    Validate the parsed arguments.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        bool: True if arguments are valid, False otherwise
    """
    # Check if MDP file exists
    if not os.path.isfile(args.mdp):
        print(f"Error: MDP file '{args.mdp}' does not exist.", file=sys.stderr)
        return False
    
    # Check if policy file exists (if provided)
    if args.policy and not os.path.isfile(args.policy):
        print(f"Error: Policy file '{args.policy}' does not exist.", file=sys.stderr)
        return False
    
    return True

def load_mdp(mdp_file):
    """
    Load MDP from file.
    
    Args:
        mdp_file (str): Path to MDP file
        
    Returns:
        dict: MDP data structure
    """
    # Implement MDP loading logic

    # Step 1: Read the file and parse the MDP components
    num_states, num_actions, end_states, transitions, mdp_type, gamma = None, None, None, [], None, None

    with open(mdp_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                split_line = line.split()
                # Process each line as needed to build the MDP structure
                if split_line[0] == 'numStates':
                    num_states = int(split_line[1])
                elif split_line[0] == 'numActions':
                    num_actions = int(split_line[1])
                elif split_line[0] == 'end':
                    end_states = list(map(int, split_line[1:]))
                elif split_line[0] == 'transition':
                    # Example transition line: transition s a s' r p
                    s = int(split_line[1])
                    a = int(split_line[2])
                    s_next = int(split_line[3])
                    r = float(split_line[4])
                    p = float(split_line[5])
                    transitions.append((s, a, s_next, r, p))
                elif split_line[0] == 'mdptype':
                    mdp_type = split_line[1]
                elif split_line[0] == 'discount':
                    gamma = float(split_line[1])

    # Step 2: Create the state transition prob and reward matrices
    P, R = np.zeros((num_states, num_actions, num_states)), np.zeros((num_states, num_actions, num_states))
    for (s, a, s_next, r, p) in transitions:
        P[s, a, s_next] += p
        R[s, a, s_next] = r  

    if mdp_type == 'episodic':
        for s in end_states:
            P[s, :, s] = 1

    # assert for every state-action pair, sum of transition probabilities is 1
    for s in range(num_states):     
        for a in range(num_actions):
            if not np.isclose(np.sum(P[s, a, :]), 1):
                print(f"Warning: Transition probabilities for state {s}, action {a} do not sum to 1.", file=sys.stderr)

    return P, R, num_states, num_actions, gamma

def load_policy(policy_file):
    """
    Load policy from file.
    
    Args:
        policy_file (str): Path to policy file
        
    Returns:
        list: Policy as list of actions for each state
    """
    if not policy_file:
        return None
        
    try:
        with open(policy_file, 'r') as f:
            policy = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    policy.append(int(line))
            return policy
    except (ValueError, IOError) as e:
        print(f"Error loading policy file: {e}", file=sys.stderr)
        return None

def evaluate_action_values(mdp_data, v):
    """
    Compute action values for given state values

    Args:
        mdp_data (tuple): MDP details
        v_pi (np.array): value functions, a 1d array of shape (num_states)
    """

    # Step 0: Load MDP
    P, R, num_states, num_actions, gamma = mdp_data

    # Step 1: Get q values
    q = np.sum(P * R, axis=2) + gamma * (P @ v)
    return q


def evaluate_policy(mdp_data, policy, tolerance=1e-8, compute_action_values=False, verbose=False):
    """
    Function to evaluate a policy using policy evaluation algorithm
    Args:
        P (np.array): state action state transition probs
        R (np.array): state action state rewards
        num_states (int): no. of states
        num_actions (int): no. of actions
        gamma (float): discount factor
        policy (list): list of actions to be taken per state
    """
    # Step 0: Load MDP
    P, R, num_states, num_actions, gamma = mdp_data

    # Step 1: define present and computed state value functions
    v_pi, cur = np.random.randn(num_states), None

    # Step 2: iterate until convergence of value function
    while True:
        # Step 2.1: initialise v_pi to cur, if not none
        if cur is not None:
            v_pi = cur.copy()

        # Step 2.2: Take P_pi and R_pi
        P_pi, R_pi = P[np.arange(num_states), policy, :], R[np.arange(num_states), policy, :] # they are of shapes (s, s)

        # Step 2.3: Write the BellMan Equation for this
        cur =np.sum(P_pi * R_pi, axis=1) + gamma * (P_pi @ v_pi)

        if np.linalg.norm(v_pi - cur) < tolerance:
            break
    
    # Step 3: Final policy and value function
    v_pi = cur.copy()

    if verbose:
        v_pi_list = v_pi.tolist()
        for v, a in zip(v_pi_list, policy):
            print(f"{v:.6f}", a)

    if compute_action_values:
        q_pi = evaluate_action_values(mdp_data, v_pi)
        return v_pi, q_pi
        
def solve_mdp_hpi(mdp_data, verbose=False):
    """
    Solve MDP using Howard Policy Iteration.
    
    Args:
        mdp_data: MDP data structure
    """
    # Step 0: Load MDP
    P, R, num_states, num_actions, gamma = mdp_data

    # Step 1: Initialise a random policy
    policy = np.random.randint(0, num_actions, size=num_states)
    v, q, optimal_policy, optimal_values = None, None, None, None

    while True:
        # Step 2.1: Compute state values and action values
        v, q = evaluate_policy(mdp_data, policy.tolist(), compute_action_values=True)
    
        # Step 2.2: Get greedy policy for every state, take the max action for every q
        policy_new = np.argmax(q, axis=1)

        # Step 2.3: Convergence criteria: see if the policies change after each step
        if (policy == policy_new).all():
            optimal_policy = policy.copy()
            optimal_values = v.copy()
            break
        else:
            policy = policy_new.copy()

    # Step 3: Print the answer
    if verbose:
        optimal_values_list, optimal_policy_list = optimal_values.tolist(), optimal_policy.tolist()

        for v, a in zip(optimal_values_list, optimal_policy_list):
            print(f"{v:.6f}", a)

    return optimal_values, optimal_policy
    
def solve_mdp_lp(mdp_data, verbose=False):
    """
    Solve MDP using Linear Programming.
    
    Args:
        mdp_data: MDP data structure
    """
    # Step 0: Load MDP
    P, R, num_states, num_actions, gamma = mdp_data

    # Step 1: Define the problem
    prob = LpProblem("MDP_Optimal_State_Value_Calculation", LpMaximize)

    # Step 2: Define the variables
    state_values = [LpVariable(f"V_{i}", lowBound=None) for i in range(num_states)]

    # Step 3: Add objective function
    prob += (-1 * sum(state_values)), "NegativeOfSumOfStateValues"

    # Step 4: Add the constraints
    constraint_cnt = 1
    for s in range(num_states):
        for a in range(num_actions):
            constraint = -state_values[s] + np.sum(P[s, a, :] * R[s, a, :]) + lpSum(gamma * P[s, a, s_next] * state_values[s_next] for s_next in range(num_states))

            prob += (constraint <= 0), f"Constraint{constraint_cnt}"
            constraint_cnt += 1

    # Step 5: Solve the LP
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # Step 6: Pick optimal value function and use that to backtrack the optimal policy

    optimal_values = [v.varValue for v in state_values]
    action_values = evaluate_action_values(mdp_data, np.array(optimal_values))
    optimal_policy = np.argmax(action_values, axis=1).tolist()

    if verbose:
        for v, a in zip(optimal_values, optimal_policy):
            print(f"{v:.6f}", a)

    return optimal_values, optimal_policy

def main():
    """
    Main function to run the MDP planner.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Load MDP
    mdp_data = load_mdp(args.mdp)
    
    if args.policy:
        policy = load_policy(args.policy)
        evaluate_policy(mdp_data, policy, verbose=True)
    elif args.algorithm == 'hpi':
        solve_mdp_hpi(mdp_data, verbose=True)
    elif args.algorithm == 'lp':
        solve_mdp_lp(mdp_data, verbose=True)
    else:
        print(f"Error: Unknown algorithm '{args.algorithm}'", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()