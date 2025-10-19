#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np
import pickle
from encoder import MDP, Hand

def parse_arguments():
    """
    Parse command line arguments for the decoder.
    
    Returns:
        argparse.Namespace: Parsed arguments containing value_policy and testcase
    """
    parser = argparse.ArgumentParser(
        description='Decoder - Determine optimal actions for test cases using value policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python decoder.py --value_policy output/value_policy.txt --testcase data/test/test_0.txt
  python decoder.py --value_policy output/value_policy.txt --automate results/optimal_policy.txt
        """
    )
    
    # Required value_policy file argument
    parser.add_argument(
        '--value_policy',
        type=str,
        required=True,
        help='Path to the value-policy file (output of planner) (required)'
    )
    
    # Required testcase file argument
    parser.add_argument(
        '--testcase',
        type=str,
        required=False,
        help='Path to the testcase file (required)'
    )
    
    # Optional automate file argument
    parser.add_argument(
        '--automate',
        type=str,
        required=False,
        help='Path to output file for automated optimal policy. If specified, writes optimal actions for all possible hands to file instead of stdout (optional)'
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
    # Check if value_policy file exists
    if not os.path.isfile(args.value_policy):
        print(f"Error: Value-policy file '{args.value_policy}' does not exist.", file=sys.stderr)
        return False
    
    # Check if testcase file exists
    if args.testcase:
        if not os.path.isfile(args.testcase):
            print(f"Error: Testcase file '{args.testcase}' does not exist.", file=sys.stderr)
            return False
    
    return True

def load_value_policy(value_policy_file):
    """
    Load value-policy from file (output of planner).
    
    Args:
        value_policy_file (str): Path to value-policy file
        
    Returns:
        tuple: (values, policy) - arrays of state values and optimal actions
    """
    try:
        with open(value_policy_file, 'r') as f:
            lines = f.readlines()
        
        values = []
        policy = []
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    # Assuming format: value action
                    values.append(float(parts[0]))
                    policy.append(int(parts[1]))
        
        return np.array(values), np.array(policy)
        
    except (ValueError, IOError) as e:
        print(f"Error loading value-policy file: {e}", file=sys.stderr)
        return None, None

def parse_testcase(testcase_file):
    """
    Parse testcase file to extract game configuration and test instances.
    
    Args:
        testcase_file (str): Path to testcase file
        
    Returns:
        tuple: (config, instances) where config is (threshold, bonus, sequence) 
               and instances is a list of agent hands
    """
    try:
        with open(testcase_file, 'r') as f:
            lines = f.readlines()
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines]
        
        # Parse configuration (same format as game_config)
        config_idx = 0
        if lines[0] == "Configuration:":
            config_idx = 1
        
        threshold = int(lines[config_idx])
        bonus = int(lines[config_idx + 1])
        sequence = list(map(int, lines[config_idx + 2].split()))
        
        # Find where test instances start
        testcase_idx = config_idx + 3
        if lines[testcase_idx] == "Testcase:":
            testcase_idx += 1
        
        # Parse test instances (each line is a hand)
        instances = []
        for i in range(testcase_idx, len(lines)):
            hand_str = lines[i]
            instances.append(hand_str)
        
        return (threshold, bonus, sequence), instances
        
    except (ValueError, IOError, IndexError) as e:
        print(f"Error parsing testcase file: {e}", file=sys.stderr)
        return None, None

def parse_hand_string(hand_str):
    """
    Parse a hand string like "1H 5D 4H" into a binary hand representation.
    
    Args:
        hand_str (str): Space-separated card representations (e.g., "1H 5D 4H")
        
    Returns:
        list: Binary list of 26 values (0-12 for Hearts, 13-25 for Diamonds)
    """
    # Initialize hand with all zeros (no cards)
    hand = [0 for _ in range(26)]
    
    if len(hand_str.strip()) == 0:
        return hand
    
    # Parse each card
    cards = hand_str.split()
    for card in cards:
        # Extract rank and suit
        # Format: <rank><suit> where rank is 1-13, suit is H or D
        rank_str = card[:-1]  # All but last character
        suit = card[-1]        # Last character
        
        rank = int(rank_str)
        
        # Convert to index (0-25)
        if suit == 'H':
            # Hearts: indices 0-12 (for ranks 1-13)
            card_idx = rank - 1
        elif suit == 'D':
            # Diamonds: indices 13-25 (for ranks 1-13)
            card_idx = 13 + rank - 1
        else:
            print(f"Warning: Unknown suit '{suit}' in card '{card}'", file=sys.stderr)
            continue
        
        hand[card_idx] = 1
    
    return hand

def load_state_mappings(pickle_file='state_mappings.pkl'):
    """
    Load state mappings from pickle file.
    
    Args:
        pickle_file (str): Path to pickle file containing state mappings
        
    Returns:
        dict: Dictionary containing states_to_id, id_to_states, and config info
    """
    try:
        with open(pickle_file, 'rb') as f:
            state_data = pickle.load(f)
        return state_data
    except (IOError, pickle.PickleError) as e:
        print(f"Error loading state mappings from {pickle_file}: {e}", file=sys.stderr)
        return None

def decode_state(state):
    hand_one_hot = state.hand

    hand = []
    for i, h in enumerate(hand_one_hot):
        if h == 1 and i < 13:
            # hearts card
            hand.append(f"{i + 1}H")
        elif h == 1:
            # diamond card
            hand.append(f"{i - 12}D")

    return " ".join(hand)

def main():
    """
    Main function to run the decoder.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Load value-policy
    values, policy = load_value_policy(args.value_policy)
    
    if values is None or policy is None:
        print("Error: Failed to load value-policy.", file=sys.stderr)
        sys.exit(1)
    
    # Load state mappings from pickle file
    pickle_path = r'state_mappings.pkl'
    state_data = load_state_mappings(pickle_path)
    
    if state_data is None:
        print("Error: Failed to load state mappings.", file=sys.stderr)
        sys.exit(1)
    
    states_to_id = state_data['states_to_id']
    id_to_states = state_data['id_to_states']
    
    if args.automate:
        try:
            # Create automate directory if it doesn't exist
            automate_dir = os.path.dirname(args.automate)
            if automate_dir and not os.path.exists(automate_dir):
                os.makedirs(automate_dir)
            
            out_strings = []
            with open(args.automate, 'w') as f:
                # Write optimal policy for all states (excluding STOP and BUST)
                for state_id in sorted(id_to_states.keys()):
                    if state_id <= 1:  # Skip STOP and BUST states
                        continue
                    
                    state = id_to_states[state_id]
                    hand = decode_state(state)
                    optimal_action = policy[state_id]
                    
                    # Write state ID, hand representation, and optimal action
                    out_strings.append(f"{hand} -> {optimal_action}")

                out_strings = sorted(out_strings)

                for o in out_strings:
                    f.write(o + "\n")
                
            # print(f"Optimal policy for all possible hands written to {args.automate}", file=sys.stderr)
        
        except IOError as e:
            print(f"Error writing automate file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.testcase:
        # Parse testcase
        config, instances = parse_testcase(args.testcase)
        
        if config is None or instances is None:
            print("Error: Failed to parse testcase.", file=sys.stderr)
            sys.exit(1)
        
        # Process each test instance and print to stdout
        for instance in instances:
            # Parse hand string to binary representation
            hand = parse_hand_string(instance)
            
            # Convert hand to state ID
            state_id = states_to_id[Hand(list(hand))]
            
            # Get optimal action
            optimal_action = policy[state_id]
            
            # Print optimal action
            print(optimal_action)

if __name__ == "__main__":
    main()
