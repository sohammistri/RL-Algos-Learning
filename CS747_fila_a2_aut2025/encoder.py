import argparse
import sys
import os
import numpy as np

class Hand:
    def __init__(self, hand):
        """
        Initialse a given hand
        Args:
            hand (list): a binary list of 26 values of each of the available cards: 1-13 D, 1-13 H
        """
        self.hand = list(hand)


    def get_hand_total(self):
        """
        Return the total cost of the hand
        """
        total = 0

        for i in range(13):
            # for cards 1-13H
            total += (self.hand[i] * (i + 1))
            # for cards 1-13D
            total += (self.hand[13 + i] * (i + 1))

        return total
    
    def update(self,added_card_id=None, removed_card_id=None):
        # Validate that at least one operation is specified
        assert added_card_id is not None or removed_card_id is not None, "Must specify at least one of added_card_id or removed_card_id"
        
        # Validate added_card_id if provided
        if added_card_id is not None:
            assert 0 <= added_card_id < len(self.hand), f"Invalid added_card_id: {added_card_id}. Must be between 0 and {len(self.hand)-1}"
            assert self.hand[added_card_id] == 0, f"Card {added_card_id} is already in the hand"
        
        # Validate removed_card_id if provided
        if removed_card_id is not None:
            assert 0 <= removed_card_id < len(self.hand), f"Invalid removed_card_id: {removed_card_id}. Must be between 0 and {len(self.hand)-1}"
            assert self.hand[removed_card_id] == 1, f"Card {removed_card_id} is not in the hand, cannot remove"
        
        # Update hand
        if added_card_id is not None:
            self.hand[added_card_id] = 1
        if removed_card_id is not None:
            self.hand[removed_card_id] = 0
    
    def to_tuple(self):
        """
        Convert hand to tuple for use as dictionary key
        
        Returns:
            tuple: Immutable tuple representation of the hand
        """
        return tuple(self.hand)
    
    def __hash__(self):
        """
        Make Hand hashable so it can be used as a dictionary key
        
        Returns:
            int: Hash value based on the tuple of the hand
        """
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        """
        Check equality between two Hand objects
        
        Args:
            other: Another Hand object or tuple
            
        Returns:
            bool: True if hands are equal, False otherwise
        """
        if isinstance(other, Hand):
            return self.hand == other.hand
        elif isinstance(other, (tuple, list)):
            return self.hand == list(other)
        return False
    
    def __repr__(self):
        """
        String representation of the Hand object
        
        Returns:
            str: String representation
        """
        return f"Hand({self.hand})"

class MDP:
    def __init__(self, threshold, bonus, sequence):
        # states info
        self.states_to_id = {"STOP": 0, "BUST": 1}
        self.id_to_states = {0: "STOP", 1: "BUST"}
        self.stop_states = (0, 1)

        # game related info
        self.threshold = threshold
        self.bonus = bonus
        self.sequence = sequence

        # start the state creation process
        hand = Hand(hand=[0 for _ in range(26)])
        self.create_hands(hand, 0)

        # actions, transitions and rewards
        self.num_states = len(self.states_to_id)
        self.num_actions = 28
        # 0 for draw, 1-26 for swap and 27 for stop

        # final out string
        self.out_string = f"numStates {self.num_states}\nnumActions {self.num_actions}\nend {" ".join(str(s) for s in self.stop_states)}"

        # self.P = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
        # self.R = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.int8)

        # # handle for stop states
        # for s in self.stop_states:
        #     self.P[s, :, s] = 1.0

        # fill the matrices
        for id, state in self.id_to_states.items():
            if id <= 1:
                continue
            # draw action
            self.draw(id, state)
            # swap action
            self.swap(id, state)
            # stop action
            self.stop(id, state)

        # assert for every state-action pair, sum of transition probabilities is 1
        # for s in range(self.num_states):     
        #     for a in range(self.num_actions):
        #         if not np.isclose(np.sum(self.P[s, a, :]), 1):
        #             print(f"Error: Transition probabilities for state {s}, action {a} do not sum to 1.", file=sys.stderr)
        #             print(f"Probabilities: {self.P[s, a, :]}", file=sys.stderr)
        #             print(f"Sum: {np.sum(self.P[s, a, :])}", file=sys.stderr)
        #             raise AssertionError(f"Invalid transition probabilities for state {s}, action {a}")

        # # assert that rewards are 0 everywhere except for action 27 (stop) transitioning to state 0 (STOP)
        # # Create a mask for valid reward positions (action 27, next_state 0)
        # valid_reward_mask = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=bool)
        # valid_reward_mask[:, 27, 0] = True
        
        # # Check if any rewards are non-zero outside the valid positions
        # invalid_rewards = self.R[~valid_reward_mask]
        # if np.any(invalid_rewards != 0):
        #     # Find the first non-zero invalid reward for error message
        #     invalid_indices = np.argwhere((self.R != 0) & (~valid_reward_mask))
        #     if len(invalid_indices) > 0:
        #         s, a, s_next = invalid_indices[0]
        #         print(self.id_to_states[s], a, self.id_to_states[s_next], self.R[s, a, s_next])
        #         raise AssertionError(f"Non-zero reward found at state {s}, action {a}, next_state {s_next}: {self.R[s, a, s_next]}")
        
        # print("Validation passed: Rewards are 0 everywhere except for action 27 (stop) to state 0 (STOP) and Probs are well defined")

        # update the last few entries in the out string
        self.out_string += "\nmdptype episodic\ndiscount  1.0"

    def create_hands(self, cur_hand, cur_card_id):
        # Base case: if all cards have been considered
        if cur_card_id == 26:
            hand_total = cur_hand.get_hand_total()
            # Only add valid hands (not busted)
            if hand_total <= self.threshold:
                if cur_hand not in self.states_to_id:
                    # Create a new Hand object with a copy of the current hand
                    hand_copy = Hand(cur_hand.hand.copy())
                    new_id = len(self.states_to_id)
                    self.states_to_id[hand_copy] = new_id
                    self.id_to_states[new_id] = hand_copy
            return
        
        # Get current hand total
        hand_total = cur_hand.get_hand_total()
        
        # If threshold already crossed, no point continuing this branch
        if hand_total > self.threshold:
            return
        
        # Step 2.3: no take situation, don't take the current card
        self.create_hands(cur_hand, cur_card_id + 1)

        # Step 2.4: take situation, take the card, update hand and continue
        cur_hand.update(added_card_id=cur_card_id)
        self.create_hands(cur_hand, cur_card_id + 1)
        # remove the card after this (backtrack)
        cur_hand.update(removed_card_id=cur_card_id)

    def draw(self, id, state):
        hand = state.hand
        num_deck_cards = 26 - sum(hand)

        for card_id, card in enumerate(hand):
            if card == 0:
                # card eligible to be drawn
                new_state = Hand(list(hand))

                # update new state and get total
                new_state.update(added_card_id=card_id)
                state_total = new_state.get_hand_total()

                if state_total > self.threshold:
                    # Bust
                    # self.P[id, 0, 1] += (1 / num_deck_cards)
                    self.out_string += f"\ntransition {id} 0 1 0 {(1 / num_deck_cards)}"
                else:
                    # transition to next state
                    new_state_id = self.states_to_id[new_state]
                    # self.P[id, 0, new_state_id] += (1 / num_deck_cards)
                    self.out_string += f"\ntransition {id} 0 {new_state_id} 0 {(1 / num_deck_cards)}"

    def swap(self, id, state):
        hand = state.hand
        num_deck_cards = 26 - sum(hand)
        deck_cards = [card_id for card_id, card in enumerate(hand) if card == 0]

        for card_id, card in enumerate(hand):
            action_id = card_id + 1
            if card == 1:
                # card eligible to be swapped
                for new_card_id in deck_cards:
                    new_state = Hand(list(hand))

                    # update new state and get total
                    new_state.update(added_card_id=new_card_id, removed_card_id=card_id)
                    state_total = new_state.get_hand_total()

                    if state_total > self.threshold:
                        # Bust
                        # self.P[id, action_id, 1] += (1 / num_deck_cards)
                        self.out_string += f"\ntransition {id} {action_id} 1 0 {(1 / num_deck_cards)}"
                    else:
                        # transition to next state
                        new_state_id = self.states_to_id[new_state]
                        # self.P[id, action_id, new_state_id] += (1 / num_deck_cards)
                        self.out_string += f"\ntransition {id} {action_id} {new_state_id} 0 {(1 / num_deck_cards)}"
            else:
                # keep well defined probs for inelgible action
                # self.P[id, action_id, id] = 1.0
                self.out_string += f"\ntransition {id} {action_id} {id} 0 1.0"

    def stop(self, id, state):
        # init reward to the total
        reward = state.get_hand_total()

        # next search for special sequences
        hand = state.hand
        if len(hand) >= 3:
            for i in range(len(hand) - 2):
                if hand[i : i + 3] == tuple(self.sequence):
                    reward += self.bonus
                    break

        # fill matrices
        # self.P[id, 27, 0] = 1.0
        # self.R[id, 27, 0] = reward
        self.out_string += f"\ntransition {id} 27 0 {reward} 1.0"

    def __repr__(self):
        """
        String representation of the MDP object
        
        Returns:
            str: String representation
        """
        return self.out_string

            

def parse_arguments():
    """
    Parse command line arguments for the encoder.
    
    Returns:
        argparse.Namespace: Parsed arguments containing game_config
    """
    parser = argparse.ArgumentParser(
        description='Encoder - Process game configuration file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python encoder.py --game_config data/gameconfig/gameconfig_0.txt
        """
    )
    
    # Required game_config file argument
    parser.add_argument(
        '--game_config',
        type=str,
        required=True,
        help='Path to the game configuration file (required)'
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
    # Check if game_config file exists
    if not os.path.isfile(args.game_config):
        print(f"Error: Game config file '{args.game_config}' does not exist.", file=sys.stderr)
        return False
    
    return True

def load_game_config(game_config_file):
    """
    Load game configuration from file.
    
    Args:
        game_config_file (str): Path to game configuration file
        
    Returns:
        tuple: (threshold, bonus, sequence) where sequence is a list of 3 integers
    """
    try:
        with open(game_config_file, 'r') as f:
            lines = f.readlines()
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Expected format:
        # Line 0: "Configuration:"
        # Line 1: threshold (positive integer)
        # Line 2: bonus (positive integer)
        # Line 3: sequence of 3 space-separated numbers
        
        if len(lines) < 4:
            print(f"Error: Invalid game config file format. Expected at least 4 lines.", file=sys.stderr)
            return None
        
        if lines[0] != "Configuration:":
            print(f"Warning: First line should be 'Configuration:' but found '{lines[0]}'", file=sys.stderr)
        
        # Parse threshold
        threshold = int(lines[1])
        if threshold < 0:
            print(f"Warning: threshold should be positive, but got {threshold}", file=sys.stderr)
        
        # Parse bonus
        bonus = int(lines[2])
        if bonus < 0:
            print(f"Warning: bonus should be positive, but got {bonus}", file=sys.stderr)
        
        # Parse sequence of 3 consecutive numbers
        sequence_parts = lines[3].split()
        if len(sequence_parts) != 3:
            print(f"Error: Expected 3 numbers in sequence, but got {len(sequence_parts)}", file=sys.stderr)
            return None
        
        sequence = [int(num) for num in sequence_parts]
        

        
        return threshold, bonus, sequence
        
    except (ValueError, IOError) as e:
        print(f"Error loading game config file: {e}", file=sys.stderr)
        return None
    

def main():
    """
    Main function to run the encoder.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Load game configuration
    config = load_game_config(args.game_config)
    
    if config is None:
        print("Error: Failed to load game configuration.", file=sys.stderr)
        sys.exit(1)
    
    threshold, bonus, sequence = config
    
    # Create MDP
    mdp = MDP(threshold, bonus, sequence)
    print(mdp)
    

if __name__ == "__main__":
    main()
