import argparse
import numpy as np
import sys
import json

class EpisodicMDP:
    """Class to represent and sample from an episodic MDP."""
    
    def __init__(self, mdp_path):
        """
        Initialize the MDP from file.
        
        Args:
            mdp_path: Path to the MDP file
        """
        self.mdp_path = mdp_path
        self.num_states = 0
        self.num_actions = 0
        self.P = None # np array of shape (num_states, num_actions, num_states)
        self.R = None # np array of shape (num_states, num_actions, num_states)
        self.discount = 0.0
        self.mdp_type = ""
        self.terminal_states = []
        
        self._load_mdp()
    
    def _load_mdp(self):
        """
        Load MDP from file.
        
        Args:
            mdp_file (str): Path to MDP file
            
        Returns:
            tuple: MDP data structure
        """
        # Implement MDP loading logic

        # Step 1: Read the file and parse the MDP components
        num_states, num_actions, terminal_states, transitions, mdp_type, gamma = None, None, None, [], None, None

        with open(self.mdp_path, 'r') as f:
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
                        terminal_states = list(map(int, split_line[1:]))
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
            for s in terminal_states:
                P[s, :, s] = 1

        # assert for every state-action pair, sum of transition probabilities is 1
        for s in range(num_states):     
            for a in range(num_actions):
                if not np.isclose(np.sum(P[s, a, :]), 1):
                    print(f"Warning: Transition probabilities for state {s}, action {a} do not sum to 1.", file=sys.stderr)

        # Update the values
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = P.copy()
        self.R = R.copy()
        self.discount = gamma
        self.mdp_type = mdp_type
        self.terminal_states = terminal_states
    
    def sample_episode(self, policy, max_steps=1000):
        """
        Sample a single episode from the MDP.
        
        Args:
            policy: a list of length num_states and values between 0 and num_actions - 1
            max_steps: Maximum number of steps before terminating
            
        Returns:
            episode: List of (state, action, reward) tuples
        """
        episode = []

        # Step 1: Sample a random state from 0 to num_states - 1 and get action
        current_state = np.random.randint(0, self.num_states)
        current_action = policy[current_state]

        # Step 2: Loop until terminal
        step_cnt = 0
        while step_cnt < max_steps:
            next_state = np.random.choice(self.num_states, p=self.P[current_state, current_action, :])
            reward = self.R[current_state, current_action, next_state]
            episode.append((current_state, current_action, reward))

            if next_state in self.terminal_states:
                break
            else:
                current_state = next_state
                current_action = policy[next_state]
                step_cnt += 1
        
        return episode
    
    def sample_multiple_episodes(self, num_episodes, policy, max_steps=1000):
        """
        Sample multiple episodes from the MDP.
        
        Args:
            num_episodes: Number of episodes to sample
            max_steps: Maximum steps per episode
            
        Returns:
            episodes: List of episodes
        """
        episodes = []
        
        for _ in range(num_episodes):
            episodes.append(self.sample_episode(policy, max_steps))
        
        return episodes
    
    def save_episodes_to_file(self, episodes, output_file):
        """
        Save sampled episodes to a file using JSON.
        
        Args:
            episodes: List of episodes to save
            output_file: Path to the output file
        """
        # Convert episodes to a JSON-serializable format
        # Each episode is a list of (state, action, reward) tuples
        episodes_serializable = []
        for episode in episodes:
            episode_list = [[int(s), int(a), float(r)] for s, a, r in episode]
            episodes_serializable.append(episode_list)
        
        with open(output_file, 'w') as f:
            for episode in episodes_serializable:
                f.write(json.dumps(episode) + "\n")
    
    def load_policy(self, policy_path):
        """
        Load policy from file.
        
        Args:
            policy_path: Path to policy file
            
        Returns:
            policy: List of actions for each state
        """
        policy = []
        with open(policy_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    action = int(line)
                    policy.append(action)
        return policy
    

def main():
    parser = argparse.ArgumentParser(description='Sample episodes from an episodic MDP')
    parser.add_argument('--mdp', type=str, required=True, help='Path to MDP file')
    parser.add_argument('--policy', type=str, default=None, help='Path to policy file (optional, random if not provided)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to sample')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output file to save episodes (JSON format)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load MDP
    mdp = EpisodicMDP(args.mdp)
    
    # Load or create policy
    if args.policy:
        policy = mdp.load_policy(args.policy)
        print(f"Loaded policy from {args.policy}")
    else:
        # Use random policy if no policy file provided
        policy = [np.random.randint(0, mdp.num_actions) for _ in range(mdp.num_states)]
        print("Using random policy")

    # print(policy)
    
    # Sample episodes
    episodes = mdp.sample_multiple_episodes(args.episodes, policy, args.max_steps)
    
    # Save episodes to file if output path is provided
    if args.output:
        mdp.save_episodes_to_file(episodes, args.output)
        print(f"Episodes saved to {args.output}")

if __name__ == "__main__":
    main()
