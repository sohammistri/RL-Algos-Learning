import argparse
from multiprocessing import Value
from typing import Optional
import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor, as_completed # <-- Change this
import numba
from sample_episodic_mdps import MDP
from tqdm import tqdm, trange
from typing import Optional, List, Tuple

@numba.jit(nopython=True)
def update_state_values_forward_view_numba(
    s_arr: np.ndarray,
    ns_arr: np.ndarray,
    r_arr: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lambda_: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A Numba-jitted function to calculate the TD(lambda) updates for a single episode.
    This function returns a list of (state, delta) tuples.
    """
    L = s_arr.shape[0]
    states = np.empty(L, dtype=np.int64)
    td_errors = np.empty(L, dtype=np.float64)

    # for every state in the episode, calculate its G_lambda return
    for i in range(L):
        s1, r1, ns1 = int(s_arr[i]), float(r_arr[i]), int(ns_arr[i])

        # Calculate n-step returns
        G_n = np.empty(L, dtype=np.float64)
        c = r1
        G_n[0] = (r1 + gamma * values[ns1])
        gamma_pow = gamma

        for j in range(i + 1, L):
            r2, s2 = float(r_arr[j]), int(ns_arr[j])
            c += (gamma_pow * r2)
            gamma_pow *= gamma
            G_n[j - i] = (c + gamma_pow * values[s2])

        # Calculate G_lambda
        G_lambda, cum_wts, cur_wt = 0.0, 0.0, 1.0 - lambda_
        for k, g in enumerate(G_n):
            if k == len(G_n) - 1:
                G_lambda += (1.0 - cum_wts) * g
            else:
                G_lambda += cur_wt * g
                cum_wts += cur_wt
                cur_wt *= lambda_

        td_error = G_lambda - values[s1]
        states[i], td_errors[i] = s1, td_error

    return states, td_errors

def td_lambda_forward_policy_evaluation_worker(
    mdp: MDP,
    policy_dist: np.ndarray,
    values: np.ndarray,
    max_steps: int,
    gamma: float,
    lambda_: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A task for a single worker thread to generate an episode and calculate updates.
    """
    np.random.seed(seed)
    num_states = mdp.num_states
    terminal_states = set(mdp.terminal_states)

    while True:
        s1 = np.random.randint(0, num_states)
        if s1 not in terminal_states:
            break

    s_arr, ns_arr, r_arr = [], [], []
    for _ in range(max_steps):
        # sample an action from a policy
        a1 = np.searchsorted(np.cumsum(policy_dist[s1, :]), np.random.rand())
        # sample next state
        s2 = np.searchsorted(np.cumsum(mdp.P[s1, a1, :]), np.random.rand())
        # get reward
        r1 = mdp.R[s1, a1, s2]
        s_arr.append(s1)
        ns_arr.append(s2)
        r_arr.append(r1)
        if s2 in terminal_states:
            break
        s1 = s2
    
    # Use the Numba-optimized function to get updates
    s_arr_np, ns_arr_np, r_arr_np = np.array(s_arr, dtype=np.int16), np.array(ns_arr, dtype=np.int16), np.array(r_arr, dtype=np.float32)
    return update_state_values_forward_view_numba(s_arr_np, ns_arr_np, r_arr_np, values, gamma, lambda_)

@numba.jit(nopython=True)
def td_lambda_backward_policy_evaluation_numba_worker(
    P: np.ndarray,
    R: np.ndarray,
    policy_dist: np.ndarray,
    values: np.ndarray,
    terminal_states: set,  # Boolean array indicating terminal states
    num_states: int,
    num_actions: int,
    num_iters: int,
    max_steps: int,
    gamma: float,
    alpha: float,
    lambda_: float,
    lr_mode: int,  # 0: fixed, 1: adagrad, 2: rmsprop
    beta: float,
    seed: int,
) -> np.ndarray:
    """
    Numba-optimized backward view TD(lambda) worker.
    lr_mode: 0=fixed, 1=adagrad, 2=rmsprop
    """
    if seed is not None:
        np.random.seed(seed)
    
    adagrad_accum = np.zeros(num_states, dtype=np.float64)
    rmsprop_accum = np.zeros(num_states, dtype=np.float64)
    eps = 1e-8
    
    values = values.copy()  # Work on a copy
    
    for _ in range(num_iters):
        # reset eligibility traces per episode
        eligibility_traces = np.zeros(num_states, dtype=np.float64)

        # Sample initial non-terminal state
        while True:
            s1 = np.random.randint(0, num_states)
            if s1 not in terminal_states:
                break
        
        for _ in range(max_steps):
            # Sample action from policy
            policy_cumsum = np.cumsum(policy_dist[s1, :])
            a1 = np.searchsorted(policy_cumsum, np.random.rand())
            
            # Sample next state
            trans_cumsum = np.cumsum(P[s1, a1, :])
            s2 = np.searchsorted(trans_cumsum, np.random.rand())
            
            # Get reward
            r1 = R[s1, a1, s2]
            
            # update eligibility trace (accumulating traces)
            # e_t = gamma * lambda * e_{t-1}; then e_t[s_t] += 1
            for i in range(num_states):
                eligibility_traces[i] = eligibility_traces[i] * (gamma * lambda_)
            eligibility_traces[s1] += 1.0

            # TD error (note: if s2 is terminal and you want V(terminal)=0,
            # ensure values[s2] has the desired terminal value)
            td_error = r1 + gamma * values[s2] - values[s1]

            # gradient vector w.r.t. values: g_i = td_error * eligibility_traces[i]
            # (we'll use this to update accumulators and the values)
            # compute g and squared g
            g = np.zeros(num_states, dtype=np.float64)
            g2 = np.zeros(num_states, dtype=np.float64)
            for i in range(num_states):
                g[i] = td_error * eligibility_traces[i]
                g2[i] = g[i] * g[i]

            if lr_mode == 0:
                # fixed LR -> scalar alpha
                for i in range(num_states):
                    values[i] += alpha * g[i]

            elif lr_mode == 1:
                # Adagrad: accumulate g^2 and scale per-parameter
                for i in range(num_states):
                    adagrad_accum[i] += g2[i]
                    adjusted_lr = alpha / math.sqrt(adagrad_accum[i] + eps)
                    values[i] += adjusted_lr * g[i]

            else:
                # RMSProp: exponential moving average of g^2
                for i in range(num_states):
                    rmsprop_accum[i] = beta * rmsprop_accum[i] + (1.0 - beta) * g2[i]
                    adjusted_lr = alpha / math.sqrt(rmsprop_accum[i] + eps)
                    values[i] += adjusted_lr * g[i]
            
            # Check if terminal state
            if s2 in terminal_states:
                break
            
            s1 = s2
    
    return values

def td_lambda_policy_evaluation(
    mdp: MDP,
    policy: Optional[object] = None,
    td_mode: str = None,
    num_iters: int = 10,
    max_steps: int = 100,
    alpha: float = 0.1,
    lambda_: float = 0.1,
    lr: str = "fixed",
    beta: float = 0.99,
    seed: int = None,
    num_threads: int = 4,
):
    """
    Optimized TD(lambda) policy evaluation using Numba and threading.
    """
    if policy is None:
        raise ValueError("Policy needs to be provided in order to be evaluated")

    rng = np.random.default_rng(seed)
    num_states, num_actions, gamma = mdp.num_states, mdp.num_actions, mdp.discount

    values = np.zeros(num_states, dtype=np.float64)
    adagrad_accum = np.zeros(num_states, dtype=np.float64)
    rmsprop_accum = np.zeros(num_states, dtype=np.float64)

    policy_arr = np.asarray(policy)
    if policy_arr.ndim == 1:
        policy_dist = np.zeros((num_states, num_actions), dtype=np.float64)
        for s in range(num_states):
            a = int(policy_arr[s])
            policy_dist[s, a] = 1.0
    elif policy_arr.ndim == 2:
        policy_dist = policy_arr.astype(np.float64, copy=False)
    else:
        raise ValueError("Policy must be 1D (actions) or 2D (distributions).")

    if td_mode == "forward":
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for _ in trange(num_iters // num_threads, desc="TD lambda Evaluation"):
                futures = [
                    executor.submit(
                        td_lambda_forward_policy_evaluation_worker,
                        mdp,
                        policy_dist,
                        values.copy(), # Pass a copy of values to avoid race conditions during read
                        max_steps,
                        gamma,
                        lambda_,
                        rng.integers(2**32) # Pass a new seed for each worker
                    )
                    for _ in range(num_threads)
                ]

                for future in as_completed(futures):
                    updates = future.result()
                    for s1, td_error in zip(updates[0], updates[1]):
                        if lr == "fixed":
                            alpha_updated = alpha
                        elif lr == "adagrad":
                            adagrad_accum[s1] += td_error * td_error
                            alpha_updated = alpha / math.sqrt(adagrad_accum[s1] + 1e-8)
                        elif lr == "rmsprop":
                            rmsprop_accum[s1] = beta * rmsprop_accum[s1] + (1.0 - beta) * (td_error * td_error)
                            alpha_updated = alpha / math.sqrt(rmsprop_accum[s1] + 1e-8)
                        else:
                            raise ValueError("Unknown lr mode. Use one of: fixed, adagrad, rmsprop.")
                        
                        values[s1] += alpha_updated * td_error
        return values

    elif td_mode == "backward":
        # Convert learning rate mode to integer
        if lr == "fixed":
            lr_mode = 0
        elif lr == "adagrad":
            lr_mode = 1
        elif lr == "rmsprop":
            lr_mode = 2
        else:
            raise ValueError("Unknown lr mode. Use one of: fixed, adagrad, rmsprop.")
        
        # Prepare data for numba function
        P = mdp.P.copy()
        R = mdp.R.copy()
        num_states = mdp.num_states
        num_actions = mdp.num_actions
        
        terminal_states = set(mdp.terminal_states)
        # values = td_lambda_backward_policy_evaluation_numba_worker(
        #     P,
        #     R,
        #     policy_dist,
        #     values,  # Each worker gets its own copy
        #     terminal_states,
        #     num_states,
        #     num_actions,
        #     num_iters,
        #     max_steps,
        #     gamma,
        #     alpha,
        #     lambda_,
        #     lr_mode,
        #     beta,
        #     seed
        # )

        # return values
        
        # Parallelize using ProcessPoolExecutor
        per_worker_iters = max(1, num_iters)
        num_actual_workers = min(num_threads, num_iters)
        
        with ProcessPoolExecutor(max_workers=num_actual_workers) as executor:
            futures = []
            for i in range(num_actual_workers):
                future = executor.submit(
                    td_lambda_backward_policy_evaluation_numba_worker,
                    P,
                    R,
                    policy_dist,
                    values.copy(),  # Each worker gets its own copy
                    terminal_states,
                    num_states,
                    num_actions,
                    per_worker_iters,
                    max_steps,
                    gamma,
                    alpha,
                    lambda_,
                    lr_mode,
                    beta,
                    rng.integers(2**32) # Pass a new seed for each worker
                )
                futures.append(future)
            
            # Collect results and average them
            all_values = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="TD(lambda) backward Evaluation", unit="worker"):
                worker_values = future.result()
                all_values.append(worker_values)
        
        # Average the values from all workers
        if all_values:
            values = np.mean(all_values, axis=0)
        
        return values

@numba.njit
def update_action_values_forward_view_numba(
    s_arr,  # np.ndarray (int64)
    a_arr,  # np.ndarray (int64)
    r_arr,  # np.ndarray (float64)
    ns_arr, # np.ndarray (int64)
    na_arr, # np.ndarray (int64)
    q_values, # np.ndarray (float64), shape (S, A)
    is_terminal, # np.ndarray (bool) shape (S,)
    gamma,    # float
    lambda_,  # float
    L         # int, actual episode length
):
    """
    Numba-compatible forward-view TD(lambda) for a single episode.
    Returns (states[0:L], actions[0:L], G_lambdas[0:L])
    """
    states = np.empty(L, dtype=np.int64)
    actions = np.empty(L, dtype=np.int64)
    G_lambdas = np.empty(L, dtype=np.float64)

    for i in range(L):
        s1 = int(s_arr[i])
        a1 = int(a_arr[i])
        r1 = float(r_arr[i])
        ns1 = int(ns_arr[i])
        na1 = int(na_arr[i])

        # Calculate n-step returns
        n_returns = L - i
        G_n = np.empty(n_returns, dtype=np.float64)
        
        # 1-step return
        if is_terminal[ns1]:
            G_n[0] = r1
        else:
            G_n[0] = r1 + gamma * q_values[ns1, na1]
        
        # Multi-step returns
        c = r1
        gamma_pow = gamma

        for j in range(i + 1, L):
            r2 = float(r_arr[j])
            ns2 = int(ns_arr[j])
            na2 = int(na_arr[j])
            
            c += gamma_pow * r2
            gamma_pow *= gamma
            
            if is_terminal[ns2]:
                G_n[j - i] = c
            else:
                G_n[j - i] = c + gamma_pow * q_values[ns2, na2]

        # Calculate G_lambda with proper weighting
        G_lambda = 0.0
        cum_wts = 0.0
        
        for k in range(n_returns - 1):
            wt = (1.0 - lambda_) * (lambda_ ** k)
            G_lambda += wt * G_n[k]
            cum_wts += wt
        
        # Last return gets remaining weight
        G_lambda += (1.0 - cum_wts) * G_n[n_returns - 1]

        states[i] = s1
        actions[i] = a1
        G_lambdas[i] = G_lambda

    return states, actions, G_lambdas

@numba.njit
def epsilon_greedy_action(q_values, state, num_actions, epsilon):
    """Select action using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.randint(0, num_actions)
    else:
        return np.argmax(q_values[state, :num_actions])

@numba.njit
def sample_next_state(P, s, a, num_states):
    """Sample next state from transition probability distribution."""
    probs = P[s, a, :num_states]
    # Numba-safe categorical sample using inverse transform sampling
    cumsum = 0.0
    rnd = np.random.rand()
    for i in range(num_states):
        cumsum += probs[i]
        if rnd < cumsum:
            return i
    return num_states - 1  # fallback, should not happen if probs sum to 1
    
@numba.njit
def td_lambda_forward_control_numba_worker(
    P,                    # (S, A, S)
    R,                    # (S, A, S)
    action_values,        # (S, A)
    is_terminal,          # (S,)
    num_states,
    num_actions,
    num_iters,
    max_steps,
    gamma,
    alpha,
    lambda_,
    epsilon_start,
    epsilon_min,
    epsilon_decay,
    lr_mode,  # 0 fixed, 1 adagrad, 2 rmsprop
    beta,
    seed,
):
    if seed is not None:
        np.random.seed(seed)

    adagrad_accum = np.zeros((num_states, num_actions), dtype=np.float64)
    rmsprop_accum = np.zeros((num_states, num_actions), dtype=np.float64)
    eps = 1e-8

    q_values = action_values.copy()

    # Preallocate arrays for one episode
    s_arr = np.empty(max_steps, dtype=np.int64)
    a_arr = np.empty(max_steps, dtype=np.int64)
    r_arr = np.empty(max_steps, dtype=np.float64)
    ns_arr = np.empty(max_steps, dtype=np.int64)
    na_arr = np.empty(max_steps, dtype=np.int64)

    for k in range(num_iters):
        # Improved epsilon schedule
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** k))

        # Sample initial non-terminal state
        s1 = np.random.randint(0, num_states)
        attempts = 0
        while is_terminal[s1] and attempts < 100:
            s1 = np.random.randint(0, num_states)
            attempts += 1
        
        if is_terminal[s1]:
            continue

        a1 = epsilon_greedy_action(q_values, s1, num_actions, epsilon)

        # Generate episode
        L = 0
        while L < max_steps:
            s2 = sample_next_state(P, s1, a1, num_states)
            rew = R[s1, a1, s2]

            a2 = epsilon_greedy_action(q_values, s2, num_actions, epsilon)

            # Store transition
            s_arr[L] = s1
            a_arr[L] = a1
            r_arr[L] = rew
            ns_arr[L] = s2
            na_arr[L] = a2
            L += 1

            if is_terminal[s2]:
                break

            s1 = s2
            a1 = a2

        if L == 0:
            continue

        # Compute G_lambda targets via forward view
        states, actions, G_lambdas = update_action_values_forward_view_numba(
            s_arr, a_arr, r_arr, ns_arr, na_arr, q_values, is_terminal, gamma, lambda_, L
        )

        # Apply updates
        for i in range(L):
            s = int(states[i])
            a = int(actions[i])
            G_lambda = float(G_lambdas[i])
            td_error = G_lambda - q_values[s, a]

            if lr_mode == 0:
                new_alpha = alpha
            elif lr_mode == 1:
                adagrad_accum[s, a] += td_error * td_error
                new_alpha = alpha / np.sqrt(adagrad_accum[s, a] + eps)
            else:  # rmsprop
                rmsprop_accum[s, a] = beta * rmsprop_accum[s, a] + (1.0 - beta) * (td_error * td_error)
                new_alpha = alpha / np.sqrt(rmsprop_accum[s, a] + eps)

            q_values[s, a] += new_alpha * td_error

    return q_values

@numba.njit
def td_lambda_backward_control_numba_worker(
    P,                    # (S, A, S)
    R,                    # (S, A, S)
    action_values,        # (S, A)
    is_terminal,          # (S,)
    num_states,
    num_actions,
    num_iters,
    max_steps,
    gamma,
    alpha,
    lambda_,
    epsilon_min,
    lr_mode,  # 0 fixed, 1 adagrad, 2 rmsprop
    beta,
    seed,
):
    if seed is not None:
        np.random.seed(seed)

    adagrad_accum = np.zeros((num_states, num_actions), dtype=np.float64)
    rmsprop_accum = np.zeros((num_states, num_actions), dtype=np.float64)
    eps = 1e-8

    q_values = action_values.copy()

    for k in range(num_iters):
        # Reset eligibility traces per episode
        eligibility_traces = np.zeros((num_states, num_actions), dtype=np.float64)
        
        epsilon = max(epsilon_min, 1 / (k + 1))

        # Sample initial non-terminal state
        s1 = np.random.randint(0, num_states)
        while is_terminal[s1]:
            s1 = np.random.randint(0, num_states)
        
        a1 = epsilon_greedy_action(q_values, s1, num_actions, epsilon)

        # Generate episode
        L = 0
        while L < max_steps:
            s2 = sample_next_state(P, s1, a1, num_states)
            rew = R[s1, a1, s2]

            a2 = epsilon_greedy_action(q_values, s2, num_actions, epsilon)

            # Update eligibility traces (accumulating traces)
            eligibility_traces[s1, a1] += 1.0

            # Calculate TD error
            if is_terminal[s2]:
                delta = rew - q_values[s1, a1]
            else:
                delta = rew + gamma * q_values[s2, a2] - q_values[s1, a1]

            td_error = delta * eligibility_traces
            td_error_cur_state = td_error[s1, a1]

            # Get learning rate
            # For backward view with eligibility traces, we use fixed alpha or 
            # adapt based on current state-action only
            if lr_mode == 0:
                new_alpha = alpha
            elif lr_mode == 1:
                adagrad_accum[s1, a1] += td_error_cur_state * td_error_cur_state
                new_alpha = alpha / np.sqrt(adagrad_accum[s1, a1] + eps)
            else:  # rmsprop
                rmsprop_accum[s1, a1] = beta * rmsprop_accum[s1, a1] + (1.0 - beta) * (td_error_cur_state * td_error_cur_state)
                new_alpha = alpha / np.sqrt(rmsprop_accum[s1, a1] + eps)

            # Update Q-values and eligibility traces for all state-action pairs using eligibility traces
            q_values += new_alpha * td_error * eligibility_traces
            eligibility_traces *= (gamma * lambda_)

            L += 1
            
            if is_terminal[s2]:
                break

            s1 = s2
            a1 = a2

    return q_values

def sarsa_lambda_epsilon_greedy(
    mdp,  # MDP object
    td_mode: str,
    num_iters: int,
    max_steps: int,
    alpha: float,
    lambda_: float,
    epsilon_min: float = 0.01,
    lr: str = "fixed",
    beta: float = 0.9,
    seed: int = None,
    num_threads: int = 1,
):
    """
    Optimized TD(lambda) epsilon-greedy control algorithm using Numba and multiprocessing.
    
    Args:
        mdp: MDP object with P, R, num_states, num_actions, discount, terminal_states
        td_mode: "forward" or "backward" view
        num_iters: Total number of episodes
        max_steps: Maximum steps per episode
        alpha: Learning rate
        lambda_: Trace decay parameter
        epsilon_min: Minimum exploration rate
        epsilon_start: Starting exploration rate
        epsilon_decay: Decay rate for epsilon
        lr: Learning rate adaptation ("fixed", "adagrad", "rmsprop")
        beta: Beta parameter for RMSProp
        seed: Random seed
        num_threads: Number of parallel workers
    
    Returns:
        q_values: Learned Q-values (S, A)
        optimal_values: Optimal state values (S,)
        optimal_actions: Optimal policy (S,)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    num_states, num_actions, gamma = mdp.num_states, mdp.num_actions, mdp.discount

    action_values = np.zeros((num_states, num_actions), dtype=np.float64)

    # Convert learning rate mode to integer
    lr_mode_map = {"fixed": 0, "adagrad": 1, "rmsprop": 2}
    if lr not in lr_mode_map:
        raise ValueError("Unknown lr mode. Use one of: fixed, adagrad, rmsprop.")
    lr_mode = lr_mode_map[lr]

    # Prepare data for numba function
    P = mdp.P.astype(np.float64)
    R = mdp.R.astype(np.float64)
    
    terminal_states = set(mdp.terminal_states)
    is_terminal = np.zeros(num_states, dtype=np.bool_)
    for s in terminal_states:
        if 0 <= s < num_states:
            is_terminal[s] = True

    # Select worker function
    if td_mode == "forward":
        worker_func = td_lambda_forward_control_numba_worker
    elif td_mode == "backward":
        worker_func = td_lambda_backward_control_numba_worker
    else:
        raise ValueError("td_mode must be 'forward' or 'backward'")

    # Run workers in parallel
    if num_threads > 1:
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            iters_per_worker = num_iters // num_threads
            
            for i in range(num_threads):
                # Each worker gets its own seed
                worker_seed = rng.integers(2**31) if seed is not None else None
                
                future = executor.submit(
                    worker_func,
                    P,
                    R,
                    action_values.copy(),
                    is_terminal,
                    num_states,
                    num_actions,
                    iters_per_worker,
                    max_steps,
                    gamma,
                    alpha,
                    lambda_,
                    epsilon_min,
                    lr_mode,
                    beta,
                    worker_seed
                )
                futures.append(future)
            
            # Collect results - take the one with best average Q-value
            all_q_values = []
            desc = f"SARSA(λ) {td_mode} view"
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="worker"):
                worker_q_values = future.result()
                all_q_values.append(worker_q_values)
            
            # Instead of averaging (which is incorrect), select best performing model
            # or use ensemble voting for policy
            best_idx = np.argmax([np.max(q) for q in all_q_values])
            q_values = all_q_values[best_idx]
    else:
        # Single-threaded execution
        worker_seed = rng.integers(2**31) if seed is not None else None
        q_values = worker_func(
            P, R, action_values, is_terminal,
            num_states, num_actions, num_iters, max_steps,
            gamma, alpha, lambda_, epsilon_min,
            lr_mode, beta, worker_seed
        )
    
    # Extract optimal policy and values
    optimal_values = np.max(q_values, axis=1)
    optimal_actions = np.argmax(q_values, axis=1)
    
    return q_values, optimal_values, optimal_actions

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TD(0) boilerplate for MDPs")
    parser.add_argument('--mode', type=str, required=True, help='Mode to run TD lambda algorithm (eval or ctrl)', choices=['eval', 'ctrl'], default='ctrl')
    parser.add_argument('--td-mode', type=str, help='Mode to run TD lambda algorithm (forward or backward)', choices=['forward', 'backward'], default='backward')
    parser.add_argument('--ctrl-mode', type=str, help='Mode to run TD lambda ctrl algorithm (exploring_starts, epsilon_greedy_sarsa or epsilon_greedy_q_learning)', choices=['exploring_starts', 'epsilon_greedy_sarsa', 'epsilon_greedy_q_learning'], default='exploring_starts')
    parser.add_argument("--mdp", type=str, required=True, help="Path to MDP file")
    parser.add_argument('--policy', type=str, default=None, help='Path to policy file (optional)')
    parser.add_argument('--eval-sol', type=str, default=None, help='Path to true value-action solution file for infinity-norm evaluation')
    parser.add_argument('--ctrl-sol', type=str, default=None, help='Path to true value-policy solution file for infinity-norm evaluation in control mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--num-iters", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Number of states per episode")
    parser.add_argument("--alpha", type=float, default=0.1, help="TD(0) learning rate alpha")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1, help="TD(lambda) lambda parameter to give weighst to individual returns")
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
    print(args)

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
        value_functions = td_lambda_policy_evaluation(
            mdp,
            policy,
            args.td_mode,
            args.num_iters,
            args.max_steps,
            args.alpha,
            args.lambda_,
            args.lr,
            args.beta,
            args.seed,
            args.num_workers
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
        if args.ctrl_mode == "epsilon_greedy_sarsa":
            Q_values, optimal_values, optimal_policy = sarsa_lambda_epsilon_greedy(
                mdp,
                args.td_mode,
                args.num_iters,
                args.max_steps,
                args.alpha,
                args.lambda_,
                args.epsilon_min,
                args.lr,
                args.beta,
                args.seed,
                args.num_workers
            )
        # elif args.ctrl_mode == "epsilon_greedy_q_learning":
        #     optimal_values, optimal_policy, Q_values = off_policy_td0_control_q_learning_epsilon_greedy(
        #         mdp,
        #         args.num_iters,
        #         args.max_steps,
        #         args.alpha,
        #         args.lr,
        #         args.beta,
        #         args.epsilon_min,
        #         args.num_workers,
        #         args.seed,
        #     )

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
                # For each state where the actions differ, show the true action Q val and calculated action Q val
                for s in range(len(optimal_policy)):
                    calc_action = optimal_policy[s]
                    true_action = true_policy[s]
                    if calc_action != true_action:
                        calc_q = Q_values[s, calc_action]
                        true_q = Q_values[s, true_action]
                        print(f"{s}: true_a={true_action} true_q={true_q:.6f} calc_a={calc_action} calc_q={calc_q:.6f}")

        if args.verbose:
            for v, a in zip(optimal_values, optimal_policy):
                print(f"{v:.6f} {a}")

if __name__ == "__main__":
    main()

