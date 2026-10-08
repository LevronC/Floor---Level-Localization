import math

def safe_log(x):
    """Safely computes the natural logarithm, returning negative infinity for zero or negative values."""
    if x <= 0.0:
        return float('-inf')
    return math.log(x)

def normal_pdf(x, mu, sigma):
    """Computes the probability density function (PDF) of a standard normal distribution."""
    if sigma <= 0.0:
        return 1.0 if x == mu else 0.0
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normal_cdf(x, mu, sigma):
    """Computes the cumulative distribution function (CDF) of a standard normal distribution."""
    if sigma <= 0.0:
        return 1.0 if x >= mu else 0.0
    # math.erf handles the error function needed for the normal CDF
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def truncated_normal_pdf(x, mu, sigma, a, b):
    """
    Computes the PDF of a truncated normal distribution on the interval [a, b].
    If x is outside [a, b], returns 0.0.
    """
    if x < a or x > b:
        return 0.0
    
    pdf_val = normal_pdf(x, mu, sigma)
    cdf_b = normal_cdf(b, mu, sigma)
    cdf_a = normal_cdf(a, mu, sigma)
    denom = cdf_b - cdf_a
    
    if denom <= 0.0:
        # Fallback to normal PDF or uniform distribution if limits are degenerate
        return pdf_val if pdf_val > 0.0 else 1.0 / (b - a)
    
    return pdf_val / denom

class HMM:
    def __init__(self, num_floors, pi, A, emission_fn):
        """
        Initializes the Hidden Markov Model.
        
        Parameters:
        - num_floors (int): F, the number of floors S = {1, ..., F}
        - pi (list): A list of length F, where pi[j-1] is the initial probability of floor j
        - A (list of lists): F x F transition matrix where A[i-1][j-1] is transition i -> j
        - emission_fn (callable): A function emission_fn(state, obs) returning b_state(obs)
        """
        self.num_floors = num_floors
        self.pi = pi
        self.A = A
        self.emission_fn = emission_fn
        
        # Verify transition matrix dimensions
        assert len(pi) == num_floors, "pi size must match number of floors"
        assert len(A) == num_floors, "A row size must match number of floors"
        for row in A:
            assert len(row) == num_floors, "A column size must match number of floors"

    def get_predecessors(self, j):
        """
        Returns the set of valid predecessor states N(j) for state j under the adjacency constraint.
        State j is 1-indexed (1 to F). Predecessors are also 1-indexed.
        N(j) = { i in S : A[i-1][j-1] > 0 and |i - j| <= 1 }
        """
        N_j = []
        # Check only adjacent states under the constraint |i - j| <= 1
        for i in [j - 1, j, j + 1]:
            if 1 <= i <= self.num_floors:
                if self.A[i - 1][j - 1] > 0.0:
                    N_j.append(i)
        return N_j

def viterbi_decode(hmm, observations):
    """
    Finds the most likely sequence of floors (1-indexed) given the observations.
    Implements Viterbi Decoding in log-space for numerical stability.
    
    Returns:
    - path (list): The sequence of floors (1-indexed)
    - delta_table (list of dicts): The delta values (log-probabilities) for each timestep
    - psi_table (list of dicts): The backpointers for each timestep
    """
    T = len(observations)
    if T == 0:
        return [], [], []
    
    F = hmm.num_floors
    
    # delta[t][j] will store log-probability of reaching state j at timestep t (1-indexed)
    delta = [{} for _ in range(T + 1)]
    # psi[t][j] will store predecessor state index (1-indexed)
    psi = [{} for _ in range(T + 1)]
    
    # Step 1: Initialization (t = 1)
    o_1 = observations[0]
    for j in range(1, F + 1):
        b_val = hmm.emission_fn(j, o_1)
        pi_val = hmm.pi[j - 1]
        delta[1][j] = safe_log(pi_val) + safe_log(b_val)
        psi[1][j] = None
        
    # Step 2: Recursion (t = 2 to T)
    for t in range(2, T + 1):
        o_t = observations[t - 1]
        for j in range(1, F + 1):
            N_j = hmm.get_predecessors(j)
            
            # Find max and argmax over i in N(j)
            max_val = float('-inf')
            best_i = None
            
            for i in N_j:
                log_trans = safe_log(hmm.A[i - 1][j - 1])
                val = delta[t - 1][i] + log_trans
                if val > max_val:
                    max_val = val
                    best_i = i
            
            # If no valid predecessor (all had log probability -inf), fall back to j or first in N(j)
            if best_i is None:
                if N_j:
                    best_i = N_j[0]
                else:
                    best_i = j
            
            b_val = hmm.emission_fn(j, o_t)
            delta[t][j] = max_val + safe_log(b_val)
            psi[t][j] = best_i
            
    # Step 3: Termination
    max_final_val = float('-inf')
    q_star_T = None
    for j in range(1, F + 1):
        if delta[T][j] > max_final_val:
            max_final_val = delta[T][j]
            q_star_T = j
            
    # Fallback if all paths have -inf probability
    if q_star_T is None:
        q_star_T = 1
        
    # Step 4: Backtracking
    q_star = [0] * (T + 1)
    q_star[T] = q_star_T
    
    for t in range(T - 1, 0, -1):
        q_star[t] = psi[t + 1][q_star[t + 1]]
        
    # Return 1-based path from t=1 to T (so we slice from 1)
    return q_star[1:], delta[1:], psi[1:]

def naive_snap(hmm, observations):
    """
    Baseline Algorithm: Naive Snap-to-Nearest-Floor.
    For each timestep independently, picks the floor j with the highest emission likelihood b_j(o_t).
    
    Returns:
    - path (list): The sequence of floors (1-indexed)
    """
    path = []
    F = hmm.num_floors
    for o_t in observations:
        best_j = None
        max_b = float('-inf')
        for j in range(1, F + 1):
            b_val = hmm.emission_fn(j, o_t)
            if b_val > max_b:
                max_b = b_val
                best_j = j
        # Fallback if all are 0 or -inf
        if best_j is None:
            best_j = 1
        path.append(best_j)
    return path
