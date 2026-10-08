import random
from hmm_model import HMM, viterbi_decode, naive_snap, truncated_normal_pdf

def sample_truncated_normal(mu, sigma, a, b):
    """Generates a sample from a truncated normal distribution using rejection sampling."""
    while True:
        x = random.gauss(mu, sigma)
        if a <= x <= b:
            return x

class GarageSimulator:
    def __init__(self, num_floors=5, pi=None, A=None, floor_params=None):
        """
        Initializes the garage simulation environment.
        
        Parameters:
        - num_floors (int): Number of floors F
        - pi (list): Initial probability distribution over floors (1-indexed). Defaults to entering on floor 1.
        - A (list of lists): Transition matrix. Defaults to a standard banded matrix.
        - floor_params (dict): Sensor emission parameters for each floor.
        """
        self.num_floors = num_floors
        
        # 1. Initial floor probability (default: starts on floor 1 with 90% prob, floor 2 with 10%)
        if pi is None:
            self.pi = [0.0] * num_floors
            self.pi[0] = 0.9
            if num_floors > 1:
                self.pi[1] = 0.1
        else:
            self.pi = pi
            
        # 2. Transition matrix A (default: high self-transition probability, banded adjacencies)
        if A is None:
            self.A = [[0.0] * num_floors for _ in range(num_floors)]
            for i in range(num_floors):
                if num_floors == 1:
                    self.A[i][i] = 1.0
                elif i == 0:
                    self.A[i][i] = 0.8
                    self.A[i][i + 1] = 0.2
                elif i == num_floors - 1:
                    self.A[i][i - 1] = 0.2
                    self.A[i][i] = 0.8
                else:
                    self.A[i][i - 1] = 0.1
                    self.A[i][i] = 0.7
                    self.A[i][i + 1] = 0.2
        else:
            self.A = A
            
        # 3. Floor emission parameters
        # Barometric pressure: Floor 1 is at 0m altitude. 3m per floor.
        # Expected pressure: P_j = 1013.25 - 0.36 * (j-1) hPa.
        # GPS signal strength: Floors < F are indoors (low signal, mean 0.1). Roof is floor F (high signal, mean 0.85).
        # Turn count: Indoor floors have more turns due to navigating ramps (mean 0.8). Roof has fewer turns (mean 0.3).
        if floor_params is None:
            self.floor_params = {}
            for j in range(1, num_floors + 1):
                self.floor_params[j] = {
                    'pressure_mean': 1013.25 - 0.36 * (j - 1),
                    'gps_mean': 0.85 if j == num_floors else 0.10,
                    'turns_mean': 0.30 if j == num_floors else 0.80
                }
        else:
            self.floor_params = floor_params

    def generate_trip(self, T, gps_dropout_prob=0.2, noise_level="medium"):
        """
        Generates a ground truth path of length T and a corresponding noisy multi-channel observation sequence.
        
        Noise levels:
        - "low": very small sensor noise, high accuracy expected
        - "medium": typical sensor noise, realistic challenge
        - "high": severe sensor noise, baseline should fail frequently
        
        Returns:
        - true_path (list of int): List of ground truth floors (1-indexed)
        - observations (list of tuples): List of (pressure, gps, turns, is_dropout) readings
        """
        # Set standard deviations based on noise level
        if noise_level == "low":
            sigma_P = 0.05
            sigma_G = 0.05
            sigma_C = 0.08
        elif noise_level == "medium":
            sigma_P = 0.20
            sigma_G = 0.15
            sigma_C = 0.25
        elif noise_level == "high":
            sigma_P = 0.50
            sigma_G = 0.30
            sigma_C = 0.50
        else:
            raise ValueError(f"Unknown noise level: {noise_level}")
            
        # Generate Ground Truth Path Q (Markov chain)
        true_path = []
        
        # Step 1: Initial state
        curr_floor = random.choices(range(1, self.num_floors + 1), weights=self.pi)[0]
        true_path.append(curr_floor)
        
        # Step 2: Transitions
        for t in range(2, T + 1):
            transitions = self.A[curr_floor - 1]
            curr_floor = random.choices(range(1, self.num_floors + 1), weights=transitions)[0]
            true_path.append(curr_floor)
            
        # Generate Observations O
        observations = []
        for curr_floor in true_path:
            params = self.floor_params[curr_floor]
            
            # Channel 1: Pressure (truncated in [950, 1050])
            P = sample_truncated_normal(params['pressure_mean'], sigma_P, 950.0, 1050.0)
            
            # Channel 2: GPS (truncated in [0.0, 1.0]) with dropout
            is_dropout = random.random() < gps_dropout_prob
            if is_dropout:
                G = 0.0
            else:
                G = sample_truncated_normal(params['gps_mean'], sigma_G, 0.0, 1.0)
                
            # Channel 3: Turn count (truncated in [0.0, 10.0])
            C = sample_truncated_normal(params['turns_mean'], sigma_C, 0.0, 10.0)
            
            observations.append((P, G, C, is_dropout))
            
        return true_path, observations

    def create_hmm_model(self, noise_level="medium", gps_dropout_prob=0.2):
        """Creates an HMM instance representing the simulated environment."""
        if noise_level == "low":
            sigma_P = 0.05
            sigma_G = 0.05
            sigma_C = 0.08
        elif noise_level == "medium":
            sigma_P = 0.20
            sigma_G = 0.15
            sigma_C = 0.25
        elif noise_level == "high":
            sigma_P = 0.50
            sigma_G = 0.30
            sigma_C = 0.50
        else:
            raise ValueError(f"Unknown noise level: {noise_level}")

        def continuous_emission_fn(state, obs):
            """
            Computes b_state(obs) where obs is (pressure, gps, turns, is_dropout).
            Utilizes conditional independence: b_state(obs) = P(p|s) * P(g|s) * P(c|s)
            """
            p_val, g_val, c_val, is_dropout = obs
            params = self.floor_params[state]
            
            # Pressure likelihood (truncated Gaussian on [950, 1050])
            p_lik = truncated_normal_pdf(p_val, params['pressure_mean'], sigma_P, 950.0, 1050.0)
            
            # GPS likelihood (truncated Gaussian on [0, 1]). Flat on dropout.
            if is_dropout or g_val == 0.0:
                g_lik = 1.0  # Flat likelihood over the domain [0, 1], treated as uninformative constant
            else:
                g_lik = truncated_normal_pdf(g_val, params['gps_mean'], sigma_G, 0.0, 1.0)
                
            # Turn count likelihood (truncated Gaussian on [0, 10])
            c_lik = truncated_normal_pdf(c_val, params['turns_mean'], sigma_C, 0.0, 10.0)
            
            return p_lik * g_lik * c_lik

        return HMM(self.num_floors, self.pi, self.A, continuous_emission_fn)

def is_sequence_physically_valid(path):
    """Checks if the floor sequence contains only valid adjacent transitions (difference <= 1)."""
    if len(path) <= 1:
        return True
    for t in range(1, len(path)):
        if abs(path[t] - path[t - 1]) > 1:
            return False
    return True

def calculate_metrics(true_path, predicted_path):
    """Computes the per-timestep accuracy and checks exact sequence matches and physical validity."""
    T = len(true_path)
    correct_steps = sum(1 for t in range(T) if true_path[t] == predicted_path[t])
    timestep_acc = correct_steps / T
    exact_match = 1.0 if true_path == predicted_path else 0.0
    validity = 1.0 if is_sequence_physically_valid(predicted_path) else 0.0
    return timestep_acc, exact_match, validity
