# In bandit_roi.py
import numpy as np
from typing import List, Dict
from simulation_roi import AdCreativeROI


class EpsilonGreedyBanditROI:
    """
    Implements the Epsilon-Greedy algorithm for optimizing profit.
    """

    def __init__(self,
                 creatives: List[AdCreativeROI],
                 initial_epsilon: float = 0.1,
                 use_decaying_epsilon: bool = False,
                 min_epsilon: float = 0.01,
                 decay_rate: float = 0.999):  # Decay rate applied per impression
        """
        Initializes the Epsilon-Greedy bandit.

        Args:
            creatives (List[AdCreativeROI]): List of ad creatives.
            initial_epsilon (float): Starting value for epsilon (exploration rate).
                                     If not decaying, this is the fixed epsilon.
            use_decaying_epsilon (bool): If True, epsilon will decay over time.
            min_epsilon (float): The minimum value epsilon can decay to.
            decay_rate (float): Multiplicative factor for decaying epsilon.
        """
        self.creatives = creatives
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon  # Store for reset or reference
        self.use_decaying_epsilon = use_decaying_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

        # Ttrack the number of times each ad was pulled
        # and the sum of rewards received from each ad.
        self.n_pulls: Dict[int, int] = {ad.creative_id: 0 for ad in self.creatives}
        self.total_reward: Dict[int, float] = {ad.creative_id: 0.0 for ad in self.creatives}
        # Q-values store the average reward for each ad.
        self.q_values: Dict[int, float] = {ad.creative_id: 0.0 for ad in self.creatives}

    def select_ad(self) -> AdCreativeROI:
        """
        Selects an ad based on the Epsilon-Greedy strategy.
        """
        # Determine if we explore or exploit
        if np.random.rand() < self.epsilon:
            # Explore: Choose a random ad
            chosen_ad = np.random.choice(self.creatives)
        else:
            # Exploit: Choose the ad with the highest current Q-value
            # Handle cases where some ads might have identical Q-values by breaking ties randomly
            # or by picking the first one found with max Q-value.
            max_q = -float('inf')
            best_ads = []
            for ad in self.creatives:
                if self.q_values[ad.creative_id] > max_q:
                    max_q = self.q_values[ad.creative_id]
                    best_ads = [ad]
                elif self.q_values[ad.creative_id] == max_q:
                    best_ads.append(ad)

            if not best_ads:  # Should not happen if creatives list is not empty
                chosen_ad = np.random.choice(self.creatives)
            else:
                chosen_ad = np.random.choice(best_ads)  # Randomly pick among the best

        # If using decaying epsilon, decay it after making a choice
        if self.use_decaying_epsilon and self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate
            if self.epsilon < self.min_epsilon:  # Ensure it doesn't go below min
                self.epsilon = self.min_epsilon

        return chosen_ad

    def update(self, ad_id: int, reward: float):
        """
        Updates the Q-value for the chosen ad based on the observed reward.

        Args:
            ad_id (int): The ID of the ad that was shown.
            reward (float): The profit received from showing that ad.
        """
        self.n_pulls[ad_id] += 1
        self.total_reward[ad_id] += reward

        # Update the average Q-value for this ad
        self.q_values[ad_id] = self.total_reward[ad_id] / self.n_pulls[ad_id]

    def get_current_epsilon(self) -> float:
        """Returns the current value of epsilon (useful for plotting decay)."""
        return self.epsilon