# In simulation_roi.py
import numpy as np

from typing import List


class AdCreativeROI:  # Renamed for clarity
    """
    Represents a single ad creative with CTR, cost, and conversion value.
    """

    def __init__(self,
                 creative_id: int,
                 true_ctr: float,
                 cost_per_impression: float,  # Cost for showing the ad once
                 click_to_conversion_rate: float,  # Probability a click leads to a sale
                 value_per_conversion: float  # Revenue from one sale
                 ):
        self.creative_id = creative_id
        self.true_ctr = true_ctr  # Probability of a click
        self.cost_per_impression = cost_per_impression # How much paid everytime ad was shown
        self.click_to_conversion_rate = click_to_conversion_rate # If a user does click - at what rate do they convert?
        self.value_per_conversion = value_per_conversion # Revenue from a single conversion.

    def process_impression(self) -> tuple[int, float]:
        """
        Simulates showing the ad to a single user and calculates the outcome.

        Returns:
            tuple[int, float]:
                - clicks (0 or 1)
                - profit (revenue from conversion - cost_per_impression) can be negative if click but no conversion
        """
        # Cost is always incurred for showing the ad
        impression_cost = self.cost_per_impression

        # Did the user click?
        was_clicked = np.random.binomial(1, self.true_ctr)

        profit_from_this_impression = 0.0

        if was_clicked == 1:
            # If clicked, did they convert?
            converted = np.random.binomial(1, self.click_to_conversion_rate)
            if converted == 1:
                profit_from_this_impression = self.value_per_conversion

        # Net profit for this single impression
        net_profit = profit_from_this_impression - impression_cost

        return was_clicked, net_profit


# Hold AdCreativeROI objects
class AdEnvironmentROI:  # Renamed for clarity
    def __init__(self, creatives: List[AdCreativeROI]):
        self.creatives = creatives

    def get_creatives(self) -> List[AdCreativeROI]:
        return self.creatives