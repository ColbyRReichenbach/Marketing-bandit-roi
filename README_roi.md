# Project: AI-Driven Ad Profit Optimization with Epsilon-Greedy Bandits

**Live App:** [https://marketing-bandit-roi-rzjtitqbqc26utskvbedat.streamlit.app/]

*This project demonstrates an advanced marketing optimization system. It uses an Epsilon-Greedy Multi-Armed Bandit, a reinforcement learning algorithm, to maximize campaign **profit** by dynamically allocating budget to the most profitable ad creatives in real-time. The system's performance is rigorously evaluated against a traditional A/B test through multiple simulation runs, showcasing its ability to deliver superior financial outcomes.*

---

## 1. Project Evolution & Business Problem

This project is an advancement of earlier explorations into ad optimization. While my initial **[Bandit vs. A/B Test for Clicks project]([https://github.com/ColbyRReichenbach/Self-Optimizing-Ad-Campaign])** demonstrated how bandits can increase click-through rates, real-world marketing success is ultimately measured in **profitability and Return on Investment (ROI)**.

Traditional A/B tests, even when optimizing for clicks, can be suboptimal for profit because:
* They don't inherently account for varying costs per impression/click or differing conversion values for different ad creatives.
* The fixed allocation during testing can lead to significant budget spent on creatives that, while perhaps getting clicks, are not driving profitable conversions.

This project tackles the challenge: **How can we build an intelligent system that learns not just which ad gets clicks, but which ad generates the most *profit*, adapting its strategy in real-time under simulated budget considerations?**

## 2. Solution: Epsilon-Greedy Bandit for Profit Maximization

I developed a simulation environment in Python where an **Epsilon-Greedy Bandit** algorithm is tasked with maximizing total profit from a set of ad creatives. Each creative has its own:
* True Click-Through Rate (CTR)
* Cost Per Impression (CPI)
* Click-to-Conversion Rate
* Value Per Conversion

The Epsilon-Greedy algorithm balances:
* **Exploitation:** Choosing the ad with the highest observed average profit.
* **Exploration:** Occasionally choosing a random ad (with probability epsilon, ε) to discover potentially better options.
The system allows for both a fixed epsilon and an **adaptive (decaying) epsilon strategy**, where the exploration rate decreases as the algorithm gains more confidence.

The entire simulation is presented via an interactive **Streamlit** dashboard, which runs multiple simulation instances to provide statistical insights into average performance and variability.

## 3. Key Features

* Interactive Streamlit Dashboard for configuring and running profit-optimization simulations.
* Comparison of an Epsilon-Greedy Bandit against a traditional A/B Test, both optimizing for total profit.
* **User-configurable Epsilon strategy:** Fixed epsilon or decaying epsilon with adjustable parameters (Initial ε, Min ε, Decay Rate) via an "Advanced Settings" panel.
* Calculation and display of **average performance metrics** (Total Profit, Uplift vs. A/B, estimated ROAS) across all simulation runs.
* Visualization of average cumulative profit over time with **standard deviation bands**, clearly showing performance consistency.
* Plots for average impression allocation and epsilon decay.

## 4. Key Results & Insights

The profit-optimizing bandit consistently demonstrates superior financial outcomes. Averaged over **[need to run sims] runs with [# of impressions per sim] impressions each**:

* The Epsilon-Greedy Bandit achieved an **average total profit of $[avg prfot over sims]**, representing an **[uplift % over sims] uplift** compared to the A/B test's average profit of $[a?b profit over sims].
* The dashboard visualizes how the bandit quickly identifies and prioritizes the most profitable ad creative(s), even if they don't have the highest CTR, leading to a significantly better financial return.

(picture of chart after sim)

The "Average Impressions per Ad" chart confirms that the bandit intelligently allocates the budget towards the creatives that yield the highest profit contributions.

## 5. Tech Stack
* **Language:** Python
* **Core Libraries:** Streamlit, Pandas, NumPy, SciPy, Matplotlib
* **Core Concepts:** Reinforcement Learning (Multi-Armed Bandits, Epsilon-Greedy), A/B Testing, Monte Carlo Simulation, ROI Optimization, Marketing Analytics, Statistical Aggregation.

## 6. Project Setup & Usage

**To run this app locally:**
1.  Clone this repository:
    ```bash
    git clone [https://github.com/ColbyRReichenbach/Marketing-bandit-roi]
    cd [Marketing-bandit-roi]
    ```
2.  Set up the Conda environment:
    ```bash
    conda create --name roi-bandit-env python=3.9 # Or your preferred Python version
    conda activate roi-bandit-env
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app_roi.py
    ```
    The application will open in your web browser. Use the sidebar to configure the simulation parameters.

---