# In app_roi.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import classes
from simulation_roi import AdCreativeROI, AdEnvironmentROI
from bandit_roi import EpsilonGreedyBanditROI

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Ad Profit Optimization (ROI Focus)")

# --- Main App Title ---
st.title("Real-Time Ad Profit Optimization: Epsilon-Greedy Bandit")
st.markdown("""
This dashboard simulates an ad campaign focused on **maximizing profit**. 
It compares an **Epsilon-Greedy Bandit** (a reinforcement learning algorithm) 
against a traditional A/B test. 
* **A/B Test:** Splits budget evenly, optimizing for clicks (indirectly for profit).
* **Epsilon-Greedy Bandit:** Learns and adapts to maximize total profit.
""")

# --- Sidebar for Controls ---
st.sidebar.header("Simulation Settings")
num_impressions = st.sidebar.slider(
    "Number of Impressions (per Simulation)",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000
)
num_simulations_to_run = st.sidebar.number_input(
    "Number of Simulations to Run",
    min_value=1,
    max_value=100,
    value=30,  # Default to fewer runs as it might be slower now
    step=5
)

st.sidebar.subheader("Epsilon-Greedy Settings")
use_decay = st.sidebar.checkbox("Use Decaying Epsilon?", value=True)

if use_decay:
    initial_epsilon_val = st.sidebar.slider("Initial Epsilon", 0.0, 1.0, 1.0, 0.01)
    min_epsilon_val = st.sidebar.slider("Minimum Epsilon", 0.0, 0.5, 0.01, 0.005)
    decay_rate_val = st.sidebar.slider("Decay Rate (e.g., 0.999)", 0.99, 0.9999, 0.999, 0.0001, format="%.4f")
else:
    initial_epsilon_val = st.sidebar.slider("Fixed Epsilon", 0.0, 1.0, 0.1, 0.01)
    # For fixed epsilon, min_epsilon and decay_rate are not used by the bandit,
    # but we need to pass them for consistent __init__ call.
    min_epsilon_val = initial_epsilon_val
    decay_rate_val = 1.0  # No decay

# Define my ad creatives with COST and CONVERSION values
# These are the "ground truth" values for the simulation
creatives_roi_config = [
    {'id': 1, 'name': 'Ad A (Low CTR, Low Cost, Low Value)', 'true_ctr': 0.015, 'cost_per_impression': 0.01,
     'click_to_conversion_rate': 0.10, 'value_per_conversion': 1.0},
    {'id': 2, 'name': 'Ad B (High CTR, Med Cost, Med Value)', 'true_ctr': 0.021, 'cost_per_impression': 0.015,
     'click_to_conversion_rate': 0.15, 'value_per_conversion': 1.5},
    {'id': 3, 'name': 'Ad C (Med CTR, High Cost, High Value)', 'true_ctr': 0.018, 'cost_per_impression': 0.02,
     'click_to_conversion_rate': 0.20, 'value_per_conversion': 2.5}  # Potentially most profitable
]
ad_creatives_roi_list = [AdCreativeROI(c['id'], c['true_ctr'], c['cost_per_impression'], c['click_to_conversion_rate'],
                                       c['value_per_conversion']) for c in creatives_roi_config]
ad_names = {c['id']: c['name'] for c in creatives_roi_config}


# --- Main Simulation Logic ---
def run_profit_simulation(creatives_list, total_impressions, epsilon_params):
    """Runs a full simulation for ROI optimization."""
    current_sim_ad_creatives = [
        AdCreativeROI(c['id'], c['true_ctr'], c['cost_per_impression'],
                      c['click_to_conversion_rate'], c['value_per_conversion'])
        for c in creatives_roi_config  # Use the global config to recreate
    ]
    environment = AdEnvironmentROI(current_sim_ad_creatives)  # Use the ROI environment

    bandit = EpsilonGreedyBanditROI(
        creatives=current_sim_ad_creatives,
        initial_epsilon=epsilon_params['initial'],
        use_decaying_epsilon=epsilon_params['use_decay'],
        min_epsilon=epsilon_params['min'],
        decay_rate=epsilon_params['decay_rate']
    )

    # Tracking variables for this single run
    bandit_impressions_run = {ad.creative_id: 0 for ad in current_sim_ad_creatives}
    # bandit_clicks_run = {ad.creative_id: 0 for ad in current_sim_ad_creatives} # We care more about profit now
    bandit_profit_run = {ad.creative_id: 0.0 for ad in current_sim_ad_creatives}
    bandit_profit_history_run = []
    epsilon_history_run = []  # To plot epsilon decay

    ab_profit_run = {ad.creative_id: 0.0 for ad in current_sim_ad_creatives}
    ab_profit_history_run = []
    # ab_impressions_run = {ad.creative_id: 0 for ad in current_sim_ad_creatives} # Can add if needed

    num_creatives_env = len(current_sim_ad_creatives)

    for i in range(total_impressions):
        # --- Bandit's Turn ---
        chosen_ad_bandit = bandit.select_ad()
        # process_impression returns (was_clicked, net_profit)
        _, profit_bandit = chosen_ad_bandit.process_impression()
        bandit.update(chosen_ad_bandit.creative_id, profit_bandit)

        bandit_impressions_run[chosen_ad_bandit.creative_id] += 1
        bandit_profit_run[chosen_ad_bandit.creative_id] += profit_bandit
        bandit_profit_history_run.append(sum(bandit_profit_run.values()))
        epsilon_history_run.append(bandit.get_current_epsilon())

        # --- A/B Test's Turn ---
        # The A/B test still just cycles through ads
        ad_to_show_ab = current_sim_ad_creatives[i % num_creatives_env]
        _, profit_ab = ad_to_show_ab.process_impression()

        ab_profit_run[ad_to_show_ab.creative_id] += profit_ab
        ab_profit_history_run.append(sum(ab_profit_run.values()))

    # Return results for this single run
    return (bandit_impressions_run, bandit_profit_run, bandit_profit_history_run, epsilon_history_run,
            ab_profit_run, ab_profit_history_run)


# --- App Layout ---
if st.sidebar.button("Run ROI Optimization Simulations"):

    # Store epsilon parameters for the bandit
    epsilon_config = {
        'initial': initial_epsilon_val,
        'use_decay': use_decay,
        'min': min_epsilon_val,
        'decay_rate': decay_rate_val
    }

    all_bandit_profit_histories = []
    all_ab_profit_histories = []
    all_epsilon_histories = []  # To store epsilon decay from each run

    # Store final profit and impressions for each ad from ALL simulations
    all_bandit_final_impressions_agg = []  # List of dicts
    all_bandit_final_profit_agg = []  # List of dicts

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(num_simulations_to_run):
        status_text.text(f"Running simulation {i + 1} of {num_simulations_to_run}...")

        b_impressions_s_run, b_profit_s_run, b_profit_hist_s_run, eps_hist_s_run, \
            ab_profit_s_run, ab_profit_hist_s_run = run_profit_simulation(ad_creatives_roi_list, num_impressions,
                                                                          epsilon_config)

        all_bandit_profit_histories.append(b_profit_hist_s_run)
        all_ab_profit_histories.append(ab_profit_hist_s_run)
        all_epsilon_histories.append(eps_hist_s_run)

        all_bandit_final_impressions_agg.append(b_impressions_s_run)
        all_bandit_final_profit_agg.append(b_profit_s_run)

        progress_bar.progress((i + 1) / num_simulations_to_run)

    status_text.text("Simulations complete! Calculating results...")

    np_bandit_profit_histories = np.array(all_bandit_profit_histories)
    np_ab_profit_histories = np.array(all_ab_profit_histories)
    np_epsilon_histories = np.array(all_epsilon_histories)

    mean_bandit_profit_hist = np.mean(np_bandit_profit_histories, axis=0)
    std_bandit_profit_hist = np.std(np_bandit_profit_histories, axis=0)

    mean_ab_profit_hist = np.mean(np_ab_profit_histories, axis=0)
    std_ab_profit_hist = np.std(np_ab_profit_histories, axis=0)

    mean_epsilon_hist = np.mean(np_epsilon_histories, axis=0)  # Average decay path

    st.header("Aggregated Profit Performance Results")
    st.markdown(f"*Based on **{num_simulations_to_run}** simulation runs of **{num_impressions}** impressions each.*")
    st.markdown("---")

    avg_total_bandit_profit = mean_bandit_profit_hist[-1]
    avg_total_ab_profit = mean_ab_profit_hist[-1]

    profit_uplift_percentage = ((avg_total_bandit_profit - avg_total_ab_profit) / abs(
        avg_total_ab_profit) if avg_total_ab_profit != 0 else float('inf')) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Bandit Total Profit", f"${avg_total_bandit_profit:.2f}",
                f"{profit_uplift_percentage:.2f}% vs A/B")
    col2.metric("Avg. A/B Test Total Profit", f"${avg_total_ab_profit:.2f}")
    # col3 can be used for ROAS or other metrics

    st.subheader("Average Cumulative Profit Over Time (with +/- 1 Std Dev)")
    fig, ax = plt.subplots(figsize=(12, 6))
    x_impressions = np.arange(num_impressions)

    ax.plot(x_impressions, mean_bandit_profit_hist, label="Avg. Epsilon-Greedy Bandit Profit", color='green',
            linewidth=2)
    ax.fill_between(x_impressions,
                    mean_bandit_profit_hist - std_bandit_profit_hist,
                    mean_bandit_profit_hist + std_bandit_profit_hist,
                    color='green', alpha=0.2, label='Bandit +/- 1 Std Dev')

    ax.plot(x_impressions, mean_ab_profit_hist, label="Avg. A/B Test Profit", color='blue', linestyle='--', linewidth=2)
    ax.fill_between(x_impressions,
                    mean_ab_profit_hist - std_ab_profit_hist,
                    mean_ab_profit_hist + std_ab_profit_hist,
                    color='blue', alpha=0.1, label='A/B +/- 1 Std Dev')

    ax.set_title("Bandit vs. A/B Test: Average Cumulative Profit")
    ax.set_xlabel("Impressions Shown")
    ax.set_ylabel("Average Total Profit ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")
    st.header("⚙️ Bandit Learning Details")

    col_eps_decay, col_impressions_chart_roi = st.columns(2)

    if epsilon_config['use_decay']:
        with col_eps_decay:
            st.subheader("Average Epsilon Decay Over Time")
            fig_eps, ax_eps = plt.subplots()
            ax_eps.plot(x_impressions, mean_epsilon_hist, label="Average Epsilon Value")
            ax_eps.set_title("Exploration Rate (Epsilon) Over Impressions")
            ax_eps.set_xlabel("Impressions Shown")
            ax_eps.set_ylabel("Epsilon Value")
            ax_eps.legend()
            ax_eps.grid(True, alpha=0.3)
            st.pyplot(fig_eps)
    else:
        with col_eps_decay:
            st.info(f"Fixed Epsilon used: {epsilon_config['initial']:.2f}")

    with col_impressions_chart_roi:
        st.subheader("Average Impressions per Ad Creative (Bandit)")
        # Calculate average impressions per ad for Bandit
        avg_b_impressions_per_ad_roi = {ad_id: 0.0 for ad_id in ad_names.keys()}
        for run_impressions in all_bandit_final_impressions_agg:
            for ad_id, count in run_impressions.items():
                avg_b_impressions_per_ad_roi[ad_id] += count
        for ad_id in avg_b_impressions_per_ad_roi:
            avg_b_impressions_per_ad_roi[ad_id] /= num_simulations_to_run

        avg_impressions_df_roi = pd.DataFrame({
            "Ad Creative": [ad_names[ad_id] for ad_id in avg_b_impressions_per_ad_roi.keys()],
            "Average Impressions": list(avg_b_impressions_per_ad_roi.values())
        })
        st.bar_chart(avg_impressions_df_roi.set_index("Ad Creative"))

else:
    st.info("Adjust the simulation settings in the sidebar and click 'Run ROI Optimization Simulations' to begin.")