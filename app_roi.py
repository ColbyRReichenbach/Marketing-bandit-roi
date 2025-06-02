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
It compares an **Epsilon-Greedy Bandit** (a reinforcement learning algorithm that learns and adapts) 
against a traditional A/B test (which splits budget evenly).
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
    value=30,
    step=5
)

st.sidebar.markdown("---")  # Separator

# --- Simplified Epsilon Settings with Advanced Expander ---
st.sidebar.subheader("Epsilon-Greedy Strategy")
# Default to decaying epsilon
default_use_decay = True
default_initial_epsilon = 0.3  # Start with more exploration
default_min_epsilon = 0.01  # Keep a tiny bit of exploration
default_decay_rate = 0.999  # Moderate decay

# Use a checkbox that, if unchecked, reveals the advanced settings
# For the primary view, we'll just state the strategy.
# The actual values will come from the advanced expander or these defaults.

show_advanced_epsilon = st.sidebar.checkbox("Customize Epsilon Settings (Advanced)", value=False)

if show_advanced_epsilon:
    st.sidebar.markdown("### Advanced Epsilon Configuration")
    adv_use_decay = st.sidebar.checkbox("Use Decaying Epsilon?", value=default_use_decay, key="adv_decay_main")
    if adv_use_decay:
        adv_initial_epsilon_val = st.sidebar.slider("Initial Epsilon", 0.0, 1.0, default_initial_epsilon, 0.01,
                                                    key="adv_initial_eps_decay")
        adv_min_epsilon_val = st.sidebar.slider("Minimum Epsilon", 0.0, 0.5, default_min_epsilon, 0.005,
                                                key="adv_min_eps_decay")
        adv_decay_rate_val = st.sidebar.slider("Decay Rate (e.g., 0.999)", 0.9900, 0.9999, default_decay_rate, 0.0001,
                                               format="%.4f", key="adv_decay_rate_val")
    else:
        adv_initial_epsilon_val = st.sidebar.slider("Fixed Epsilon", 0.0, 1.0, 0.1, 0.01,
                                                    key="adv_fixed_eps_val")  # Default fixed epsilon if not decaying
        # For fixed epsilon, min_epsilon and decay_rate are ignored by the bandit logic if use_decay is False
        adv_min_epsilon_val = adv_initial_epsilon_val
        adv_decay_rate_val = 1.0  # No decay
else:
    # If advanced settings are hidden, use the predefined defaults for decaying epsilon
    adv_use_decay = default_use_decay
    adv_initial_epsilon_val = default_initial_epsilon
    adv_min_epsilon_val = default_min_epsilon
    adv_decay_rate_val = default_decay_rate
    st.sidebar.info(
        f"Using default adaptive epsilon strategy (initial: {default_initial_epsilon}, decay: {default_decay_rate}, min: {default_min_epsilon}).")

# --- Ad Creative Configuration ---
# SCENARIO: Ad C is the clear winner, Ad A is okay, Ad B is a trap
creatives_roi_config = [
    {'id': 1, 'name': 'Ad A (Balanced)',
     'true_ctr': 0.020, 'cost_per_impression': 0.015,
     'click_to_conversion_rate': 0.15, 'value_per_conversion': 1.8},
    # Expected Profit/Imp: (0.020 * 0.15 * 1.8) - 0.015 = 0.0054 - 0.015 = -0.0096

    {'id': 2, 'name': 'Ad B (High CTR Trap)',
     'true_ctr': 0.030, 'cost_per_impression': 0.025,
     'click_to_conversion_rate': 0.08, 'value_per_conversion': 1.0},
    # Expected Profit/Imp: (0.030 * 0.08 * 1.0) - 0.025 = 0.0024 - 0.025 = -0.0226

    {'id': 3, 'name': 'Ad C (The Profit Winner)',
     'true_ctr': 0.018, 'cost_per_impression': 0.010,
     'click_to_conversion_rate': 0.25, 'value_per_conversion': 3.00}
    # Expected Profit/Imp: (0.018 * 0.25 * 3.00) - 0.010 = 0.0135 - 0.010 = +0.0035
]
ad_creatives_roi_list = [AdCreativeROI(c['id'], c['true_ctr'], c['cost_per_impression'], c['click_to_conversion_rate'],
                                       c['value_per_conversion']) for c in creatives_roi_config]
ad_names = {c['id']: c['name'] for c in creatives_roi_config}


# --- Main Simulation Logic (Function definition remains the same) ---
def run_profit_simulation(creatives_list_config, total_impressions, epsilon_params_dict):

    current_sim_ad_creatives = [
        AdCreativeROI(c['id'], c['true_ctr'], c['cost_per_impression'],
                      c['click_to_conversion_rate'], c['value_per_conversion'])
        for c in creatives_list_config
    ]
    # environment = AdEnvironmentROI(current_sim_ad_creatives) # Not strictly needed if bandit takes creatives directly

    bandit = EpsilonGreedyBanditROI(
        creatives=current_sim_ad_creatives,
        initial_epsilon=epsilon_params_dict['initial'],
        use_decaying_epsilon=epsilon_params_dict['use_decay'],
        min_epsilon=epsilon_params_dict['min'],
        decay_rate=epsilon_params_dict['decay_rate']
    )

    bandit_impressions_run = {ad.creative_id: 0 for ad in current_sim_ad_creatives}
    bandit_profit_run = {ad.creative_id: 0.0 for ad in current_sim_ad_creatives}
    bandit_profit_history_run = []
    epsilon_history_run = []

    ab_profit_run = {ad.creative_id: 0.0 for ad in current_sim_ad_creatives}
    ab_profit_history_run = []

    num_creatives_env = len(current_sim_ad_creatives)

    for i in range(total_impressions):
        chosen_ad_bandit = bandit.select_ad()
        _, profit_bandit = chosen_ad_bandit.process_impression()
        bandit.update(chosen_ad_bandit.creative_id, profit_bandit)

        bandit_impressions_run[chosen_ad_bandit.creative_id] += 1
        bandit_profit_run[chosen_ad_bandit.creative_id] += profit_bandit
        bandit_profit_history_run.append(sum(bandit_profit_run.values()))
        epsilon_history_run.append(bandit.get_current_epsilon())  # Assumes get_current_epsilon() method exists

        ad_to_show_ab = current_sim_ad_creatives[i % num_creatives_env]
        _, profit_ab = ad_to_show_ab.process_impression()

        ab_profit_run[ad_to_show_ab.creative_id] += profit_ab
        ab_profit_history_run.append(sum(ab_profit_run.values()))

    return (bandit_impressions_run, bandit_profit_run, bandit_profit_history_run, epsilon_history_run,
            ab_profit_run, ab_profit_history_run)


# --- App Layout ---
if st.sidebar.button("Run ROI Optimization Simulations"):

    # Prepare epsilon_config based on current UI state
    epsilon_config_to_use = {
        'initial': adv_initial_epsilon_val,
        'use_decay': adv_use_decay,
        'min': adv_min_epsilon_val,
        'decay_rate': adv_decay_rate_val
    }

    all_bandit_profit_histories = []
    all_ab_profit_histories = []
    all_epsilon_histories = []
    all_bandit_final_impressions_agg = []
    all_bandit_final_profit_agg = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(num_simulations_to_run):
        status_text.text(f"Running simulation {i + 1} of {num_simulations_to_run}...")

        b_impressions_s_run, b_profit_s_run, b_profit_hist_s_run, eps_hist_s_run, \
            ab_profit_s_run, ab_profit_hist_s_run = run_profit_simulation(creatives_roi_config, num_impressions,
                                                                          epsilon_config_to_use)

        all_bandit_profit_histories.append(b_profit_hist_s_run)
        all_ab_profit_histories.append(ab_profit_hist_s_run)
        all_epsilon_histories.append(eps_hist_s_run)
        all_bandit_final_impressions_agg.append(b_impressions_s_run)
        all_bandit_final_profit_agg.append(b_profit_s_run)

        progress_bar.progress((i + 1) / num_simulations_to_run)

    status_text.success(f"Simulations complete! ({num_simulations_to_run} runs)")

    np_bandit_profit_histories = np.array(all_bandit_profit_histories)
    np_ab_profit_histories = np.array(all_ab_profit_histories)
    np_epsilon_histories = np.array(all_epsilon_histories)

    mean_bandit_profit_hist = np.mean(np_bandit_profit_histories, axis=0)
    std_bandit_profit_hist = np.std(np_bandit_profit_histories, axis=0)

    mean_ab_profit_hist = np.mean(np_ab_profit_histories, axis=0)
    std_ab_profit_hist = np.std(np_ab_profit_histories, axis=0)

    mean_epsilon_hist = np.mean(np_epsilon_histories, axis=0)

    st.header("Aggregated Profit Performance Results")
    st.markdown(f"*Based on **{num_simulations_to_run}** simulation runs of **{num_impressions}** impressions each.*")
    st.markdown("---")

    avg_total_bandit_profit = mean_bandit_profit_hist[-1]
    avg_total_ab_profit = mean_ab_profit_hist[-1]

    # Handle division by zero or cases where A/B profit is zero or positive
    if avg_total_ab_profit == 0 and avg_total_bandit_profit > 0:
        profit_uplift_percentage = float('inf')  # Or some large number / "Significant improvement"
    elif avg_total_ab_profit == 0 and avg_total_bandit_profit == 0:
        profit_uplift_percentage = 0.0
    elif avg_total_ab_profit == 0 and avg_total_bandit_profit < 0:
        profit_uplift_percentage = float('-inf')  # Or "Significant underperformance if A/B was 0"
    elif avg_total_ab_profit < 0:  # If A/B test is losing money
        profit_uplift_percentage = ((avg_total_bandit_profit - avg_total_ab_profit) / abs(avg_total_ab_profit)) * 100
    else:  # A/B test profit is positive
        profit_uplift_percentage = ((avg_total_bandit_profit - avg_total_ab_profit) / avg_total_ab_profit) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Bandit Total Profit", f"${avg_total_bandit_profit:.2f}",
                f"{profit_uplift_percentage:.2f}% vs A/B")
    col2.metric("Avg. A/B Test Total Profit", f"${avg_total_ab_profit:.2f}")
    # col3 can be used for ROAS. Need total cost.
    avg_cpi_overall = np.mean([c['cost_per_impression'] for c in creatives_roi_config])
    total_cost_campaign = num_impressions * avg_cpi_overall  # Simplified total cost

    bandit_roas = (
                              avg_total_bandit_profit + total_cost_campaign) / total_cost_campaign if total_cost_campaign > 0 else 0
    col3.metric("Est. Bandit ROAS", f"{bandit_roas:.2f}x")

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
    st.header("Bandit Learning Details")

    col_eps_decay_viz, col_impressions_chart_roi_viz = st.columns(2)

    # Get the epsilon config that was actually used for the simulation run
    current_epsilon_config_display = {
        'initial': adv_initial_epsilon_val,
        'use_decay': adv_use_decay,
        'min': adv_min_epsilon_val,
        'decay_rate': adv_decay_rate_val
    }

    if current_epsilon_config_display['use_decay']:
        with col_eps_decay_viz:
            st.subheader("Average Epsilon Decay Over Time")
            fig_eps, ax_eps = plt.subplots()
            ax_eps.plot(x_impressions, mean_epsilon_hist,
                        label=f"Avg. Epsilon (Initial: {current_epsilon_config_display['initial']:.2f})")
            ax_eps.set_title("Exploration Rate (Epsilon) Over Impressions")
            ax_eps.set_xlabel("Impressions Shown")
            ax_eps.set_ylabel("Epsilon Value")
            ax_eps.legend()
            ax_eps.grid(True, alpha=0.3)
            st.pyplot(fig_eps)
    else:
        with col_eps_decay_viz:
            st.info(f"Fixed Epsilon used for these simulations: {current_epsilon_config_display['initial']:.2f}")

    with col_impressions_chart_roi_viz:
        st.subheader("Average Impressions per Ad Creative (Bandit)")
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