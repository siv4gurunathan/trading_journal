"""
Venture Capital Investment Evaluation Model
This script evaluates VC investments using real industry benchmarks.
Startups are classified by stage (Seed / Early / Growth) based on
their number of funding rounds, and each stage gets its own discount
rate, exit multiple and Monte-Carlo outcome probabilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# LOAD AND CLEAN THE DATASET

df = pd.read_excel('/Users/sivvagurunathan/Desktop/investments_VC.xlsx', header=1)
df.columns = df.columns.str.strip()

# keep only the columns we need
df = df[['name', 'funding_total_usd', 'funding_rounds', 'founded_at']]
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
df['founded_at']        = pd.to_datetime(df['founded_at'], errors='coerce')
df['founded_year']      = df['founded_at'].dt.year
df = df.dropna()
df['age'] = 2024 - df['founded_year']

print("Dataset size:", df.shape)
print(df[['name','funding_total_usd','funding_rounds','age']].head())

# STAGE CLASSIFICATION AND VC BENCHMARKS
# VC industry standard: the number of funding rounds a startup has
# completed tells us how mature and risky it is.
#   Seed  (rounds = 1)   - highest risk - 40% discount rate, 10× exit
#   Early (rounds = 2-3) - medium risk  - 30% discount rate,  5× exit
#   Growth(rounds ≥ 4)   - lower risk   - 20% discount rate,  3× exit
# Higher discount rates for earlier stages reflect the greater
# probability of failure and the illiquidity of VC investments.
# Exit multiples are higher for earlier stages to compensate for that risk.

def classify_stage(rounds):
    """Classify startup stage from number of funding rounds."""
    if rounds == 1:
        return 'Seed'
    elif rounds <= 3:
        return 'Early'
    else:
        return 'Growth'

# VC benchmark parameters per stage
stage_params = {
    'Seed':   {'rate': 0.40, 'exit_multiple': 10},
    'Early':  {'rate': 0.30, 'exit_multiple':  5},
    'Growth': {'rate': 0.20, 'exit_multiple':  3},
}

df['stage'] = df['funding_rounds'].apply(classify_stage)

print("\nStartups per stage:")
print(df['stage'].value_counts())

# CASH FLOW STRUCTURE
# Standard VC cash flow model (exit horizon = 6 years):
#   Year 0 : invest  → -funding_total_usd
#   Years 1-5 : no cash flows (startup is building the business)
#   Year 6 : exit    → investment × exit_multiple
# This matches how VC funds actually work: capital is deployed upfront
# and the return is realised at exit (IPO or acquisition), typically
# 5-7 years later.  We use 6 years as the standard horizon.

EXIT_YEAR = 6  # VC exit horizon in years

def make_cashflows(investment, exit_multiple):
    cf = [0.0] * (EXIT_YEAR + 1)
    cf[0] = -investment
    cf[EXIT_YEAR] = investment * exit_multiple
    return cf

# NPV FUNCTION

def NPV(cf, r):
    N, t = 0, 0
    for c in cf:
        N += c / (1 + r) ** t
        t += 1
    return N

# IRR FUNCTION  

def IRR(cf):
    r1, r2 = 0.05, 0.50   # wider bracket to cover high VC returns
    n1 = NPV(cf, r1)
    n2 = NPV(cf, r2)
    for _ in range(200):
        denom = n1 - n2
        if abs(denom) < 1e-14:
            break
        r3 = r1 - (r1 - r2) * n1 / denom
        n3 = NPV(cf, r3)
        if abs(n2) > abs(n1):
            n2, r2 = n3, r3
        else:
            n1, r1 = n3, r3
        if abs(n3) < 1e-9:
            break
    return r3

# COMPUTE NPV AND IRR FOR EVERY STARTUP IN THE DATASET
# We use the funding_total_usd as the investment amount,
# the stage-specific discount rate and exit multiple,
# and the 6-year cash flow structure.
# cap investment at $500M to remove a handful of extreme outliers
df = df[df['funding_total_usd'] <= 500_000_000].copy()

npv_list = []
irr_list = []

for _, row in df.iterrows():
    params     = stage_params[row['stage']]
    r          = params['rate']
    mult       = params['exit_multiple']
    investment = row['funding_total_usd']

    cf  = make_cashflows(investment, mult)
    npv_list.append(NPV(cf, r))
    irr_list.append(IRR(cf))

df['NPV'] = npv_list
df['IRR'] = irr_list

print("\n Per-startup NPV/IRR (first 5 rows)")
print(df[['name','stage','funding_total_usd','NPV','IRR']].head().to_string(index=False))

# MONTE-CARLO SIMULATION
# Real VC outcome distribution (industry benchmark):
#   60% of startups fail completely     - 0× return
#   25% achieve moderate success        - 2× return
#   10% achieve strong success          - 5× return
#    5% achieve exceptional success     - 20× return
# We simulate 10,000 scenarios using np.random.choice with these
# probabilities, then compute NPV and IRR for each scenario.
# The investment used is the median from the dataset.

nsim       = 10_000
outcomes   = [0, 2, 5, 20]           # possible exit multiples
probs      = [0.60, 0.25, 0.10, 0.05] # VC industry outcome probabilities
med_invest = df['funding_total_usd'].median()  # representative investment

# draw a random exit multiple for each simulation
sim_multiples = np.random.choice(outcomes, size=nsim, p=probs)

NPVs = np.zeros(nsim)
IRRs = np.zeros(nsim)

for i in range(nsim):
    # use Seed-stage discount rate for MC
    cf       = make_cashflows(med_invest, sim_multiples[i])
    NPVs[i]  = NPV(cf, 0.40)
    # IRR is only meaningful when there is a positive exit
    if sim_multiples[i] > 0:
        IRRs[i] = IRR(cf)
    else:
        IRRs[i] = -1.0   # mark total failures as -100% return

print("\n Monte-Carlo Results (10,000 simulations)")
print(f"  Median investment used : ${med_invest:,.0f}")
print(f"  Mean NPV               : ${NPVs.mean():,.0f}")
print(f"  Std  NPV               : ${NPVs.std():,.0f}")
print(f"  P(NPV > 0)             : {(NPVs > 0).mean()*100:.1f}%")
print(f"  Mean IRR (non-zero)    : {IRRs[IRRs > 0].mean()*100:.1f}%")
print(f"  P(total failure)       : {(sim_multiples == 0).mean()*100:.1f}%")

# STAGE-BY-STAGE COMPARISON

print("\n Summary by Stage")
print(f"  {'Stage':<8}  {'Count':>6}  {'Mean NPV ($M)':>14}  {'Mean IRR':>9}  {'P(NPV>0)':>9}")

for stage in ['Seed', 'Early', 'Growth']:
    sub = df[df['stage'] == stage]
    mean_npv = sub['NPV'].mean() / 1e6
    mean_irr = sub['IRR'].mean() * 100
    p_pos    = (sub['NPV'] > 0).mean() * 100
    print(f"  {stage:<8}  {len(sub):>6}  {mean_npv:>14.2f}  {mean_irr:>8.1f}%  {p_pos:>8.1f}%")

# PORTFOLIO OPTIMISATION
# We treat Seed, Early, and Growth stages as three separate "assets"
# and find the allocation that maximises the portfolio Sharpe ratio.
# Returns per asset are drawn from the stage-specific NPV distribution.
# build a (1000 × 3) returns matrix: one column per stage
R = np.column_stack([
    df[df['stage'] == s]['NPV'].sample(1000, replace=True).values / 1e6
    for s in ['Seed', 'Early', 'Growth']
])

def sharpe(w, r_i):
    """Negative Sharpe ratio (minimise to maximise)."""
    Rp = r_i @ w
    return -Rp.mean() / Rp.std()

w0          = np.array([1/3, 1/3, 1/3])
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds      = [(0, 1)] * 3

result = minimize(sharpe, w0, args=(R,), constraints=constraints, bounds=bounds)

print("\n Portfolio Optimisation (Seed / Early / Growth)")
for stage, w in zip(['Seed', 'Early', 'Growth'], result.x):
    print(f"  {stage:<8} weight: {w:.4f}")
print(f"  Sharpe ratio  : {-result.fun:.4f}")

# PLOTS
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("VC Investment Evaluation – Industry Benchmark Model", fontsize=13)

# Monte-Carlo NPV distribution 
axes[0, 0].hist(NPVs / 1e6, bins=40, color='steelblue', edgecolor='white')
axes[0, 0].axvline(0, color='red', linestyle='--', label='NPV = 0')
axes[0, 0].set_title("Figure 1:Monte-Carlo NPV Distribution")
axes[0, 0].set_xlabel("NPV ($M)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()

# Monte-Carlo outcome probabilities 
labels = ['Failure\n(0×)', 'Moderate\n(2×)', 'Strong\n(5×)', 'Exceptional\n(20×)']
counts = [(sim_multiples == m).sum() for m in outcomes]
colors = ['tomato', 'gold', 'steelblue', 'seagreen']
axes[0, 1].bar(labels, counts, color=colors, edgecolor='white')
axes[0, 1].set_title("Figure 2:Monte-Carlo Outcome Distribution")
axes[0, 1].set_ylabel("Number of Simulations")

# NPV by stage  
seed_npv   = df[df['stage'] == 'Seed']['NPV'].values   / 1e6
early_npv  = df[df['stage'] == 'Early']['NPV'].values  / 1e6
growth_npv = df[df['stage'] == 'Growth']['NPV'].values / 1e6
axes[1, 0].boxplot([seed_npv, early_npv, growth_npv],
                   tick_labels=['Seed', 'Early', 'Growth'],
                   patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].set_title("Figure 3:NPV by Stage ($M)")
axes[1, 0].set_ylabel("NPV ($M)")

# Portfolio weights
stages = ['Seed', 'Early', 'Growth']
colors2 = ['tomato', 'gold', 'seagreen']
axes[1, 1].bar(stages, result.x, color=colors2, edgecolor='white')
axes[1, 1].set_title("Figure 4:Optimal Portfolio Weights")
axes[1, 1].set_ylabel("Weight")
plt.tight_layout()
plt.show()
