"""
Synthetic CRM dataset generator for the Retention Decision Agent.

ALL DATA PRODUCED HERE IS SIMULATED. It is designed to contain planted,
known treatment effects so the agent's output can be validated against
ground truth. Nothing here is real customer data.

Each account has:
  - CRM-style features (tenure, plan, MRR, usage, tickets, NPS, ...)
  - a historical retention campaign assignment (randomised, 4 arms)
  - an observed churn outcome from that campaign
  - hidden ground-truth churn probability under each action (for evaluation only)
"""
import numpy as np
import pandas as pd

ACTIONS = ["none", "csm_call", "discount_15", "training"]
ACTION_COST = {"none": 0.0, "csm_call": 60.0, "discount_15": None, "training": 25.0}
# discount cost is computed per account: 15% of MRR for 12 months


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=15000, seed=42):
    rng = np.random.default_rng(seed)

    plan = rng.choice(["Basic", "Pro", "Enterprise"], size=n, p=[0.5, 0.35, 0.15])
    seats = np.where(plan == "Basic", rng.integers(1, 6, n),
             np.where(plan == "Pro", rng.integers(5, 30, n), rng.integers(30, 200, n)))
    price_per_seat = np.where(plan == "Basic", 12, np.where(plan == "Pro", 25, 40))
    mrr = (seats * price_per_seat * rng.uniform(0.85, 1.15, n)).round(0)

    tenure_months = rng.integers(1, 60, n)
    logins_30d = np.clip(rng.poisson(np.where(plan == "Basic", 6, 14), n)
                         + rng.integers(-3, 4, n), 0, None)
    days_since_last_login = np.clip((30 - logins_30d * 1.5 + rng.normal(0, 5, n)).round(), 0, 90)
    support_tickets_90d = rng.poisson(1.2, n) + (rng.random(n) < 0.15) * rng.integers(3, 8, n)
    nps = np.clip((7 - 0.6 * support_tickets_90d + 0.08 * logins_30d + rng.normal(0, 2, n)).round(), 0, 10)
    discount_pct = rng.choice([0, 0, 0, 5, 10, 20], n)
    csm_touches_90d = rng.poisson(np.where(plan == "Enterprise", 3, 0.7), n)
    contract_months_left = rng.integers(0, 12, n)
    industry = rng.choice(["Retail", "Fintech", "Healthcare", "Education", "Manufacturing"], n)

    # ---------- hidden ground truth ----------
    # baseline churn logit: low usage, tickets, low NPS, short tenure, price pressure
    base_logit = (-1.6
                  + 0.045 * days_since_last_login
                  - 0.05 * logins_30d
                  + 0.25 * support_tickets_90d
                  - 0.18 * nps
                  - 0.012 * tenure_months
                  + 0.25 * (plan == "Basic")
                  - 0.15 * (plan == "Enterprise")
                  - 0.06 * csm_touches_90d
                  + rng.normal(0, 0.4, n))
    p_none = _sigmoid(base_logit)

    # treatment effects (change in churn probability; negative = good)
    # CSM call: helps frustrated-but-engaged accounts; HURTS disengaged low-value
    # accounts (sleeping dogs: the call reminds them to cancel)
    frustrated_engaged = (support_tickets_90d >= 3) & (logins_30d >= 6)
    sleeping_dog = (logins_30d <= 3) & (plan == "Basic") & (nps <= 6)
    tau_call = (-0.22 * frustrated_engaged
                - 0.05 * (~frustrated_engaged & ~sleeping_dog)
                + 0.12 * sleeping_dog)

    # Discount: helps price-sensitive accounts (Basic/Pro, no existing discount,
    # moderate usage); does almost nothing for accounts with existing discounts
    price_sensitive = (plan != "Enterprise") & (discount_pct == 0) & (logins_30d >= 4)
    tau_disc = (-0.20 * price_sensitive
                - 0.03 * (~price_sensitive & (discount_pct == 0))
                + 0.00 * (discount_pct > 0))

    # Training: helps new, low-usage accounts (onboarding failure)
    onboarding_gap = (tenure_months <= 9) & (logins_30d <= 8)
    tau_train = -0.18 * onboarding_gap - 0.02 * (~onboarding_gap)

    p_call = np.clip(p_none + tau_call, 0.01, 0.99)
    p_disc = np.clip(p_none + tau_disc, 0.01, 0.99)
    p_train = np.clip(p_none + tau_train, 0.01, 0.99)

    # ---------- historical randomised campaign ----------
    assigned = rng.choice(ACTIONS, size=n, p=[0.4, 0.2, 0.2, 0.2])
    p_assigned = np.select(
        [assigned == "none", assigned == "csm_call", assigned == "discount_15", assigned == "training"],
        [p_none, p_call, p_disc, p_train])
    churned = (rng.random(n) < p_assigned).astype(int)

    df = pd.DataFrame({
        "account_id": [f"ACC-{i:05d}" for i in range(n)],
        "industry": industry, "plan": plan, "seats": seats, "mrr": mrr,
        "tenure_months": tenure_months, "logins_30d": logins_30d,
        "days_since_last_login": days_since_last_login.astype(int),
        "support_tickets_90d": support_tickets_90d, "nps": nps.astype(int),
        "discount_pct": discount_pct, "csm_touches_90d": csm_touches_90d,
        "contract_months_left": contract_months_left,
        "campaign_action": assigned, "churned": churned,
        # ground truth columns -- used ONLY for evaluation, never for training
        "gt_p_none": p_none.round(4), "gt_p_csm_call": p_call.round(4),
        "gt_p_discount_15": p_disc.round(4), "gt_p_training": p_train.round(4),
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("crm_accounts_synthetic.csv", index=False)
    print(f"Wrote {len(df)} synthetic accounts")
    print(df.groupby("campaign_action")["churned"].mean().round(3))
