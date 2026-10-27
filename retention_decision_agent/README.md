# Retention Decision Agent — upload-your-own-data version

An AI agent that takes a CRM's churn score one step further: for every account it estimates
**which retention action would change the outcome, what it is worth, and who should be left alone**,
then builds a budget-constrained plan with a plain-language reason for each recommendation.

You upload the data. Nothing is built in except an optional template generator.

## 1. Install and run

```
pip install -r requirements.txt
python -m streamlit run app.py
```
Opens at http://localhost:8501. To run again later: open cmd, `cd` to this folder, run the second line.

## 2. What to upload

**File 1 — historical retention campaign (required).** One row per account that was in a past campaign.
Accounts must have been assigned to actions *including a control group labelled `none`*, and you must know
whether each one churned afterwards.

| Column | Meaning |
|---|---|
| `account_id` | unique id |
| `plan` | text category, e.g. Basic / Pro / Enterprise |
| `mrr` | monthly recurring revenue |
| `tenure_months` | months as a customer |
| `logins_30d` | logins in the last 30 days |
| `days_since_last_login` | days since last login |
| `support_tickets_90d` | tickets in the last 90 days |
| `nps` | last NPS response (0–10) |
| `discount_pct` | current discount % |
| `csm_touches_90d` | CSM contacts in the last 90 days |
| `contract_months_left` | months to renewal |
| `campaign_action` | `none` for control; any other label for a treatment (e.g. `csm_call`, `discount_15`, `training`). Each label needs at least 100 rows; a few thousand per label gives usable models. |
| `churned` | 1 if the account churned after the campaign, else 0 |

**File 2 — current accounts to score (optional).** Same columns minus `campaign_action` and `churned`.
If you skip it, the app scores a 30% holdout of File 1.

Treatment labels are free text; the app asks for the cost of each one on the Run page
(fixed $ per contact plus any % of 12-month MRR given away).

**No data yet?** Page 1 has a "Build template CSV" button that generates a synthetic file in the correct
layout with planted treatment effects. It is for learning the format and testing the app; it is not real data.
Files produced this way contain `gt_` columns, which unlock ground-truth evaluation on page 4.

## 3. Pages

1. **Upload data** — upload, validate, preview; template generator.
2. **Run the agent** — set action costs, run; see segments, the risk-vs-uplift scatter, and the plan.
3. **Account detail** — one account: churn score, uplift per action, cost, expected value, rationale.
4. **Evaluation** — Qini curves on the randomised holdout (the honest test on real data) and a policy
   comparison at equal budget.
5. **Export for CRM** — CSV in Zoho CRM's import layout. No live CRM API is implemented.

## 4. How the agent calculates

For account *i* and action *a*:

- CRM churn score: `p0(i)` from a gradient-boosting model trained on the `none` group.
- Treated churn probability: `pa(i)` from a model trained on accounts that received `a`.
- Uplift: `u(i,a) = p0(i) − pa(i)` (positive = action reduces churn). This is a T-learner.
- Value at stake: `V(i) = mrr × 12 × gross margin`.
- Cost: `c(a) = fixed + pct × mrr × 12`.
- Expected value: `EV(i,a) = u(i,a) × V(i) − c(a)`; best action = highest EV.
- Segment: high risk & uplift ≥ threshold → Persuadable; low risk & uplift ≥ threshold → Persuadable (low risk);
  low risk otherwise → Sure thing; high risk, no useful action → Lost cause; best uplift < 0 → Do not disturb.
- Allocation: greedy by EV per dollar among Persuadables until the budget is spent. Nothing else is funded.

The uplift estimate is causal only because the historical campaign was randomised. If your campaign assigned
actions by judgement rather than at random, the estimates are biased and the Qini curves will show it.

## 5. How to judge the results

- **Qini curves (page 4)** are the real test: they use only held-out rows and observed churn. If the uplift
  model's curve sits above the churn-score curve, the agent ranks better than the CRM. With a few thousand rows
  the curves are noisy — change the seed and look for a consistent gap.
- **Policy comparison (page 4)**: on synthetic data it uses planted truth (a simulation). On real data it uses
  the agent's own estimates, so it shows the agent's *expectation*, not evidence.
- **In production**: keep a permanent randomised holdout and compare realised retention of targeted vs. holdout
  accounts. Incremental retained MRR, retention ROI, cost per saved customer, and do-not-disturb avoidance are the KPIs.

## 6. Limitations

Synthetic template only if you lack data; T-learner without confidence intervals; assumes a randomised
campaign; 12-month MRR proxy rather than fitted CLV; no live CRM connection; no fairness checks.

## 7. Files

`app.py` (UI) · `agent.py` (models, decision engine, evaluation) · `generate_data.py` (template generator) ·
`requirements.txt`
