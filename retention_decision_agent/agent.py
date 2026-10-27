"""
Retention Decision Agent (RDA) -- core analytics and decision logic.
Works on any CSV that follows the column contract in README.md.

Pipeline:
  1. Baseline churn model (proxy for the CRM's native churn score)
  2. T-learner uplift model per retention action found in the data
  3. Value model (12-month gross margin at stake) and expected-value decision engine
  4. Four-quadrant persuadability segmentation
  5. Budget-constrained allocation (only Persuadables are funded)
  6. Template-based explanations
  7. Evaluation: empirical Qini curves on a randomised holdout; ground-truth
     simulation only if gt_* columns exist (synthetic data)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

REQUIRED_HISTORY = ["account_id", "plan", "mrr", "tenure_months", "logins_30d",
                    "days_since_last_login", "support_tickets_90d", "nps",
                    "discount_pct", "csm_touches_90d", "contract_months_left",
                    "campaign_action", "churned"]
REQUIRED_BOOK = REQUIRED_HISTORY[:-2]
NUMERIC = ["mrr", "tenure_months", "logins_30d", "days_since_last_login",
           "support_tickets_90d", "nps", "discount_pct", "csm_touches_90d",
           "contract_months_left"]
CONTROL = "none"

DEFAULT_COSTS = {  # fixed $ per contact, plus % of 12-month MRR given away
    "csm_call": {"fixed": 60.0, "pct_mrr12": 0.0},
    "discount_15": {"fixed": 0.0, "pct_mrr12": 0.15},
    "training": {"fixed": 25.0, "pct_mrr12": 0.0},
}
GROSS_MARGIN = 0.70
HORIZON_MONTHS = 12


# ---------------------------------------------------------------- validation
def validate(df, kind="history"):
    """Return a list of problems (empty list = OK)."""
    req = REQUIRED_HISTORY if kind == "history" else REQUIRED_BOOK
    problems = [f"missing column: {c}" for c in req if c not in df.columns]
    if problems:
        return problems
    for c in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"column {c} must be numeric")
    if df["account_id"].duplicated().any():
        problems.append("account_id has duplicates")
    if kind == "history":
        if not set(df["churned"].dropna().unique()) <= {0, 1}:
            problems.append("churned must be 0 or 1")
        if CONTROL not in set(df["campaign_action"]):
            problems.append(f"campaign_action needs a control group labelled '{CONTROL}'")
        counts = df["campaign_action"].value_counts()
        small = counts[counts < 100]
        for a, n in small.items():
            problems.append(f"campaign_action '{a}' has only {n} rows (need >= 100 for a usable model)")
        if len(counts) < 2:
            problems.append("need at least one treated action besides 'none'")
    return problems


def actions_in(df):
    return [a for a in sorted(df["campaign_action"].unique()) if a != CONTROL]


def _prep(df, plans):
    X = df.copy()
    for p in plans:
        X[f"plan_{p}"] = (X["plan"] == p).astype(int)
    return X[NUMERIC + [f"plan_{p}" for p in plans]]


def _gbm(seed=0):
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, min_samples_leaf=40,
                                      learning_rate=0.05, subsample=0.8, random_state=seed)


def action_cost(action, mrr, costs):
    c = costs.get(action, {"fixed": 50.0, "pct_mrr12": 0.0})
    return c["fixed"] + c["pct_mrr12"] * mrr * HORIZON_MONTHS


def pretty(action):
    return action.replace("_", " ").capitalize()


# ---------------------------------------------------------------- agent
class RetentionDecisionAgent:
    def __init__(self, costs=None, margin=GROSS_MARGIN, risk_thresh=0.12,
                 uplift_thresh=0.03, seed=0):
        self.costs = costs or DEFAULT_COSTS
        self.margin = margin
        self.risk_thresh = risk_thresh
        self.uplift_thresh = uplift_thresh
        self.seed = seed
        self.actions, self.plans = [], []
        self.baseline, self.treated_models = None, {}

    def fit(self, hist):
        self.actions = actions_in(hist)
        self.plans = sorted(hist["plan"].unique())
        X = _prep(hist, self.plans)
        y = hist["churned"].astype(int).values
        ctrl = (hist["campaign_action"] == CONTROL).values
        self.baseline = _gbm(self.seed).fit(X[ctrl], y[ctrl])
        for i, a in enumerate(self.actions):
            m = (hist["campaign_action"] == a).values
            self.treated_models[a] = _gbm(self.seed + i + 1).fit(X[m], y[m])
        return self

    def score(self, book):
        X = _prep(book, self.plans)
        out = book[REQUIRED_BOOK].copy().reset_index(drop=True)
        out["crm_churn_score"] = self.baseline.predict_proba(X)[:, 1]
        out["value_at_stake"] = out["mrr"] * HORIZON_MONTHS * self.margin
        for a in self.actions:
            p_treat = self.treated_models[a].predict_proba(X)[:, 1]
            out[f"uplift_{a}"] = out["crm_churn_score"] - p_treat
            out[f"cost_{a}"] = [action_cost(a, m, self.costs) for m in out["mrr"]]
            out[f"ev_{a}"] = out[f"uplift_{a}"] * out["value_at_stake"] - out[f"cost_{a}"]
        ev_cols = [f"ev_{a}" for a in self.actions]
        up_cols = [f"uplift_{a}" for a in self.actions]
        out["best_action"] = out[ev_cols].idxmax(axis=1).str.replace("ev_", "", regex=False)
        out["best_ev"] = out[ev_cols].max(axis=1)
        out["best_uplift"] = out[up_cols].max(axis=1)
        out["best_cost"] = [out.loc[i, f"cost_{a}"] for i, a in zip(out.index, out["best_action"])]
        out["segment"] = self._segment(out, up_cols)
        out["rationale"] = [self._explain(r) for _, r in out.iterrows()]
        return out

    def _segment(self, s, up_cols):
        seg = np.full(len(s), "Sure thing", dtype=object)
        high_risk = s["crm_churn_score"] >= self.risk_thresh
        persuadable = s["best_uplift"] >= self.uplift_thresh
        dnd = (s[up_cols].max(axis=1) < 0) & (s[up_cols].min(axis=1) < -self.uplift_thresh)
        seg[high_risk & persuadable] = "Persuadable"
        seg[high_risk & ~persuadable] = "Lost cause"
        seg[~high_risk & persuadable] = "Persuadable (low risk)"
        seg[dnd] = "Do not disturb"
        return seg

    @staticmethod
    def allocate(scored, budget):
        cand = scored[(scored["best_ev"] > 0) & scored["segment"].str.startswith("Persuadable")].copy()
        cand["ev_per_cost"] = cand["best_ev"] / cand["best_cost"].clip(lower=1)
        cand = cand.sort_values("ev_per_cost", ascending=False)
        spent, chosen = 0.0, []
        for idx, r in cand.iterrows():
            if spent + r["best_cost"] <= budget:
                spent += r["best_cost"]; chosen.append(idx)
        plan = scored.loc[chosen].copy()
        plan["expected_retained_mrr"] = plan["best_uplift"] * plan["mrr"]
        return plan.sort_values("best_ev", ascending=False), spent

    def _explain(self, r):
        sig = []
        if r["support_tickets_90d"] >= 3:
            sig.append(f"{int(r['support_tickets_90d'])} support tickets in 90 days")
        if r["days_since_last_login"] >= 20:
            sig.append(f"no login for {int(r['days_since_last_login'])} days")
        if r["nps"] <= 5:
            sig.append(f"NPS of {int(r['nps'])}")
        if r["tenure_months"] <= 9 and r["logins_30d"] <= 8:
            sig.append("new account with low adoption")
        if r["discount_pct"] == 0:
            sig.append("no discount on file")
        sig = "; ".join(sig) if sig else "no strong risk signals"
        a = pretty(r["best_action"])
        if r["segment"] == "Do not disturb":
            return (f"Churn risk {r['crm_churn_score']:.0%}. Every available action is estimated to increase "
                    f"churn for accounts like this ({sig}). Recommendation: do not contact.")
        if r["segment"] == "Lost cause":
            return (f"Churn risk {r['crm_churn_score']:.0%} but no action moves it meaningfully "
                    f"(best uplift {r['best_uplift']*100:+.1f} pts). Budget is better spent elsewhere.")
        if r["segment"] == "Sure thing":
            return (f"Churn risk only {r['crm_churn_score']:.0%}; likely to stay without intervention. "
                    f"{a} would mostly be wasted spend.")
        return (f"Churn risk {r['crm_churn_score']:.0%}. {a} is estimated to cut churn by "
                f"{r['best_uplift']*100:.1f} points for this profile ({sig}). Expected margin protected over "
                f"12 months: ${r['best_uplift'] * r['value_at_stake']:,.0f}, for a cost of ${r['best_cost']:,.0f}.")


# ---------------------------------------------------------------- evaluation
def qini_curve(df, score_col, treat_col, y_col, n_bins=20):
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(d)
    fracs, gains = [0.0], [0.0]
    for k in range(1, n_bins + 1):
        top = d.iloc[: int(n * k / n_bins)]
        t, c = top[top[treat_col] == 1], top[top[treat_col] == 0]
        fracs.append(k / n_bins)
        if len(t) == 0 or len(c) == 0:
            gains.append(gains[-1]); continue
        gains.append((c[y_col].mean() - t[y_col].mean()) * len(t))
    return np.array(fracs), np.array(gains)


def auuc(fracs, gains):
    return float(np.trapezoid(gains, fracs))


def holdout_qini(agent, holdout, scored_holdout, seed=0):
    """Empirical Qini per action on randomised holdout data (no ground truth needed)."""
    out = {}
    for a in agent.actions:
        sub = holdout[holdout["campaign_action"].isin([CONTROL, a])][["account_id", "campaign_action", "churned"]]
        sub = sub.merge(scored_holdout[["account_id", "crm_churn_score", f"uplift_{a}"]], on="account_id")
        sub["treated"] = (sub["campaign_action"] == a).astype(int)
        sub["rand"] = np.random.default_rng(seed).random(len(sub))
        out[a] = {name: qini_curve(sub, col, "treated", "churned")
                  for name, col in [("Uplift model", f"uplift_{a}"),
                                    ("CRM churn score", "crm_churn_score"),
                                    ("Random", "rand")]}
    return out


def policy_comparison(agent, scored, budget, truth=None):
    """
    Compare targeting policies at equal budget.
    If `truth` (a DataFrame with gt_p_none and gt_p_<action> indexed by account_id)
    is given, outcomes use planted true effects (simulation). Otherwise outcomes use
    the agent's own estimated uplift -- an internal estimate, not an observed result.
    """
    def tau(r):
        if truth is not None:
            g = truth.loc[r["account_id"]]
            return g["gt_p_none"] - g[f"gt_p_{r['best_action']}"]
        return r[f"uplift_{r['best_action']}"]

    def greedy(df_sorted, action):
        d = df_sorted.copy()
        d["best_action"] = action
        d["best_cost"] = d[f"cost_{action}"]
        spent, keep = 0.0, []
        for idx, c in d["best_cost"].items():
            if spent + c <= budget:
                spent += c; keep.append(idx)
        return d.loc[keep]

    by_score = scored.sort_values("crm_churn_score", ascending=False)
    policies = {f"Churn score + {pretty(a)} to all": greedy(by_score, a) for a in agent.actions}
    policies["Random + " + pretty(agent.actions[0])] = greedy(scored.sample(frac=1, random_state=1), agent.actions[0])
    policies["Retention Decision Agent"], _ = agent.allocate(scored, budget)

    rows = []
    for name, p in policies.items():
        taus = np.array([tau(r) for _, r in p.iterrows()]) if len(p) else np.array([])
        rows.append({"policy": name, "accounts_contacted": len(p),
                     "budget_spent": float(p["best_cost"].sum()),
                     "churners_prevented": float(taus.sum()),
                     "retained_mrr": float((taus * p["mrr"].values).sum()),
                     "accounts_harmed": int((taus < 0).sum())})
    basis = "planted ground truth (simulation)" if truth is not None else "agent's own estimates"
    return pd.DataFrame(rows), basis


def run(history, book=None, budget=20000, costs=None, margin=GROSS_MARGIN,
        risk_thresh=0.12, uplift_thresh=0.03, seed=0, holdout_frac=0.3):
    """
    history: labelled campaign data. book: accounts to score now (optional).
    If book is None, a stratified holdout of history is scored and used for evaluation.
    """
    hist, hold = train_test_split(history, test_size=holdout_frac, random_state=seed,
                                  stratify=history["campaign_action"])
    agent = RetentionDecisionAgent(costs, margin, risk_thresh, uplift_thresh, seed).fit(hist)
    scored_hold = agent.score(hold)
    qini = holdout_qini(agent, hold, scored_hold, seed)
    truth = hold.set_index("account_id") if "gt_p_none" in hold.columns else None
    comparison, basis = policy_comparison(agent, scored_hold, budget, truth)
    target = book if book is not None else hold
    scored = agent.score(target) if book is not None else scored_hold
    plan, spent = agent.allocate(scored, budget)
    return {"agent": agent, "train": hist, "holdout": hold, "scored_holdout": scored_hold,
            "scored": scored, "plan": plan, "spent": spent, "qini": qini,
            "comparison": comparison, "comparison_basis": basis,
            "has_truth": truth is not None, "book_was_uploaded": book is not None}
