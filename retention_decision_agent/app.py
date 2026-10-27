"""
Retention Decision Agent -- upload your own CRM data and run the agent.
Run:  python -m streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import agent as A
from generate_data import generate

st.set_page_config(page_title="Retention Decision Agent", layout="wide")

# ------------------------------------------------------------------ state
if "history_df" not in st.session_state:
    st.session_state.history_df = None
    st.session_state.book_df = None
    st.session_state.result = None
    st.session_state.source = None

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Retention Decision Agent")
page = st.sidebar.radio("Go to", ["Overview", "1. Upload data", "2. Run the agent", "3. Account detail",
                                  "4. Evaluation", "5. Export for CRM"])
st.sidebar.markdown("---")
st.sidebar.subheader("Settings")
budget = st.sidebar.number_input("Retention budget ($)", 1000, 1_000_000, 20000, step=1000)
margin = st.sidebar.slider("Gross margin", 0.2, 0.95, 0.70, 0.05)
risk_thresh = st.sidebar.slider("High-risk threshold (churn score)", 0.02, 0.5, 0.12, 0.01)
uplift_thresh = st.sidebar.slider("Minimum uplift to act (points)", 0.01, 0.15, 0.03, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 99, 0)

# ================================================================== overview
if page == "Overview":
    st.title("From churn score to retention decision")

    st.subheader("Statement of the problem")
    st.write(
        "Subscription businesses lose customers every month. Modern CRMs such as Zoho, Salesforce, HubSpot and "
        "Dynamics 365 now predict which customers are likely to leave by giving each one a churn score. "
        "But a churn score only answers **who might leave**. It does not answer the question a retention manager "
        "has to act on: **which customers can be saved, with which action, and is it worth the cost?**")
    st.write(
        "Because of this gap, teams contact the highest-scoring customers and apply a standard action such as a "
        "discount or a call. Evidence from telecom retention programmes shows this wastes budget three ways: "
        "discounts go to customers who would have stayed anyway, effort goes to customers who cannot be saved, "
        "and some customers are pushed into leaving by the contact itself. The missing capability is an estimate "
        "of the **effect** of each action on each customer - a causal question, not a prediction question - "
        "combined with the value of the customer and the cost of the action.")

    st.subheader("What the agent does")
    st.markdown("""
1. **Reads** the uploaded customer data - plan, revenue, usage, support tickets, NPS - and computes a churn score the way a CRM would.
2. **Learns what each action actually does.** For every customer it estimates how much each action would lower (or raise)
   their chance of leaving, by comparing customers who received that action with similar customers who received nothing
   in a past randomised campaign.
3. **Works out whether it is worth it.** Estimated churn reduction x revenue at stake - cost of the action = expected value
   for every customer-action pair.
4. **Sorts customers into four groups:** Persuadable (worth acting on), Sure thing (will stay anyway), Lost cause
   (nothing helps), Do not disturb (contact makes it worse). Only Persuadables are funded.
5. **Builds the plan and explains it.** Within the budget it picks the customers and actions with the best return,
   writes a plain-language reason for each recommendation, and exports a file in Zoho CRM's import layout.
   It recommends; a human decides and acts.
""")

    st.subheader("Why this is beyond existing CRM features")
    st.markdown("""
| Platform | Provides | Does not provide |
|---|---|---|
| Zoho CRM (Zia) | Churn score per record, model-accuracy rating | Effect of an action, action choice, cost awareness |
| Dynamics 365 Customer Insights | Churn probability, aggregate influencing factors | Per-customer intervention effect, do-not-disturb detection |
| Salesforce Einstein / Agentforce | Churn prediction, next-best-action, task execution | Causal evidence that the action changes the outcome |
| HubSpot Breeze | Health agent that triggers outreach when a score drops | Evidence that outreach helps this customer |

All four answer "how likely is churn?". The agent answers "what is the incremental effect of each action, and is it worth paying for?"
""")

    st.subheader("How to use this app")
    st.markdown("""
1. **Upload data** - a past retention campaign with a `none` control group and a `churned` outcome (required),
   plus current accounts to score (optional). A template generator is provided if you have no data.
2. **Run the agent** - set the cost of each action and the budget, then run.
3. **Account detail** - inspect any account: churn score, uplift per action, cost, expected value, rationale.
4. **Evaluation** - Qini curves on held-out data and a policy comparison at equal budget.
5. **Export for CRM** - download the recommendations in Zoho CRM import layout.
""")

    st.subheader("What is real and what is simulated")
    st.markdown("""
- **Real:** the business problem and the evidence behind it (Telenor and T-Mobile Austria uplift results,
  Validity and Gartner survey findings, documented outputs of the four CRM churn features).
- **Simulated:** any data produced by the template generator, including the sample files supplied with this
  project. Numbers shown on those files demonstrate the mechanism, not real-world magnitudes.
- **Not implemented:** a live CRM API connection. The export is a file for manual import.
""")

# ================================================================== page 1
elif page == "1. Upload data":
    st.title("Upload your data")
    st.write("The agent needs one file: a **historical retention campaign** where accounts were "
             "assigned to actions (including a 'none' control group) and you know whether each one churned. "
             "Optionally add a second file of **current accounts** to score. If you skip the second file, "
             "a 30% holdout of the first file is scored and used for evaluation.")

    with st.expander("Required columns", expanded=False):
        st.markdown("""
| Column | Meaning |
|---|---|
| `account_id` | unique id |
| `plan` | text, e.g. Basic / Pro / Enterprise |
| `mrr` | monthly recurring revenue |
| `tenure_months`, `logins_30d`, `days_since_last_login`, `support_tickets_90d`, `nps`, `discount_pct`, `csm_touches_90d`, `contract_months_left` | numeric account signals |
| `campaign_action` | **history file only** - action received: `none` for control, any other labels for treatments (e.g. `csm_call`, `discount_15`, `training`). Each label needs at least 100 rows. |
| `churned` | **history file only** - 1 if the account churned after the campaign, else 0 |

Any treatment labels work. Costs for each label are set on the Run page.
Columns beginning `gt_` are optional and only appear in synthetic files; they enable ground-truth evaluation.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historical campaign file (required)")
        up_hist = st.file_uploader("CSV", type="csv", key="hist")
        if up_hist is not None:
            df = pd.read_csv(up_hist)
            probs = A.validate(df, "history")
            if probs:
                st.error("File rejected:\n\n- " + "\n- ".join(probs))
            else:
                st.session_state.history_df = df
                st.session_state.result = None
                st.session_state.source = up_hist.name
                st.success(f"Loaded {len(df):,} rows, actions: {', '.join(A.actions_in(df))}")
    with col2:
        st.subheader("Current accounts to score (optional)")
        up_book = st.file_uploader("CSV", type="csv", key="book")
        if up_book is not None:
            bk = pd.read_csv(up_book)
            probs = A.validate(bk, "book")
            if probs:
                st.error("File rejected:\n\n- " + "\n- ".join(probs))
            else:
                st.session_state.book_df = bk
                st.session_state.result = None
                st.success(f"Loaded {len(bk):,} accounts to score")
        elif st.session_state.book_df is not None and st.button("Clear current-accounts file"):
            st.session_state.book_df = None
            st.session_state.result = None

    st.markdown("---")
    st.subheader("No data yet? Download a template")
    st.write("Generates a synthetic file in the right format with planted treatment effects. "
             "Use it to learn the layout or to test the app; it is not real customer data.")
    n = st.number_input("Rows", 1000, 50000, 6000, step=1000)
    if st.button("Build template CSV"):
        tmpl = generate(n=int(n), seed=42)
        st.download_button("Download synthetic_campaign_template.csv", tmpl.to_csv(index=False).encode(),
                           "synthetic_campaign_template.csv", "text/csv")

    if st.session_state.history_df is not None:
        st.markdown("---")
        st.subheader("Preview of loaded history")
        st.dataframe(st.session_state.history_df.head(15), use_container_width=True)
        st.dataframe(st.session_state.history_df.groupby("campaign_action")["churned"]
                     .agg(rows="count", churn_rate="mean").round(3))

# ------------------------------------------------------------------ guard
elif st.session_state.history_df is None:
    st.title(page[3:])
    st.info("Upload a historical campaign file on page 1 first.")

# ================================================================== page 2
elif page == "2. Run the agent":
    st.title("Run the agent")
    hist = st.session_state.history_df
    actions = A.actions_in(hist)

    st.subheader("Cost of each action")
    st.write("Fixed cost per contact, plus any share of 12-month MRR given away (for discounts).")
    costs = {}
    cols = st.columns(len(actions))
    for c, a in zip(cols, actions):
        d = A.DEFAULT_COSTS.get(a, {"fixed": 50.0, "pct_mrr12": 0.0})
        with c:
            st.markdown(f"**{A.pretty(a)}**")
            fixed = st.number_input("Fixed cost ($)", 0.0, 10000.0, float(d["fixed"]), 5.0, key=f"f_{a}")
            pct = st.number_input("% of 12-mo MRR given away", 0.0, 100.0, float(d["pct_mrr12"] * 100), 1.0, key=f"p_{a}")
            costs[a] = {"fixed": fixed, "pct_mrr12": pct / 100}

    if st.button("Run agent", type="primary"):
        with st.spinner("Training models and building the plan..."):
            st.session_state.result = A.run(hist, st.session_state.book_df, budget, costs, margin,
                                            risk_thresh, uplift_thresh, int(seed))
    res = st.session_state.result
    if res is None:
        st.stop()

    scored, plan = res["scored"], res["plan"]
    src = "your uploaded current-accounts file" if res["book_was_uploaded"] else "a 30% holdout of your history file"
    st.write(f"Scored **{len(scored):,}** accounts from {src}. Budget ${budget:,}; "
             f"allocated **${res['spent']:,.0f}** across **{len(plan)}** accounts.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accounts recommended", len(plan))
    k2.metric("Est. churners prevented", f"{plan['best_uplift'].sum():.1f}")
    k3.metric("Est. MRR protected / month", f"${plan['expected_retained_mrr'].sum():,.0f}")
    k4.metric("Do-not-disturb excluded", int((scored["segment"] == "Do not disturb").sum()))
    st.caption("Estimates come from the agent's uplift models; see Evaluation for how well they rank.")

    c1, c2 = st.columns(2)
    with c1:
        seg = scored["segment"].value_counts()
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.barh(seg.index, seg.values, color="#3b5b8c"); ax.invert_yaxis(); ax.set_xlabel("accounts")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 3))
        colors = {"Persuadable": "#2a7f62", "Persuadable (low risk)": "#8fbf9f",
                  "Sure thing": "#bbbbbb", "Lost cause": "#d98c3f", "Do not disturb": "#b23a3a"}
        for sname, grp in scored.groupby("segment"):
            ax2.scatter(grp["crm_churn_score"], grp["best_uplift"], s=6, alpha=0.5, label=sname,
                        color=colors.get(sname, "grey"))
        ax2.axhline(0, color="black", lw=0.7)
        ax2.set_xlabel("CRM churn score"); ax2.set_ylabel("best estimated uplift")
        ax2.legend(markerscale=3, fontsize=7, frameon=False)
        for s in ["top", "right"]: ax2.spines[s].set_visible(False)
        st.pyplot(fig2)
    st.caption("A churn-score policy reads left to right. The agent reads top to bottom and skips the red points.")

    st.subheader("Recommended actions")
    show = ["account_id", "plan", "mrr", "segment", "crm_churn_score", "best_action", "best_uplift",
            "best_cost", "best_ev", "expected_retained_mrr"]
    st.dataframe(plan[show].round(3), use_container_width=True, height=380)

# ================================================================== page 3
elif page == "3. Account detail":
    st.title("Account detail")
    res = st.session_state.result
    if res is None:
        st.info("Run the agent on page 2 first."); st.stop()
    scored, agent = res["scored"], res["agent"]
    seg_filter = st.selectbox("Show accounts in segment", ["All"] + sorted(scored["segment"].unique()))
    pool = scored if seg_filter == "All" else scored[scored["segment"] == seg_filter]
    acc = st.selectbox("Account", pool["account_id"].tolist())
    r = scored[scored["account_id"] == acc].iloc[0]
    in_plan = acc in set(res["plan"]["account_id"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CRM churn score", f"{r['crm_churn_score']:.0%}")
    c2.metric("Segment", r["segment"])
    c3.metric("Best action", A.pretty(r["best_action"]))
    c4.metric("In this plan", "Yes" if in_plan else "No")
    st.info(r["rationale"])
    st.subheader("Estimated effect of each action")
    st.dataframe(pd.DataFrame({
        "action": [A.pretty(a) for a in agent.actions],
        "churn reduction (uplift)": [r[f"uplift_{a}"] for a in agent.actions],
        "cost": [r[f"cost_{a}"] for a in agent.actions],
        "expected value": [r[f"ev_{a}"] for a in agent.actions]}).round(3), use_container_width=True)
    st.subheader("Account signals")
    st.dataframe(r[A.REQUIRED_BOOK[1:]].to_frame("value"))

# ================================================================== page 4
elif page == "4. Evaluation":
    st.title("Evaluation")
    res = st.session_state.result
    if res is None:
        st.info("Run the agent on page 2 first."); st.stop()
    agent = res["agent"]

    st.subheader("Ranking quality on the randomised holdout (Qini curves)")
    st.write(f"{len(res['holdout']):,} rows of your history were held out and never used for training. "
             "For each action, accounts are ranked by a score and the curve shows cumulative churners prevented "
             "(control churn rate minus treated churn rate, times treated count). Higher is better.")
    action = st.selectbox("Action", agent.actions, format_func=A.pretty)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, (f, g) in res["qini"][action].items():
        ax.plot(f, g, label=f"{name} (AUUC {A.auuc(f, g):.1f})")
    ax.set_xlabel("share of accounts targeted"); ax.set_ylabel("cumulative churners prevented")
    ax.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    st.pyplot(fig)
    st.caption("This is measurable on real data. With a few thousand rows it is noisy; change the seed to see the variation.")

    st.subheader("Policy comparison at equal budget (holdout accounts)")
    if res["has_truth"]:
        st.warning("Your file contains gt_ columns (synthetic data). Outcomes below are computed from the planted "
                   "true effects - a simulation, not an observed result.")
    else:
        st.warning("Outcomes below use the agent's own estimated uplift, because true effects are unknown on real data. "
                   "Treat them as the agent's expectation, not as evidence. The Qini curves above are the honest test.")
    st.dataframe(res["comparison"].round(1), use_container_width=True)

    st.subheader("How to measure impact for real")
    st.markdown("""
Keep a permanent **randomised holdout**: each cycle, a slice of eligible accounts gets no agent-driven action.
Then compare realised retention between agent-targeted accounts and the holdout:

- Incremental retained MRR = (holdout churn rate - targeted churn rate) x targeted MRR
- Retention ROI = incremental retained revenue / campaign cost
- Cost per saved customer, offer waste rate (offers to Sure things), and do-not-disturb avoidance
""")

# ================================================================== page 5
elif page == "5. Export for CRM":
    st.title("Export for CRM import")
    res = st.session_state.result
    if res is None:
        st.info("Run the agent on page 2 first."); st.stop()
    st.warning("No live CRM connection is implemented. This produces a CSV in the column layout Zoho CRM's "
               "Import tool accepts (Setup > Data Administration > Import > Accounts). Create the five custom "
               "fields first, then map Account_ID to Account Name or an external ID.")
    plan = res["plan"]
    export = plan[["account_id", "best_action", "segment", "best_uplift", "expected_retained_mrr", "rationale"]].rename(
        columns={"account_id": "Account_ID", "best_action": "Recommended_Action",
                 "segment": "Persuadability_Segment", "best_uplift": "Expected_Churn_Reduction",
                 "expected_retained_mrr": "Expected_Retained_MRR", "rationale": "Agent_Rationale"})
    export["Recommended_Action"] = export["Recommended_Action"].map(A.pretty)
    export["Task_Subject"] = "Retention: " + export["Recommended_Action"]
    st.dataframe(export.head(20), use_container_width=True)
    st.download_button("Download recommendations CSV", export.to_csv(index=False).encode(),
                       "retention_recommendations.csv", "text/csv")
    full = res["scored"].drop(columns=["rationale"]).round(4)
    st.download_button("Download all scored accounts (every segment)", full.to_csv(index=False).encode(),
                       "all_scored_accounts.csv", "text/csv")
