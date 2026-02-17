from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .features import build_first_session_features

def md_write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exports").mkdir(parents=True, exist_ok=True)

    users = pd.read_csv(data_dir / "users.csv")
    events = pd.read_csv(data_dir / "events.csv")

    feats = build_first_session_features(users, events)

    # Funnel
    ev = events.copy()
    ev["event_ts"] = pd.to_datetime(ev["event_ts"])
    funnel = (ev.groupby("event_name")["user_id"].nunique()
              .reindex(["signup","first_prompt","first_output","project_created","project_shipped"])
              .fillna(0).astype(int))
    funnel_df = funnel.reset_index().rename(columns={"event_name":"step","user_id":"users"})
    funnel_df.to_csv(out_dir / "exports" / "funnel.csv", index=False)

    # Cohorts
    feats["signup_week"] = pd.to_datetime(feats["signup_ts"]).dt.to_period("W").astype(str)
    cohort_tbl = feats.groupby("signup_week", observed=False)[["activated_48h","retained_7d"]].mean().reset_index()
    cohort_tbl.to_csv(out_dir / "exports" / "cohorts_by_week.csv", index=False)

    # Driver summaries (robust indexing)
    template_rate = feats.groupby("template_used_flag")["activated_48h"].mean()
    tmpl1 = float(template_rate.get(1, np.nan))
    tmpl0 = float(template_rate.get(0, np.nan))
    tmpl_uplift = tmpl1 - tmpl0 if (np.isfinite(tmpl1) and np.isfinite(tmpl0)) else np.nan

    err_bucket = feats.groupby(feats["error_bucket"], observed=False)["activated_48h"].mean()
    lat_bucket = feats.groupby(feats["latency_bucket"], observed=False)["activated_48h"].mean()

    # Model: activation propensity
    num_cols = ["template_used","error_hit","latency_mean_ms","time_to_first_output_min","multi_step_session"]
    cat_cols = ["acquisition_channel","signup_intent","platform","country"]

    X = feats[num_cols + cat_cols].copy()
    y = feats["activated_48h"].astype(int)

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])

    feats["activation_score"] = pipe.predict_proba(X)[:,1]
    feats.to_csv(out_dir / "exports" / "features.csv", index=False)

    # Markdown outputs
    md_write(out_dir / "01_metrics_summary.md", f"""# Metrics Summary

- **Activation (ship in 48h)**: {feats["activated_48h"].mean():.3f}
- **7-day retention**: {feats["retained_7d"].mean():.3f}
- **Template used (first 24h)**: {feats["template_used_flag"].mean():.3f}

## Key driver readout
- **Template uplift (1 vs 0)**: {tmpl_uplift:.3f}

### Activation by errors
{err_bucket.reset_index().to_markdown(index=False)}

### Activation by latency
{lat_bucket.reset_index().to_markdown(index=False)}
""")

    md_write(out_dir / "02_funnel.md", "# Funnel\n\n" + funnel_df.to_markdown(index=False))
    md_write(out_dir / "03_cohorts.md", "# Cohorts (by signup week)\n\n" + cohort_tbl.to_markdown(index=False))
    md_write(out_dir / "04_model.md", f"""# Activation model (ship in 48h)

Model: logistic regression on early signals + acquisition context.

- **ROC AUC:** {auc:.3f}
""")

    md_write(out_dir / "05_experiment_plan.md", """# Experiment Plan

## Bet 1: Intent-based template suggestions
- Primary: activation_48h
- Guardrails: error_hit, latency_mean_ms

## Bet 2: Debug helper when errors >= 2
- Primary: project_created rate, activation_48h
- Guardrails: time_to_first_output_min

## Bet 3: First-output latency guardrails
- Primary: retained_7d
- Guardrails: activation_48h
""")

if __name__ == "__main__":
    main()
