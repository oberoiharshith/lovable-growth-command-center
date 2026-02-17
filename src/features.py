from __future__ import annotations
import pandas as pd
import numpy as np


def build_first_session_features(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["event_ts"] = pd.to_datetime(e["event_ts"], errors="coerce")

    # First timestamps per user (wide table)
    firsts = e.pivot_table(index="user_id", columns="event_name", values="event_ts", aggfunc="min")

    # Rename timestamp columns so they won't collide with count columns later
    firsts = firsts.rename(columns={
        "signup": "signup_event_ts",
        "first_prompt": "first_prompt_ts",
        "first_output": "first_output_ts",
        "project_created": "project_created_ts",
        "project_shipped": "project_shipped_ts",
        "template_used": "template_used_ts",
        "multi_step_session": "multi_step_session_ts",
        "error_hit": "error_hit_ts",
        "return_session": "return_session_ts",
        "nps_response": "nps_response_ts",
    })

    # Base user table
    out = users[["user_id", "signup_ts", "country", "acquisition_channel", "signup_intent", "platform"]].copy()
    out["signup_ts"] = pd.to_datetime(out["signup_ts"], errors="coerce")

    # Merge first timestamps (optional convenience fields)
    out = out.merge(firsts.reset_index(), on="user_id", how="left")

    # Merge signup_ts into events ONCE (avoid signup_ts_x/y)
    e = e.merge(out[["user_id", "signup_ts"]], on="user_id", how="left")
    e["signup_ts"] = pd.to_datetime(e["signup_ts"], errors="coerce")

    # Activity counts in first 24h
    e24 = e[
        (e["event_ts"] >= e["signup_ts"]) &
        (e["event_ts"] <= e["signup_ts"] + pd.Timedelta(hours=24))
    ]

    counts = e24.groupby(["user_id", "event_name"]).size().unstack(fill_value=0)

    expected = [
        "template_used", "error_hit", "multi_step_session",
        "project_created", "project_shipped", "return_session",
        "first_prompt", "first_output"
    ]
    for c in expected:
        if c not in counts.columns:
            counts[c] = 0

    out = out.merge(counts.reset_index(), on="user_id", how="left").fillna(0).infer_objects(copy=False)

    # Flag for template usage
    out["template_used_flag"] = (out["template_used"] > 0).astype(int)

    # Latency features in first 24h (only from rows where latency_ms is present and > 0)
    if "latency_ms" in e24.columns:
        lat_col = pd.to_numeric(e24["latency_ms"], errors="coerce")
        lat = (
            e24[lat_col.fillna(0) > 0]
            .assign(latency_ms=lat_col)
            .groupby("user_id")["latency_ms"]
            .agg(["mean", "max"])
            .reset_index()
        )
        out = out.merge(lat, on="user_id", how="left")
        out.rename(columns={"mean": "latency_mean_ms", "max": "latency_max_ms"}, inplace=True)
    else:
        out["latency_mean_ms"] = 0.0
        out["latency_max_ms"] = 0.0

    out["latency_mean_ms"] = pd.to_numeric(out.get("latency_mean_ms", 0), errors="coerce").fillna(0)
    out["latency_max_ms"] = pd.to_numeric(out.get("latency_max_ms", 0), errors="coerce").fillna(0)

    # Time deltas (from first timestamps)
    out["time_to_first_prompt_min"] = (
        pd.to_datetime(out.get("first_prompt_ts"), errors="coerce") - out["signup_ts"]
    ).dt.total_seconds() / 60.0

    out["time_to_first_output_min"] = (
        pd.to_datetime(out.get("first_output_ts"), errors="coerce") - out["signup_ts"]
    ).dt.total_seconds() / 60.0

    out["time_to_first_prompt_min"] = out["time_to_first_prompt_min"].fillna(np.inf)
    out["time_to_first_output_min"] = out["time_to_first_output_min"].fillna(np.inf)

    # -----------------------------
    # Labels (IMPORTANT FIX)
    # Compute activation from RAW events, not from pivoted columns
    # -----------------------------
    ship_events = (
        e[e["event_name"] == "project_shipped"]
        .groupby("user_id")["event_ts"]
        .min()
        .rename("ship_ts")
    )

    out = out.merge(ship_events, on="user_id", how="left")

    out["activated_48h"] = (
        out["ship_ts"].notna() &
        ((out["ship_ts"] - out["signup_ts"]).dt.total_seconds() <= 48 * 3600)
    ).astype(int)

    # Retention: any return_session between day 1 and day 7
    retained = (
        e.query("event_name == 'return_session'")
         .assign(deltas=lambda d: (d["event_ts"] - d["signup_ts"]).dt.days)
         .groupby("user_id")["deltas"]
         .apply(lambda s: ((s >= 1) & (s <= 7)).any())
    )
    out["retained_7d"] = out["user_id"].map(retained).fillna(False).astype(int)

    # Buckets for slicing
    out["error_bucket"] = pd.cut(out["error_hit"], bins=[-0.1, 0.5, 1.5, 100], labels=["0", "1", "2+"]).astype(str)
    out["latency_bucket"] = pd.cut(out["latency_mean_ms"], bins=[-1, 500, 1000, 2000, 1e9], labels=["<500", "500-1000", "1000-2000", ">2000"]).astype(str)

    return out