from __future__ import annotations
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path

rng = np.random.default_rng(42)
N = 5000
start = dt.datetime(2025, 12, 15)

countries = ["US","IN","DE","BR","GB","CA"]
channels = ["organic","paid","referral","social"]
intents = ["build_app","prototype","learn","internal_tool"]
platforms = ["web","desktop"]

def random_signup():
    return start + dt.timedelta(hours=float(rng.uniform(0, 24*45)))

users = pd.DataFrame({
    "user_id": [f"user_{i}" for i in range(N)],
    "signup_ts": [random_signup().isoformat() for _ in range(N)],
    "country": rng.choice(countries, N, p=[0.42,0.18,0.10,0.10,0.10,0.10]),
    "acquisition_channel": rng.choice(channels, N, p=[0.55,0.25,0.12,0.08]),
    "signup_intent": rng.choice(intents, N, p=[0.40,0.25,0.20,0.15]),
    "platform": rng.choice(platforms, N, p=[0.55,0.45]),
})

rows = []
def add(uid, name, ts, latency=None):
    rows.append([uid, name, ts.isoformat(), None if latency is None else float(latency)])

for _, u in users.iterrows():
    uid = u.user_id
    signup_ts = dt.datetime.fromisoformat(u.signup_ts)
    intent = u.signup_intent

    uses_template = rng.random() < 0.38
    errors = rng.choice([0,1,2,3], p=[0.45,0.30,0.18,0.07])
    latency = float(rng.lognormal(mean=6.4, sigma=0.35))

    add(uid, "signup", signup_ts)
    t_prompt = signup_ts + dt.timedelta(minutes=float(rng.uniform(1,18)))
    add(uid, "first_prompt", t_prompt, latency)
    t_output = t_prompt + dt.timedelta(seconds=float(rng.uniform(4,35)))
    add(uid, "first_output", t_output, latency)

    if uses_template:
        add(uid, "template_used", t_prompt)
    if errors > 0:
        for _ in range(int(errors)):
            add(uid, "error_hit", t_output)
    if rng.random() < (0.30 + (0.10 if uses_template else 0.0)):
        add(uid, "multi_step_session", t_output + dt.timedelta(minutes=float(rng.uniform(2,40))))

    intent_boost = 0.10 if intent in ("build_app","internal_tool") else (0.05 if intent=="prototype" else 0.0)

    p_create = 0.42 + (0.14 if uses_template else 0.0) - 0.10*(errors >= 2) - 0.08*(latency > 2000) + intent_boost
    p_create = float(np.clip(p_create, 0.03, 0.85))

    if rng.random() < p_create:
        t_create = t_output + dt.timedelta(minutes=float(rng.uniform(5,120)))
        add(uid, "project_created", t_create)

        p_ship = 0.14 + (0.10 if uses_template else 0.0) - 0.10*(errors >= 2) - 0.06*(latency > 2000) + 0.03*intent_boost
        p_ship = float(np.clip(p_ship, 0.01, 0.45))

        if rng.random() < p_ship:
            if rng.random() < (0.30 + (0.10 if uses_template else 0.0)):
                hours = float(max(0.3, rng.lognormal(mean=2.2, sigma=0.55)))
            else:
                hours = float(rng.lognormal(mean=4.2, sigma=0.6))
            add(uid, "project_shipped", t_create + dt.timedelta(hours=hours))

    if rng.random() < 0.22:
        add(uid, "return_session", signup_ts + dt.timedelta(days=int(rng.uniform(1,7)), hours=float(rng.uniform(0,6))))

events = pd.DataFrame(rows, columns=["user_id","event_name","event_ts","latency_ms"])

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)
users.to_csv(out / "users.csv", index=False)
events.to_csv(out / "events.csv", index=False)
print("Generated data/raw/users.csv and data/raw/events.csv")
