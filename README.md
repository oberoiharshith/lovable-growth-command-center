# Lovable Growth Command Center (POC)

A product-data POC aligned to Lovable’s Data Scientist role.

## 🚀 Live App
https://lovable-growth-command-center.streamlit.app/

**What it includes**
- Metrics framework (activation, retention, time-to-aha proxies)
- Funnel + slice analysis
- Cohorts by signup week
- Simple activation propensity model + experiment plan
- Streamlit dashboard for self-serve insights

## Setup
Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

## Run

```bash
python -m src.analysis --data_dir data/raw --out_dir out
streamlit run app/dashboard.py
```

## Files used by the dashboard

* `out/exports/features.csv`
* `out/exports/cohorts_by_week.csv`
* `out/exports/funnel.csv`

## Regenerate synthetic data (optional)

```bash
python scripts/generate_realistic_data.py
```

## Why this project

This POC demonstrates how to:

* turn raw event data into a trusted metrics layer
* identify activation levers and retention drivers
* translate behavioral insights into concrete product bets
* enable self-serve, decision-grade analytics for a fast-moving team

```
