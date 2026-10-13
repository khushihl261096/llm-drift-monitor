# LLM Drift Monitor

A minimum working prototype for a paper on longitudinal drift attribution in LLMs.

## Requirements
- Python 3.10+
- Packages in `requirements.txt`

## Run on Windows CMD
```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data
`data/sample_scores.csv` contains simulated scores for three model versions. The data is deliberately designed to demonstrate score changes; it is not collected from actual LLM evaluations.

## Scope and limitations
- Threshold-based comparison of consecutive run means.
- Subgroup comparison and charts.
- CSV export.
- No live LLM API calls, formal PELT/Bayesian change-point detection, validated bias scorer, or production database.
