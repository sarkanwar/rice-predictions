
# 🌾 basmati-rice-forecast

Streamlit app + CLI to fetch basmati price data, engineer indicators (technical, currency, weather),
train a SARIMAX + XGBoost hybrid, and produce 1‑week / 1‑month / 6‑month forecasts.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud
- Main file path: `streamlit_app.py`
- Python version: 3.10 or 3.11

## CLI
```bash
python cli.py fetch-agmarknet --state "Haryana" --market "Karnal"   --variety_keywords "Basmati,1121,1509,1718,PB-1"   --date_from 2023-01-01 --date_to 2025-10-25 --out_csv data/basmati_prices.csv

python cli.py run-all --horizons 7 30 180
```

Outputs are saved under `artifacts/YYYY-MM-DD/` as CSVs and PNG charts.
Replace `data/basmati_prices.csv` with your real series (Date,Price).
