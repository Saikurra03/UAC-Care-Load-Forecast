# Predictive Forecasting of Care Load, Placement Demand, and Capacity Pressure

This repository contains a production-ready Streamlit application for forecasting care load, discharge demand, capacity pressure, and operational risk based on the available HHS dataset.

## Structure

- `app.py` - Streamlit dashboard entry point
- `src/` - Modular application package
- `data/` - Data file discovery location
- `logs/` - Logging output
- `artifacts/` - Generated datasets and reports
- `models/` - Saved model artifacts
- `reports/` - Exported analysis outputs

## Installation

1. Create a Python environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the app

```bash
streamlit run app.py
```

## Capabilities

- Automatic dataset discovery from CSV, Excel, and Parquet
- Dynamic schema inspection and type inference
- Data quality reporting and cleaned artifact creation
- Feature engineering, time series diagnostics, and forecasting
- Forecast horizons from 7 to 90 days
- Risk scoring for capacity and pressure
- Downloadable forecast exports

## Notes

- The app is designed to handle unexpected schema variations and missing values gracefully.
- If optional libraries such as `xgboost`, `lightgbm`, or `catboost` are missing, the app will still execute using baseline and core models.
