import json
from pathlib import Path

import numpy as np
import pandas as pd


def _to_jsonable(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _series_mode(series: pd.Series):
    try:
        modes = series.mode(dropna=True)
        return _to_jsonable(modes.iloc[0]) if len(modes) else None
    except Exception:
        return None


def _profile_categorical(series: pd.Series, top_n: int = 25) -> dict:
    s = series
    counts = s.value_counts(dropna=False)
    probs = (counts / counts.sum()).head(top_n)
    top_counts = counts.head(top_n)

    null_rate = float(s.isna().mean())
    unique_count = int(s.nunique(dropna=True))

    return {
        "type": "categorical",
        "null_rate": null_rate,
        "unique_count": unique_count,
        "mode": _series_mode(s),
        "top": [
            {
                "value": _to_jsonable(idx),
                "count": int(top_counts.loc[idx]),
                "probability": float(probs.loc[idx]),
            }
            for idx in probs.index
        ],
    }


def _profile_numeric(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    data = s.dropna()

    if data.empty:
        return {
            "type": "numerical",
            "null_rate": float(s.isna().mean()),
            "count": 0,
        }

    mean = float(data.mean())
    std = float(data.std(ddof=1)) if len(data) > 1 else 0.0

    percentiles = data.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    percentiles = {f"p{int(k * 100):02d}": float(v) for k, v in percentiles.items()}

    return {
        "type": "numerical",
        "null_rate": float(s.isna().mean()),
        "count": int(len(data)),
        "mean": mean,
        "median": float(data.median()),
        "std": std,
        "min": float(data.min()),
        "max": float(data.max()),
        "zero_rate": float((data == 0).mean()),
        "coefficient_of_variation": (std / mean) if mean != 0 else None,
        "percentiles": percentiles,
    }


def _profile_date(series: pd.Series) -> dict:
    dt = pd.to_datetime(series, errors="coerce")
    data = dt.dropna()

    if data.empty:
        return {
            "type": "date",
            "null_rate": float(dt.isna().mean()),
            "count": 0,
        }

    min_date = data.min()
    max_date = data.max()
    date_range_days = int((max_date - min_date).days)

    dow = data.dt.dayofweek.value_counts(normalize=True).sort_index().to_dict()
    month = data.dt.month.value_counts(normalize=True).sort_index().to_dict()
    dom = data.dt.day.value_counts(normalize=True).sort_index().to_dict()

    return {
        "type": "date",
        "null_rate": float(dt.isna().mean()),
        "count": int(len(data)),
        "min_date": _to_jsonable(min_date),
        "max_date": _to_jsonable(max_date),
        "date_range_days": date_range_days,
        "day_of_week_probs": {int(k): float(v) for k, v in dow.items()},
        "month_probs": {int(k): float(v) for k, v in month.items()},
        "day_of_month_probs": {int(k): float(v) for k, v in dom.items()},
    }


class InsuranceProductionDataAnalyzer:

    DEFAULT_CATEGORICAL = [
        "MULTIPRODICTDISCOUTN_TP",
        "INSUREDITEM_TP",
        "POLICYRATEDSTATE_TP",
        "RATEDCOUNTY_TP",
        "INTEGRATEDCOVERAGE_TP",
        "PROPERTYCOVERAGESUBTYPE_TP",
        "CREDITMODEL_CD",
        "SEASONAL_IN",
        "CONSTRUCTION_TP",
        "ROOF_TP",
    ]

    DEFAULT_NUMERICAL = [
        "DIRECTWRITTENPREMIUNM_AM",
        "PPCVRGLIMIT_AM",
        "DWELLINGSQUAREFEET_CT",
        "DWELLINGSTORY_CT",
        "EARNEDEXPOSURE_CT",
        "EARNEDPREMIUM_AM",
        "WRITENEXPOSURE_CT",
        "TAX_AM",
        "POLICYTERM_CT",
    ]

    DEFAULT_DATE = ["POLICYEFFECTIVE_DT"]

    def __init__(self, prod_data: pd.DataFrame, top_n: int = 25):
        self.prod_data = prod_data
        self.top_n = int(top_n)

        self.distributions: dict = {}

    def analyze_all_patterns(self):
        print("Starting frequency distribution analysis...")
        self._compute_frequency_distributions()
        print("Analysis complete!")
        return self

    def _infer_columns(self):
        df = self.prod_data

        categorical = [c for c in self.DEFAULT_CATEGORICAL if c in df.columns]
        numeric = [c for c in self.DEFAULT_NUMERICAL if c in df.columns]
        date_cols = [c for c in self.DEFAULT_DATE if c in df.columns]

        return categorical, numeric, date_cols

    def _compute_frequency_distributions(self):
        print("Computing frequency distributions...")

        categorical_cols, numeric_cols, date_cols = self._infer_columns()

        for col in categorical_cols:
            self.distributions[col] = _profile_categorical(self.prod_data[col], top_n=self.top_n)

        for col in numeric_cols:
            self.distributions[col] = _profile_numeric(self.prod_data[col])

        for col in date_cols:
            self.distributions[col] = _profile_date(self.prod_data[col])

    def generate_analysis_report(self, output_path: str = "insurance_analysis_report.json"):
        report = {
            "meta": {
                "rows": int(self.prod_data.shape[0]),
                "columns": int(self.prod_data.shape[1]),
            },
            "distributions": self.distributions,
        }

        output_path = Path(output_path)
        output_path.write_text(json.dumps(report, indent=2, default=_to_jsonable), encoding="utf-8")
        return report

    def print_summary(self):
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Rows: {len(self.prod_data):,}")
        print(f"Columns: {self.prod_data.shape[1]:,}")
        print(f"Profiled columns: {len(self.distributions):,}")
        print("=" * 80 + "\n")


# Edit these two paths and run the file.
CSV_PATH = "production_insurance_data.csv"
OUT_JSON_PATH = "insurance_analysis_report.json"


if __name__ == "__main__":
    print(f"Loading production data from: {CSV_PATH}")
    prod_data = pd.read_csv(CSV_PATH)

    analyzer = InsuranceProductionDataAnalyzer(prod_data, top_n=25)
    analyzer.analyze_all_patterns()
    analyzer.print_summary()
    analyzer.generate_analysis_report(OUT_JSON_PATH)

    print("\n✓ Done")
    
