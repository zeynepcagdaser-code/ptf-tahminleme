from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.config import PROJECT_ROOT


PTF_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_clean.csv"
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "external_features" / "standardized_selected_features.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "feature_compatibility"


def run_feature_ptf_compatibility_analysis() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = build_ptf_feature_matrix()
    report = analyze_feature_matrix(merged)
    save_reports(report)
    print_summary(report)
    return report


def build_ptf_feature_matrix() -> pd.DataFrame:
    ptf = pd.read_csv(PTF_PATH)
    ptf["datetime"] = pd.to_datetime(ptf["datetime"], errors="coerce")
    ptf = ptf.dropna(subset=["datetime", "ptf"]).sort_values("datetime")

    features = pd.read_csv(FEATURE_PATH)
    if features.empty:
        return ptf[["datetime", "ptf"]].copy()
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    features = features.dropna(subset=["datetime", "feature_name", "feature_value"])
    features["feature_value"] = pd.to_numeric(features["feature_value"], errors="coerce")
    features = features.dropna(subset=["feature_value"])

    wide = (
        features.groupby(["datetime", "feature_name"], as_index=False)["feature_value"].mean()
        .pivot(index="datetime", columns="feature_name", values="feature_value")
        .reset_index()
    )
    merged = ptf[["datetime", "ptf"]].merge(wide, on="datetime", how="left")
    merged.to_csv(PROJECT_ROOT / "data" / "processed" / "external_features" / "ptf_with_selected_features.csv", index=False)
    return merged


def analyze_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [
        c for c in df.columns
        if c not in {"datetime", "ptf"} and not c.startswith("ptf__") and c != "ptf"
    ]
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df["ptf"], errors="coerce")
        valid = x.notna() & y.notna()
        missing_ratio = float(1 - valid.mean())
        zero_ratio = float((x[valid] == 0).mean()) if valid.any() else np.nan
        variance = float(x[valid].var()) if valid.sum() > 1 else 0.0
        pearson = safe_corr(x, y, "pearson")
        spearman = safe_corr(x, y, "spearman")
        lag_1 = safe_corr(x.shift(1), y, "pearson")
        lag_24 = safe_corr(x.shift(24), y, "pearson")
        lag_168 = safe_corr(x.shift(168), y, "pearson")
        mi = safe_mutual_information(x, y)
        r2 = safe_univariate_r2(x, y)
        decision = decide_feature(missing_ratio, variance, pearson, spearman, max_abs([lag_1, lag_24, lag_168]), mi, r2)
        rows.append(
            {
                "feature_name": col,
                "missing_ratio": missing_ratio,
                "zero_ratio": zero_ratio,
                "variance": variance,
                "pearson": pearson,
                "spearman": spearman,
                "lag_1_corr": lag_1,
                "lag_24_corr": lag_24,
                "lag_168_corr": lag_168,
                "mutual_information": mi,
                "simple_univariate_r2": r2,
                "decision": decision,
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report["score"] = (
            report[["pearson", "spearman", "lag_1_corr", "lag_24_corr", "lag_168_corr"]].abs().max(axis=1).fillna(0)
            + report["mutual_information"].fillna(0)
            + report["simple_univariate_r2"].fillna(0)
            - report["missing_ratio"].fillna(1)
        )
        report = report.sort_values("score", ascending=False).reset_index(drop=True)
    return report


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 10 or x[valid].nunique() < 2:
        return np.nan
    return float(x[valid].corr(y[valid], method=method))


def safe_mutual_information(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 20 or x[valid].nunique() < 2:
        return np.nan
    return float(mutual_info_regression(x[valid].to_frame(), y[valid], random_state=42)[0])


def safe_univariate_r2(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 20 or x[valid].nunique() < 2:
        return np.nan
    model = LinearRegression()
    model.fit(x[valid].to_frame(), y[valid])
    return float(r2_score(y[valid], model.predict(x[valid].to_frame())))


def max_abs(values: list[float]) -> float:
    clean = [abs(v) for v in values if pd.notna(v)]
    return max(clean) if clean else 0.0


def decide_feature(missing: float, variance: float, pearson: float, spearman: float, lag_corr: float, mi: float, r2: float) -> str:
    relation = max_abs([pearson, spearman, lag_corr])
    mi = 0 if pd.isna(mi) else mi
    r2 = 0 if pd.isna(r2) else r2
    if missing > 0.70:
        return "cok_eksik_elenecek"
    if variance == 0 or pd.isna(variance):
        return "dusuk_varyans_elenecek"
    if lag_corr >= 0.25 and lag_corr > relation * 0.8:
        return "lag_ile_kullanilabilir"
    if relation >= 0.30 or r2 >= 0.08 or mi >= 0.05:
        return "kullanilabilir"
    if relation >= 0.12 or mi >= 0.02:
        return "zayif_ama_tutulabilir"
    if missing > 0.35:
        return "manuel_incelenmeli"
    return "iliski_yok_elenecek"


def save_reports(report: pd.DataFrame) -> None:
    report.to_csv(OUT_DIR / "feature_report.csv", index=False)
    report.to_excel(OUT_DIR / "feature_report.xlsx", index=False)
    selected = report[report["decision"].isin(["kullanilabilir", "lag_ile_kullanilabilir", "zayif_ama_tutulabilir"])]
    dropped = report[report["decision"].str.contains("elenecek", na=False)]
    manual = report[report["decision"].eq("manuel_incelenmeli")]
    selected.to_csv(OUT_DIR / "selected_features.csv", index=False)
    dropped.to_csv(OUT_DIR / "dropped_features.csv", index=False)
    manual.to_csv(OUT_DIR / "manual_review_features.csv", index=False)


def print_summary(report: pd.DataFrame) -> None:
    print("\nPTF uyum analizi ozeti")
    print("-" * 40)
    print(f"Analiz edilen feature: {len(report):,}")
    if report.empty:
        return
    print(report["decision"].value_counts().to_string())
    print("\nEn guclu 20 feature")
    print(report.head(20)[["feature_name", "score", "pearson", "spearman", "lag_24_corr", "mutual_information", "simple_univariate_r2", "decision"]].to_string(index=False))
