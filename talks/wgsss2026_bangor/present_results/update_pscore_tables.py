#!/usr/bin/env python3
"""
update_pscore_tables.py

Regenerates probabilistic forecasting tables from marimo cache files:
  - presentation/present_results/pscore_dict_mod_sing
  - presentation/present_results/pscore_multi_7
  - presentation/present_results/pscore_multi_dict

Outputs generated:
  - presentation/tables/pscore_series_summary.csv
  - presentation/tables/pscore_summary.csv

Usage:
    python presentation/present_results/update_pscore_tables.py
"""

from pathlib import Path
import glob
import pickle
import numpy as np
import pandas as pd


def get_base_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "present_results":
        return script_dir.parent
    return Path.cwd() / "presentation"


def load_marimo_cache(folder_path: Path) -> dict:
    pickle_files = glob.glob(str(folder_path / "*.pickle"))
    if not pickle_files:
        raise FileNotFoundError(f"No pickle file found in {folder_path}")
    with open(pickle_files[0], "rb") as f:
        cache = pickle.load(f)
    return list(cache.defs.values())[0]


def build_pscore_tables():
    pres_dir = get_base_dir()
    results_dir = pres_dir / "present_results"
    tables_dir = pres_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pscore pickle caches...")
    sing_data = load_marimo_cache(results_dir / "pscore_dict_mod_sing")
    m7_data = load_marimo_cache(results_dir / "pscore_multi_7")
    mdict_data = load_marimo_cache(results_dir / "pscore_multi_dict")

    models = ["lgb", "xgb", "cat", "rf", "ridge"]
    horizon_ranges = {
        "All (1-42)": (1, 42),
        "1-7": (1, 7),
        "8-14": (8, 14),
        "15-21": (15, 21),
        "22-42": (22, 42),
    }

    # -------------------------------------------------------------
    # 1. Build pscore_series_summary.csv
    # -------------------------------------------------------------
    print("Building pscore_series_summary.csv...")
    series_rows = []

    # Get specialty list from first model
    sample_model = models[0]
    specialty_list = list(sing_data[sample_model].keys())

    for model in models:
        for specialty in specialty_list:
            clean_spec = specialty.replace("adm_", "").replace("dis_", "").replace("_", " ").title()
            flow_type = "Admission" if specialty.startswith("adm_") else "Discharge"

            for range_name, (h_start, h_end) in horizon_ranges.items():
                sing_vals = [
                    sing_data[model][specialty][fold][h]
                    for fold in sing_data[model][specialty]
                    for h in range(h_start, h_end + 1)
                ]
                m7_vals = [
                    m7_data[model][fold][specialty][h]
                    for fold in m7_data[model]
                    for h in range(h_start, h_end + 1)
                ]
                mdict_vals = [
                    mdict_data[model][fold][specialty][h]
                    for fold in mdict_data[model]
                    for h in range(h_start, h_end + 1)
                ]

                ps_sing = float(np.mean(sing_vals))
                ps_m7 = float(np.mean(m7_vals))
                ps_mdict = float(np.mean(mdict_vals))

                imp_m7 = ((ps_sing - ps_m7) / ps_sing) * 100.0
                imp_mdict = ((ps_sing - ps_mdict) / ps_sing) * 100.0

                series_rows.append({
                    "model": model,
                    "specialty": specialty,
                    "specialty_clean": clean_spec,
                    "flow_type": flow_type,
                    "horizon_range": range_name,
                    "ps_sing": ps_sing,
                    "ps_multi_7": ps_m7,
                    "ps_multi_dict": ps_mdict,
                    "imp_m7_pct": imp_m7,
                    "imp_mdict_pct": imp_mdict,
                })

    df_series = pd.DataFrame(series_rows)
    series_csv = tables_dir / "pscore_series_summary.csv"
    df_series.to_csv(series_csv, index=False)
    print(f"Successfully wrote {len(df_series)} rows to {series_csv}")

    # -------------------------------------------------------------
    # 2. Build pscore_summary.csv (fold-level scores)
    # -------------------------------------------------------------
    print("Building pscore_summary.csv...")
    fold_ranges = {
        "1-7": (1, 7),
        "8-14": (8, 14),
        "15-21": (15, 21),
        "22-42": (22, 42),
    }

    fold_rows = []
    # Match structure: model -> specialty -> horizon_range -> fold (Single, Multi_7, Multi_Dict)
    for model in models:
        for specialty in specialty_list:
            for range_name, (h_start, h_end) in fold_ranges.items():
                folds = sorted(list(sing_data[model][specialty].keys()))
                for fold in folds:
                    score_sing = float(np.mean([
                        sing_data[model][specialty][fold][h]
                        for h in range(h_start, h_end + 1)
                    ]))
                    score_m7 = float(np.mean([
                        m7_data[model][fold][specialty][h]
                        for h in range(h_start, h_end + 1)
                    ]))
                    score_mdict = float(np.mean([
                        mdict_data[model][fold][specialty][h]
                        for h in range(h_start, h_end + 1)
                    ]))

                    fold_rows.append({
                        "pscore": score_sing,
                        "model": model,
                        "method": "Single",
                        "specialty": specialty,
                        "horizon_range": range_name,
                        "fold": fold,
                    })
                    fold_rows.append({
                        "pscore": score_m7,
                        "model": model,
                        "method": "Multi_7",
                        "specialty": specialty,
                        "horizon_range": range_name,
                        "fold": fold,
                    })
                    fold_rows.append({
                        "pscore": score_mdict,
                        "model": model,
                        "method": "Multi_Dict",
                        "specialty": specialty,
                        "horizon_range": range_name,
                        "fold": fold,
                    })

    df_fold = pd.DataFrame(fold_rows)
    fold_csv = tables_dir / "pscore_summary.csv"
    df_fold.to_csv(fold_csv, index=False)
    print(f"Successfully wrote {len(df_fold)} rows to {fold_csv}")

    # -------------------------------------------------------------
    # 3. Optional Website Sync
    # -------------------------------------------------------------
    website_tables_dir = Path("/Users/aslanm/Desktop/desktop/mustafaslanCoto.github.io/talks/wgsss2026_bangor/tables")
    if website_tables_dir.exists():
        try:
            df_series.to_csv(website_tables_dir / "pscore_series_summary.csv", index=False)
            df_fold.to_csv(website_tables_dir / "pscore_summary.csv", index=False)
            print("Successfully synced pscore tables to website directory.")
        except PermissionError:
            print("Note: Website sync skipped due to sandbox permissions.")

    return df_series, df_fold


if __name__ == "__main__":
    build_pscore_tables()

