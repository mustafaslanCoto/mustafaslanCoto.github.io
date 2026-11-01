#!/usr/bin/env python3
"""
update_point_comparison.py

Regenerates presentation/tables/point_comparison.csv from marimo cache files:
  - presentation/present_results/point_res_sing
  - presentation/present_results/point_res7
  - presentation/present_results/point_res_dict

Usage:
    python presentation/present_results/update_point_comparison.py
"""

from pathlib import Path
import glob
import pickle
import pandas as pd


def get_base_dir() -> Path:
    # Anchor to presentation directory
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


def build_point_comparison():
    pres_dir = get_base_dir()
    results_dir = pres_dir / "present_results"
    tables_dir = pres_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    res_sing = load_marimo_cache(results_dir / "point_res_sing")
    res_7 = load_marimo_cache(results_dir / "point_res7")
    res_dict = load_marimo_cache(results_dir / "point_res_dict")

    models = ["lgb", "xgb", "cat", "rf", "ridge"]
    rows = []

    for model in models:
        df_sing = (
            res_sing[model]
            .set_index("eval_metric")
            .loc["RMSSE"]
            .rename(lambda x: x.replace("_sing", ""))
        )
        df_7 = res_7[model].loc["RMSSE"].drop("overall", errors="ignore")
        df_dict = res_dict[model].loc["RMSSE"].drop("overall", errors="ignore")

        # Keep series alphabetically sorted to maintain consistency
        series_list = sorted(list(df_7.index))

        for s in series_list:
            clean_spec = s.replace("adm_", "").replace("dis_", "").replace("_", " ").title()
            flow_type = "Admission" if s.startswith("adm_") else "Discharge"

            val_sing = float(df_sing[s])
            val_7 = float(df_7[s])
            val_dict = float(df_dict[s])

            imp_m7 = ((val_sing - val_7) / val_sing) * 100
            imp_mdict = ((val_sing - val_dict) / val_sing) * 100

            rows.append({
                "series": s,
                "specialty_clean": clean_spec,
                "flow_type": flow_type,
                "model": model,
                "RMSSE_sing": val_sing,
                "RMSSE_multi_7": val_7,
                "RMSSE_multi_dict": val_dict,
                "RMSSE_multi": val_7,
                "imp_m7_pct": imp_m7,
                "imp_mdict_pct": imp_mdict,
                "improvement_pct": imp_m7,
            })

    df = pd.DataFrame(rows)

    out_csv = tables_dir / "point_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"Successfully wrote {len(df)} rows to {out_csv}")

    # Also sync to website if it exists
    website_tables_dir = Path("/Users/aslanm/Desktop/desktop/mustafaslanCoto.github.io/talks/wgsss2026_bangor/tables")
    if website_tables_dir.exists():
        website_csv = website_tables_dir / "point_comparison.csv"
        try:
            df.to_csv(website_csv, index=False)
            print(f"Successfully synced to website: {website_csv}")
        except PermissionError:
            print(f"Note: Could not directly write to website ({website_csv}) due to sandbox permissions.")

    return df


if __name__ == "__main__":
    build_point_comparison()
