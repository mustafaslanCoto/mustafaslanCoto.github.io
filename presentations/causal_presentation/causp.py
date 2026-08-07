# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly @ ./wheels/plotly-6.7.0-py3-none-any.whl",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", layout_file="layouts/causp.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import pyarrow

    return np, pd


@app.cell
def _():
    import io
    import urllib.request

    def fetch(path):
        """Read a data file into memory before handing it to pandas.

        Under WASM these files are served over HTTP, and GitHub Pages responds
        with `Content-Encoding: gzip`. The browser has already decompressed the
        body by the time pandas sees it, but pandas reads that header and
        gunzips a second time, raising BadGzipFile. Passing an in-memory buffer
        skips pandas' URL handling entirely. Local paths pass straight through.
        """
        location = str(path)
        if location.startswith(("http://", "https://")):
            with urllib.request.urlopen(location) as response:
                return io.BytesIO(response.read())
        return path

    return (fetch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Causal Impact of Diagnosis on LoS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grouping diagnosis

    ### Healthcare Resource Groups (HRGs) - Why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What are Healthcare Resource Groups (HRGs)?

    - Healthcare Resource Groups (HRGs) are designed to be standard groupings of clinically **similar treatments** which use **common levels of healthcare resource.**
    - Crucially, they are constructed to ensure **clinical meaningfulness** while accurately reflecting **expected resource consumption**.

    **How are they constructed?**
    A strict algorithm (the NHS Local Grouper) categorizes patients based on:
    *   **Diagnoses & Procedures** (ICD-10 and OPCS-4 codes)
    *   **Patient & Care Context** (Age, gender, and admission/discharge methods)

    **Why use HRGs over other grouping methods?**
    We initially evaluated standard clinical groupers like the **Elixhauser Comorbidity Index**. However, Elixhauser is designed for specific comorbidities and failed to categorize almost 50% of the diagnoses in our dataset. Because the NHS designed HRGs to account for total hospital activity, using HRGs guarantees that *every* admission is mapped to a clinically valid category that reflects the true intensity of care.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why Use HRGs for ICD-10 Dimensionality Reduction?
    In this study, we are estimating the causal impact of different diagnoses on **Length of Stay (LoS)**. Feeding raw diagnosis codes directly into our machine learning models (e.g., LightGBM) presents significant methodological challenges that HRGs perfectly solve:

    1.  **Solving Extreme Sparsity:** We have over 800 distinct ICD10 codes in our dataset. Modeling these directly results in an excessively sparse matrix, which destabilizes causal inference and inflates variance. Grouping them by their root HRG chapter collapses these 800+ codes into ~36 highly robust categories.
    2.  **Direct Alignment with the Target Variable:** Because HRGs were explicitly engineered by the NHS to measure *healthcare resource consumption*, they inherently correlate with Length of Stay (our primary resource metric). This makes them a highly effective latent representation of the diagnosis.
    3.  **Reproducibility & Interpretability:** Rather than using an opaque mathematical clustering technique (like PCA or autoencoders) to reduce dimensions, the HRG algorithm provides a transparent, clinically validated standard. The resulting effect sizes are directly interpretable by hospital capacity planners and stakeholders.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Anatomy of an HRG Code
    Before detailing the grouping strategy, it is helpful to understand how an HRG code is structured. The characters in an HRG represent different levels of clinical granularity:

    *   **2-Character HRGs (The Chapter/Sub-chapter):** This is the broader clinical umbrella. The first letter denotes the main body system or specialty (e.g., "F" for Digestive System), and the second letter denotes the sub-category. It provides a high-level grouping.
    *   **4-Character HRGs (The Base HRG):** This adds two numbers to the sub-chapter to identify a highly specific diagnosis or procedure group (e.g., "FD10" for benign colorectal neoplasms). It provides granular, highly specific clinical detail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Our Methodology: Frequency-Based ICD-10 to HRG Mapping
    To balance statistical power with clinical precision in our causal models, we did not apply a one-size-fits-all grouping. Instead, we grouped the 800+ raw ICD-10 codes based on their unique admission volumes:

    1.  **The Top 99% of Admissions (High Volume):**
        We mapped the most frequent ICD-10 codes to their highly specific **4-character HRG groups**. Because these conditions appear frequently in the dataset, our model has plenty of statistical power to handle them at a granular level.
    2.  **The Remaining 1% of Admissions (The Long Tail):**
        For rare, very low-volume ICD-10 codes, mapping them at the 4-character level would re-introduce the exact data sparsity problem we are trying to avoid. Instead, we rolled these rare codes up to their broader **2-character HRG chapters**.

    **Why do this?**
    This hybrid approach guarantees that **100% of the ICD-10 codes are captured and utilized** in the causal model. It preserves granular insights where we have the data to support it (the 99%), while safely capturing the long tail of rare diseases (the 1%) without destabilizing the machine learning algorithm.

    ---

    ### The Final Result: 36 Distinct Clinical Groups
    After applying this frequency-based mapping strategy, we successfully collapsed the 800+ raw ICD-10 codes into just **36 robust categories**.

    **The "Unknown" Cohort**
    Crucially, this final set of 36 includes an **"unknown" group**. This category represents admissions where patients were not assigned a valid diagnosis code during their stay (often due to diagnostic ambiguity, pending test results, or coding delays).

    It is important to highlight this group because it is substantial—accounting for approximately **28% of all admissions** in our dataset. Rather than dropping these records (which would introduce severe selection bias), retaining them as a distinct category allows our model to estimate the causal impact of *diagnostic uncertainty itself* on Length of Stay.
    """)
    return


@app.cell
def _(fetch, mo, pd):
    data_path = mo.notebook_location() / "public"
    order_df = pd.read_csv(fetch(data_path / "diagnosis_order.csv"), sep=None, engine='python')       # Use the more robust engine)
    diags = pd.read_csv(fetch(data_path / "caus_hrg_lgb.csv"), sep=None, engine='python')
    diags = diags.merge(order_df, on='profile', how ='left')
    diags.rename(columns={"proportion": "dominance"}, inplace=True)
    # diags["dominance"] = diags["dominance"]*100
    cleand_df = pd.read_parquet(fetch(data_path /"clean_df_present.parquet"))
    # cleand_df.to_csv(data_path / "clean_df_present.csv", index=False)
    # cleand_df = pd.read_csv(data_path / "clean_df_present.csv")
    exist_codes = cleand_df[cleand_df["code"]!= "$$X"]["code"].drop_duplicates().tolist()

    diags = diags.drop(columns=["variance"])

    # the desired order as a list
    _order = order_df["profile"].tolist()   # adjust column name to match your file

    # make diags' profile column an ordered categorical
    diags["profile"] = pd.Categorical(diags["profile"], categories=_order, ordered=True)

    # sort by it
    diags = diags.sort_values("profile").reset_index(drop=True)
    return cleand_df, data_path, diags, exist_codes


@app.cell
def _(diags):
    hrg_est = diags["profile"].tolist()
    return (hrg_est,)


@app.cell
def _(data_path, exist_codes, fetch, pd):
    hrg = pd.read_csv(fetch(data_path / "nhs_group.csv"))[["code", "HRG 1","Code Description"]].drop_duplicates().rename(columns={"code":"ICD_code", "HRG 1": "HRG", "Code Description": "ICD_description"})
    ## filter ICD
    hrg = hrg[hrg["ICD_code"].isin(exist_codes)]
    hrg["HRG2"] = hrg["HRG"].str[:2]
    return (hrg,)


@app.cell
def _(hrg, hrg_est, np):
    hrg["isin_est"] = hrg["HRG2"].isin(hrg_est) | hrg["HRG"].isin(hrg_est)
    hrg_in = hrg[hrg["isin_est"] == True].copy()
    ## create another column if the value of HRG is in the estimated list take it from HRG column otwerwise take it from HRG2 column
    hrg_in.loc[:, "HRG_final"] = np.where(
                hrg_in["HRG"].isin(hrg_est), 
                hrg_in["HRG"], 
                hrg_in["HRG2"]
    )
    return (hrg_in,)


@app.cell
def _():
    # mo.ui.table(diags.round(2), pagination=True, page_size=15)
    # cleand_df[cleand_df["code"]!="$$X"]["code"].value_counts(normalize=True).cumsum()
    return


@app.cell
def _(mo):
    # Cell 1
    hrg_search = mo.ui.text(
        placeholder="Type an HRG group (e.g. FD10)",
        label="Look up ICD codes for HRG group:",
    )
    return (hrg_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Causal effect of diagnosis (HRGs) on LoS
    """)
    return


@app.cell
def _():
    # _q = hrg_search.value.strip().upper()

    # # left: results (always shown)
    # _left = mo.vstack([
    #     mo.md("### Effect Sizes of HRGs on LoS"),
    #     mo.ui.table(diags.round(2), pagination=True, page_size=15),
    # ])

    # # _left = mo.ui.table(diags.round(2), pagination=True, page_size=15)

    # # right: descriptions (only when a group is typed)
    # if not _q:
    #     _right = mo.md("*Type an HRG group to see its ICD codes here.*")
    # else:
    #     _filtered = hrg_in[hrg_in["HRG_final"].astype(str).str.upper() == _q][
    #         ["ICD_code", "ICD_description"]
    #     ].reset_index(drop=True)
    #     if len(_filtered) == 0:
    #         _right = mo.md(f"*No ICD codes for **{_q}**.*")
    #     else:
    #         _right = mo.vstack([
    #             mo.md(f"### {_q} — {len(_filtered)} ICD codes"),
    #             mo.ui.table(_filtered, pagination=True, page_size=15)
    #             # mo.plain(_filtered),
    #         ])

    # mo.vstack([
    #     hrg_search,
    #     mo.hstack([_left, _right], widths=[0.6, 0.4], gap=1, align="start"),
    # ])
    return


@app.cell
def _(diags, hrg_in, hrg_search, mo, np, pd):
    import plotly.graph_objects as go


    def create_forest_plot(
        df: pd.DataFrame, top_n: int = 20
    ) -> go.Figure | mo.Html:
        """Generates a production-ready interactive Forest Plot for causal effect sizes.

        Incorporates categorical diagnosis dominance (%) via marker scale and hover text.

        Parameters
        ----------
        df : pd.DataFrame
            Data containing 'profile', 'ate_days', 'ci_low', 'ci_high', and
            'dominance'.
        top_n : int, optional
            Maximum number of HRG profiles to display, sorted by absolute ATE
            magnitude.

        Returns
        -------
        go.Figure | mo.Html
            Interactive Plotly Figure or Marimo HTML error component.
        """
        if df.empty:
            return mo.md("*No effect size data available.*")

        # Guard against missing values & enforce column typing for performance
        required_cols = ["profile", "ate_days", "ci_low", "ci_high", "dominance"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return mo.md(f"*Missing required columns: `{list(missing_cols)}`*")

        # Vectorized subset filtering, absolute value calculation, and top-N ranking
        clean_df = (
            df.dropna(subset=required_cols)
            .assign(
                abs_ate=lambda x: x["ate_days"].abs(),
                dominance_pct=lambda x: x["dominance"] * 100,
            )
            .sort_values(by="abs_ate", ascending=True)  # Bottom-to-top layout
            .tail(top_n)
            .reset_index(drop=True)
        )

        if clean_df.empty:
            return mo.md("*No valid numeric rows found after cleaning.*")

        # Vectorized color assignment based on directional effect
        marker_colors = np.where(
            clean_df["ate_days"] > 0,
            "#E53E3E",  # Red: Increases Length of Stay
            "#319795",  # Teal: Decreases Length of Stay
        )

        # Vectorized calculation of error bar vectors
        error_x_minus = clean_df["ate_days"] - clean_df["ci_low"]
        error_x_plus = clean_df["ci_high"] - clean_df["ate_days"]

        # Vectorized scaling of marker size proportional to dominance (min 8px, max 24px)
        dom_min = clean_df["dominance_pct"].min()
        dom_max = clean_df["dominance_pct"].max()

        if dom_max > dom_min:
            scaled_marker_sizes = (
                8 + 16 * (clean_df["dominance_pct"] - dom_min) / (dom_max - dom_min)
            ).to_numpy()
        else:
            scaled_marker_sizes = np.full(len(clean_df), 12)

        fig = go.Figure()

        # Forest plot points, confidence interval bounds, and dominance scaling
        fig.add_trace(
            go.Scatter(
                x=clean_df["ate_days"],
                y=clean_df["profile"].astype(str),
                mode="markers",
                marker=dict(
                    size=scaled_marker_sizes,
                    color=marker_colors,
                    line=dict(width=1, color="#2D3748"),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=error_x_plus,
                    arrayminus=error_x_minus,
                    color="#4A5568",
                    thickness=2,
                    width=5,
                ),
                hovertemplate=(
                    "<b>HRG Profile:</b> %{y}<br>"
                    "<b>ATE (Days):</b> %{x:.2f}<br>"
                    "<b>95%% CI:</b> [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<br>"
                    "<b>Dominance (Frequency):</b> %{customdata[2]:.2f}%%"
                    "<extra></extra>"
                ),
                customdata=clean_df[
                    ["ci_low", "ci_high", "dominance_pct"]
                ].to_numpy(),
            )
        )

        # Configure top X-axis positioning, transparent backgrounds, and reference baseline
        fig.update_layout(
            title=dict(
                # text="Effect Sizes of HRGs on LoS (Point size = Dominance %)",
                font=dict(size=14, color="#2D3748"),
                pad=dict(b=20),
            ),
            xaxis_title="Causal Effect Sizes of Diagnosis Groups on LoS (Days) - (Point size = Dominance % in data)",
            yaxis_title="HRG Profile",
            margin=dict(l=60, r=30, t=80, b=30),
            height=max(240, len(clean_df) * 28),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(237, 242, 247, 0.45)",
            xaxis=dict(
                side="top",
                showgrid=True,
                gridcolor="rgba(160, 174, 192, 0.35)",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="#718096",
                zerolinewidth=1.5,
                ticks="outside",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(160, 174, 192, 0.35)",
                gridwidth=1,
                type="category",
                ticks="outside",
            ),
            hoverlabel=dict(bgcolor="#1A202C", font_color="#FFFFFF", font_size=12),
        )

        return mo.ui.plotly(fig)


    # --- Marimo Reactive Cell Logic ---
    _q = hrg_search.value.strip().upper()

    # Interactive left-hand panel
    _left = mo.vstack([create_forest_plot(diags, top_n=36)])

    # Right-hand panel logic
    if not _q:
        _right = mo.md("*Type an HRG group to see its ICD codes here.*")
    else:
        _filtered = hrg_in.loc[
            hrg_in["HRG_final"].astype(str).str.upper() == _q,
            ["ICD_code", "ICD_description"],
        ].reset_index(drop=True)

        if _filtered.empty:
            _right = mo.md(f"*No ICD codes found for **{_q}**.*")
        else:
            _right = mo.vstack(
                [
                    mo.md(f"### {_q} — {len(_filtered)} ICD codes"),
                    mo.ui.table(_filtered, pagination=True, page_size=15),
                ]
            )

    mo.vstack(
        [
            hrg_search,
            mo.hstack([_left, _right], widths=[0.55, 0.45], gap=2, align="start"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Attributable bed-days from the uncoded group and codes increasing LoS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reducing diagnostic ambiguity — faster coding, resolving ungroupable admissions — could free up the bed-days
    """)
    return


@app.cell
def _():
    # cleand_df[cleand_df["code"] == "$$X"]["PATIENT_SPECIALTY"].value_counts(normalize=True)
    # cleand_df[cleand_df["code"] != "$$X"]["PATIENT_SPECIALTY"].value_counts(normalize=True)
    return


@app.cell
def _(mo):
    # Create the UI elements
    my_options = {
        "📁 Uncoded Group (+3.34 days - 28.7% dominance)": "unknown", 
        "🏥 HE11 (7.44 days - 1.5% dominance)": "HE11",
        "🏥 HE (0.78 days - 1.8% dominance)": "HE",
        "⭐ Combined (All 3 Groups)": "combined"
    }

    label_style = "font-size: 18px; font-weight: 600; color: #1e293b;"

    group_select = mo.ui.dropdown(
        options=my_options,
        value="📁 Uncoded Group (+3.34 days - 28.7% dominance)",
        label=f"<span style='{label_style}'>Select Diagnosis Group:</span>"
    )

    # 2. Causal Estimator Metric Selection (Point Estimate vs Uncertainty Bounds)
    metric_options = {
        "📊 Point Estimate (ATE - Mean Days)": "ate_days",
        "📉 Conservative Estimate (CI Lower Bound)": "ci_low",
        "📈 Aggressive Estimate (CI Upper Bound)": "ci_high",
    }

    metric_select = mo.ui.dropdown(
        options=metric_options,
        value="📊 Point Estimate (ATE - Mean Days)",
        label=f"<span style='{label_style}'>Select Causal Estimator:</span>",
    )

    success_rate = mo.ui.slider(
        start=0, 
        stop=100, 
        step=5, 
        value=100, 
        label=f"<span style='{label_style}'>Ambiguity Resolution Success Rate (%):</span>"
    )

    # controls_display

    custom_css = mo.Html("""
    <style>
        /* Scale up the dropdown options text */
        select { 
            font-size: 16px !important; 
            padding: 6px 10px !important; 
            cursor: pointer;
        }

        /* Slightly enlarge the slider track for better UX */
        input[type=range] { 
            transform: scale(1.1); 
            margin-left: 8px; 
            cursor: pointer;
        }
    </style>
    """)

    # 4. Container Styling
    controls_display = mo.hstack(
        [group_select, metric_select, success_rate], 
        justify="start", 
        gap=4
    ).style(
        style={
            "padding": "24px", 
            "background-color": "#f8fafc", 
            "border-radius": "12px",
            "border": "1px solid #e2e8f0",
            "align-items": "center" # Keeps the text and sliders vertically aligned
        }
    )

    # Output both the hidden CSS block and the UI container
    mo.vstack([custom_css, controls_display])

    # # Just display the UI in this cell
    # mo.hstack([group_select, success_rate], justify="start", gap=2)
    return group_select, metric_select, success_rate


@app.cell
def _(cleand_df, diags, group_select, metric_select, mo, success_rate):
    # Counterfactual calculations (without leading underscores)
    test_cut = "2025-03-01"
    clean_test = cleand_df[cleand_df["HS_START_DATE"] >= test_cut]
    total_admits = clean_test.shape[0]
    # 1. Filter the dataset based on dropdown selection
    selected_group = group_select.value

    if selected_group == "combined":
        active_diag = clean_test[clean_test["group"].isin(["unknown", "HE11", "HE"])]
        group_label = "Combined Groups"

        # Calculate excess for each subgroup based on its specific baseline
        excess_los = 0
        for profile in ["unknown", "HE11", "HE"]:
            sub_group = active_diag[active_diag["group"] == profile]
            len_sub = sub_group.shape[0]
            if not sub_group.empty:
                baseline = diags.loc[diags["profile"] == profile, metric_select.value].values[0]
                # excess_los += (sub_group["spell_los"] - baseline).clip(lower=0).sum()
                excess_los += len_sub * baseline
    else:
        active_diag = clean_test[clean_test["group"] == selected_group]
        len_act = active_diag.shape[0]
        group_label = "Uncoded Group" if selected_group == "unknown" else selected_group
        baseline_los = diags.loc[diags["profile"] == selected_group, metric_select.value].values[0]
        # excess_los = (active_diag["spell_los"] - baseline_los).clip(lower=0).sum()
        excess_los = len_act*baseline_los
    num_admits = active_diag.shape[0]

    # Calculate admission ratio
    admits_pct = (num_admits / total_admits * 100.0) if total_admits > 0 else 0.0
    # 2. Compute excess Length of Stay metrics
    # mu = diags.loc[0, "ate_days"]
    # se = diags.loc[0, "se"]

    # # Generate random samples
    # baseline_los = np.random.normal(mu, se, len(active_diag))
    total_group_los = active_diag["spell_los"].sum()

    # We clip the excess LoS at 0 so patients below baseline don't subtract from the savings
    # Apply the success rate
    rate = success_rate.value / 100.0
    savings_bed_days = excess_los * rate

    # 3. Calculate percentages
    pct_saved_group = (savings_bed_days / total_group_los * 100.0) if total_group_los > 0 else 0.0

    total_test_cohort_los = clean_test["spell_los"].sum()
    pct_saved_total = (savings_bed_days / total_test_cohort_los * 100.0) if total_test_cohort_los > 0 else 0.0


    # 4. Generate the styled cards dynamically
    style_base = {
        "padding": "16px",
        "border-radius": "12px",
        "text-align": "center",
        "flex": "1",
        "min-width": "180px",
    }

    card_total_group = mo.md(
        f"""
        ### 📁 **{total_group_los:,.0f}**
        **Total {group_label} Bed-Days**
        """
    ).style(
        style={
            **style_base,
            "border": "2px solid #8c8c8c",
            "background-color": "#fafafa",
            "color": "#1a1a1a",
        }
    )

    card_savings = mo.md(
        f"""
        ### 🛏️ **{savings_bed_days:,.0f}**
        **Bed-Days Saved**
        """
    ).style(
        style={
            **style_base,
            "border": "2px solid #007bc0",
            "background-color": "#f4f9fc",
            "color": "#004d7a",
        }
    )

    card_group_pct = mo.md(
        f"""
        ### 🎯 **{pct_saved_group:.1f}%**
        **Saved ({group_label})**
        """
    ).style(
        style={
            **style_base,
            "border": "2px solid #d97706",
            "background-color": "#fffbeb",
            "color": "#78350f",
        }
    )

    card_total = mo.md(
        f"""
        ### 📈 **{pct_saved_total:.1f}%**
        **Saved (Total Cohort)**
        """
    ).style(
        style={
            **style_base,
            "border": "2px solid #16a34a",
            "background-color": "#f0fdf4",
            "color": "#14532d",
        }
    )


    card_admits = mo.md(
        f"""
        ### 🏥 **Cohort Analysis: {num_admits:,} {group_label} Admissions** out of {total_admits:,} Total
        **Representing {admits_pct:.1f}% of the test cohort**
        """
    ).style(
        style={
            **style_base,
            "border": "2px solid #6366f1",
            "background-color": "#f5f3ff", # Very light purple
            "color": "#4338ca",
            "flex": "1", # This will make it stretch to fill the width
        }
    )

    # 5. Assemble the dashboard layout
    header_label = "Combined High-LoS Groups" if selected_group == "combined" else f"Group '{selected_group}'"
    dashboard_header = mo.md(f"#### 🔍 Counterfactual Scenario Results for **{header_label}**")

    # Assemble the dashboard layout
    dashboard_row = mo.hstack(
        [card_total_group, card_savings, card_group_pct, card_total], 
        justify="space-between", 
        align="stretch",
        gap=1.0 # Reduced gap slightly to fit 5 cards better
    )

    # Wrap controls in a light grey box for better structure
    # control_panel = mo.md("").style(
    #     style={
    #         "background-color": "#f9f9f9", 
    #         "padding": "20px", 
    #         "border-radius": "10px", 
    #         "margin-bottom": "20px"
    #     }
    # )
    # Display header above the metric row
    mo.vstack([dashboard_header, card_admits, dashboard_row])
    return


@app.cell
def _():
    # # Sample from a distribution with mean and variance
    # import matplotlib.pyplot as plt
    # mu = diags.loc[0, "ate_days"]
    # se = diags.loc[0, "se"]

    # # Generate random samples
    # samples = np.random.normal(mu, se, 1000)
    # ## plot the distribution
    # plt.hist(samples, bins=30, edgecolor='black')
    # plt.title("Distribution of Samples")
    # plt.xlabel("Days")
    # plt.ylabel("Frequency")
    # plt.show()
    return


@app.cell
def _():
    # _selected_group = "unknown"
    # _active_diag = clean_test[clean_test["group"] == _selected_group]
    # baseline_tst = diags.loc[diags["profile"] == _selected_group, "ci_low"].values[0]
    # excess_tst = (active_diag["spell_los"] - baseline_tst)
    return


@app.cell
def _():
    # unkn = clean_test[clean_test["group"] == 'unknown'].groupby("PATIENT_SPECIALTY")["ADMIT_NUMBER"].nunique().sort_values(ascending=False)
    # knw = clean_test[clean_test["group"] != 'unknown'].groupby("PATIENT_SPECIALTY")["ADMIT_NUMBER"].nunique().sort_values(ascending=False)
    return


@app.cell
def _():
    # unkn*100/unkn.sum()
    return


@app.cell
def _():
    # knw*100/knw.sum()
    return


if __name__ == "__main__":
    app.run()
