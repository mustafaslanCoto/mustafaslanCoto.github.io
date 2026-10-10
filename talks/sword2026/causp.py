# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "mcp==2.0.0",
#     "numpy",
#     "pandas",
#     "plotly==6.7.0",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="full",
    css_file="css/custom.css",
    html_head_file="",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    import re
    import sys

    _RAW = "https://raw.githubusercontent.com/mustafaslanCoto/mustafaslanCoto.github.io/main/talks"

    if sys.platform == "emscripten":
        # exported HTML: read the include and logos straight from the repo
        from pyodide.http import open_url

        _html = open_url(f"{_RAW}/sword2026/title-slide.html").read()
        _src = lambda m: f'src="{_RAW}/{m.group(1)}"'
    else:
        import base64
        import mimetypes

        _dir = mo.notebook_dir()
        _html = (_dir / "title-slide.html").read_text()

        def _src(m):
            _p = _dir.parent / m.group(1)
            _mime = mimetypes.guess_type(_p.name)[0] or "image/png"
            return f'src="data:{_mime};base64,{base64.b64encode(_p.read_bytes()).decode()}"'

    # drop Quarto's ```{=html} fences; logos live in talks/images
    _html = re.sub(r"^```.*$", "", _html, flags=re.M)
    _html = re.sub(r'src="(images/[^"]+)"', _src, _html)
    mo.Html(_html)
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
def _():
    import io as _io
    from urllib.request import urlopen as _urlopen

    import pandas as pd
    import numpy as np

    def fetch(url):
        """Fetch remote data into memory before passing it to pandas.

        Passing an in-memory buffer avoids pandas handling the compressed HTTP
        response itself in the browser.
        """
        with _urlopen(url) as response:
            return _io.BytesIO(response.read())


    data_url = (
        "https://raw.githubusercontent.com/"
        "mustafaslanCoto/mustafaslanCoto.github.io/main/"
        "talks/sword2026/public"
    )
    order_df = pd.read_csv(fetch(f"{data_url}/diagnosis_order.csv"), sep=None, engine='python')
    diags = pd.read_csv(fetch(f"{data_url}/caus_hrg_lgb.csv"), sep=None, engine='python')
    diags = diags.merge(order_df, on='profile', how ='left')
    diags.rename(columns={"proportion": "dominance"}, inplace=True)
    diags.sort_values("dominance", ascending=False, inplace=True)
    # diags["dominance"] = diags["dominance"]*100
    cleand_df = pd.read_parquet(fetch(f"{data_url}/clean_df_present.parquet"))
    exist_codes = cleand_df[cleand_df["code"]!= "$$X"]["code"].drop_duplicates().tolist()

    diags = diags.drop(columns=["variance"])

    # the desired order as a list
    _order = order_df["profile"].tolist()   # adjust column name to match your file

    # make diags' profile column an ordered categorical
    diags["profile"] = pd.Categorical(diags["profile"], categories=_order, ordered=True)

    # sort by it
    diags = diags.sort_values("profile").reset_index(drop=True)

    confounders = pd.read_csv(fetch(f"{data_url}/confounders.csv"), sep=None, engine='python')
    return cleand_df, confounders, data_url, diags, exist_codes, fetch, np, pd


@app.cell
def _(diags):
    hrg_est = diags["profile"].tolist()
    return (hrg_est,)


@app.cell
def _(data_url, exist_codes, fetch, pd):
    hrg = pd.read_csv(fetch(f"{data_url}/nhs_group.csv"))[["code", "HRG 1","Code Description"]].drop_duplicates().rename(columns={"code":"ICD_code", "HRG 1": "HRG", "Code Description": "ICD_description"})
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confounders
    """)
    return


@app.cell
def _(confounders):
    confounders.drop(columns=["nf", "under"]).rename(columns={"c": "confounders"})
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
def _(diags, mo):
    # 1. Dynamically build options for all groups with positive ATE days
    pos_diags = diags[diags["ate_days"] > 0].copy()

    my_options = {}
    for _, row in pos_diags.iterrows():
        prof = str(row["profile"])
        ate = row["ate_days"]
        dom = row["dominance"] * 100 if row["dominance"] <= 1 else row["dominance"]
        icon = "📁" if prof == "unknown" else "🏥"
        label_text = "Uncoded Group" if prof == "unknown" else prof

        display_str = f"{icon} {label_text} (+{ate:.2f} days - {dom:.1f}% dominance)"
        my_options[display_str] = prof

    # Add combined option at the end for all positive groups
    my_options[f"⭐ Combined (All {len(pos_diags)} Positive Groups)"] = "combined"

    label_style = "font-size: 18px; font-weight: 600; color: #1e293b;"

    group_select = mo.ui.dropdown(
        options=my_options,
        value=list(my_options.keys())[0],
        label=f"<span style='{label_style}'>Select Diagnosis Group:</span>"
    )

    # 2. Causal Estimator Metric Selection
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

    _custom_css = mo.Html("""
    <style>
        select { 
            font-size: 16px !important; 
            padding: 6px 10px !important; 
            cursor: pointer;
        }
        input[type=range] { 
            transform: scale(1.1); 
            margin-left: 8px; 
            cursor: pointer;
        }
    </style>
    """)

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
            "align-items": "center"
        }
    )

    mo.vstack([_custom_css, controls_display])

    # # Just display the UI in this cell
    # mo.hstack([group_select, success_rate], justify="start", gap=2)
    return group_select, metric_select, success_rate


@app.cell
def _(cleand_df, diags, test_cut):
    ## filter rows where there is no 0 between ci_low and ci_high
    suc_rate = 0.2
    _test_cut = "2025-03-01"
    _clean_test = cleand_df[cleand_df["HS_START_DATE"] >= test_cut]
    sign_diaglist = diags[(~((diags['ci_low'] <= 0) & (diags['ci_high'] >= 0))) & (diags['ate_days'] > 0) ]["profile"].tolist()
    sign_diagdf = _clean_test[_clean_test["group"].isin(sign_diaglist)]
    sign_diagdf = sign_diagdf.merge(diags[["profile", "ate_days"]], left_on="group", right_on="profile", how="left").drop(columns=["profile"])


    ## calcualate total bed-days and excess bed-days caused by the significant diagnosis groups
    total_bed_days = cleand_df["spell_los"].sum()
    excess_bed_days = sign_diagdf["ate_days"].sum()*suc_rate
    ## calculate the percentage of excess bed-days caused by the significant diagnosis groups
    excess_percentage = (excess_bed_days / total_bed_days) * 100
    excess_percentage
    return excess_bed_days, sign_diagdf


@app.cell
def _(excess_bed_days, sign_diagdf):
    excess_bed_days/sign_diagdf["spell_los"].quantile(0.5)
    return


@app.cell
def _(sign_diagdf):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))

    sns.kdeplot(
        sign_diagdf["spell_los"].dropna(),
        fill=True,
        color="skyblue",
        alpha=0.5,
    )

    plt.xlabel("Length of Stay (days)")
    plt.ylabel("Density")
    plt.title("Distribution of Length of Stay")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(cleand_df, diags, group_select, metric_select, mo, success_rate):
    # Counterfactual calculations
    test_cut = "2025-03-01"
    clean_test = cleand_df[cleand_df["HS_START_DATE"] >= test_cut]
    total_admits = clean_test.shape[0]

    # List of all groups with positive ATE from diags
    positive_groups = diags[diags["ate_days"] > 0]["profile"].tolist()

    selected_group = group_select.value

    if selected_group == "combined":
        active_diag = clean_test[clean_test["group"].isin(positive_groups)]
        group_label = "Combined Positive Groups"

        # Calculate excess for each subgroup based on its specific baseline
        excess_los = 0
        for profile in positive_groups:
            sub_group = active_diag[active_diag["group"] == profile]
            len_sub = sub_group.shape[0]
            if not sub_group.empty:
                baseline = diags.loc[diags["profile"] == profile, metric_select.value].values[0]
                excess_los += len_sub * baseline
    else:
        active_diag = clean_test[clean_test["group"] == selected_group]
        len_act = active_diag.shape[0]
        group_label = "Uncoded Group" if selected_group == "unknown" else selected_group
        baseline_los = diags.loc[diags["profile"] == selected_group, metric_select.value].values[0]
        excess_los = len_act * baseline_los

    num_admits = active_diag.shape[0]

    # Calculate admission ratio
    admits_pct = (num_admits / total_admits * 100.0) if total_admits > 0 else 0.0

    total_group_los = active_diag["spell_los"].sum()

    # Apply the success rate
    rate = success_rate.value / 100.0
    savings_bed_days = excess_los * rate

    # Calculate percentages
    pct_saved_group = (savings_bed_days / total_group_los * 100.0) if total_group_los > 0 else 0.0

    total_test_cohort_los = clean_test["spell_los"].sum()
    pct_saved_total = (savings_bed_days / total_test_cohort_los * 100.0) if total_test_cohort_los > 0 else 0.0

    # Generate styled cards
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
            "background-color": "#f5f3ff",
            "color": "#4338ca",
            "flex": "1",
        }
    )

    # Assemble dashboard layout
    header_label = "Combined Positive Groups" if selected_group == "combined" else f"Group '{selected_group}'"
    dashboard_header = mo.md(f"#### 🔍 Counterfactual Scenario Results for **{header_label}**")

    dashboard_row = mo.hstack(
        [card_total_group, card_savings, card_group_pct, card_total], 
        justify="space-between", 
        align="stretch",
        gap=1.0
    )

    mo.vstack([dashboard_header, card_admits, dashboard_row])
    return (test_cut,)


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


@app.cell
def _():
    # mo.accordion(
    #     {
    #         "🔹 Fragment 1: Problem Statement": mo.md(
    #             "Length of stay varies significantly across hospital departments."
    #         ),
    #         "🔹 Fragment 2: Proposed Solution": mo.md(
    #             "Apply Healthcare Resource Groups (HRGs) for dimensionality reduction."
    #         ),
    #         "🔹 Fragment 3: Expected Outcome": mo.md(
    #             "Better causal estimates and attributable bed-day reductions."
    #         ),
    #     }
    # )
    return


@app.cell
def _():
    # import plotly.express as px

    # # Sample dataset
    # df = px.data.gapminder()

    # # Plotly animation frame
    # fig = px.scatter(
    #     df,
    #     x="gdpPercap",
    #     y="lifeExp",
    #     animation_frame="year",
    #     animation_group="country",
    #     size="pop",
    #     color="continent",
    #     hover_name="country",
    #     log_x=True,
    #     size_max=55,
    #     range_x=[100, 100000],
    #     range_y=[25, 90],
    #     title="Life Expectancy vs GDP Over Time"
    # )

    # fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500

    # mo.ui.plotly(fig)
    return


@app.cell
def _():
    # import matplotlib.pyplot as plt
    # def generate_plot():
    #     # Load data from local poster tables
    #     caus_df = diags.copy()
    #     diag_order = order_df.copy()

    #     df = pd.merge(caus_df, diag_order, on='profile', how='left')

    #     # Clinically intuitive mapping grounded in ICD_HRG_mapping.csv descriptions
    #     clinical_names = {
    #         'HE11': 'Hip & Femur Fractures (HE11)',
    #         'HE': 'Musculoskeletal Trauma (HE)',
    #         'DZ11': 'Severe Pneumonia & Lung Infection (DZ11)',
    #         'EB10': 'Heart Attack & Cardiac Shock (EB10)',
    #         'WJ': 'Sepsis & Severe Systemic Infection (WJ)',
    #         'HC': 'Spinal & Joint Disorders (HC)',
    #         'AA': 'Stroke & Acute Neurological (AA)',
    #         'unknown': '★ Diagnostic Ambiguity (Uncoded Stay)',
    #         'HD': 'Bone & Soft-Tissue Musculoskeletal (HD)',
    #         'FD11': 'Digestive & Oesophageal Cancers (FD11)',
    #         'AA26': 'Cerebrovascular & Nerve Disorders (AA26)',
    #         'LA04': 'Severe Kidney Infection / Pyelonephritis (LA04)',
    #         'YQ50': 'Arterial Thrombosis & Vascular Disease (YQ50)',
    #         'EB14': 'Cardiomyopathy & Heart Disease (EB14)',
    #         'EB': 'Valvular & Inflammatory Heart Disease (EB)',
    #         'DZ': 'Chronic Respiratory Conditions (DZ)',
    #         'KC': 'Metabolic & Electrolyte Disorders (KC)',
    #         'LA': 'Kidney & Urinary Conditions (LA)',
    #         'EB07': 'Cardiac Arrhythmias & Conduction (EB07)',
    #         'JD07': 'Skin & Subcutaneous Conditions (JD07)',
    #         'GC17': 'Liver & Pancreatic Conditions (GC17)',
    #         'WH': 'General Clinical Surveillance (WH)',
    #         'LB38': 'Urinary Symptoms & Haematuria (LB38)',
    #         'MB': 'Gynaecological Conditions (MB)',
    #         'CB': 'Oral & Upper Respiratory (CB)',
    #         'FD': 'Gastrointestinal Infections (FD)',
    #         'LB': 'Bladder & Kidney Disorders (LB)',
    #         'FD10': 'General Intestinal Disorders (FD10)',
    #         'FD02': 'Inflammatory Bowel Disease (FD02)',
    #         'DZ65': 'Chronic Bronchitis & COPD (DZ65)',
    #         'FD03': 'Peptic Ulcer & GI Bleeding (FD03)',
    #         'SA04': 'Iron Deficiency Anaemia (SA04)',
    #         'DZ19': 'Wheezing & Bronchospasm (DZ19)',
    #         'EB12': 'Non-cardiac Chest Pain (EB12)',
    #         'FD05': 'Non-specific Abdominal Pain (FD05)',
    #         'WH52': 'Post-Surgical Cancer Surveillance (WH52)'
    #     }

    #     df['clinical_name'] = df['profile'].map(clinical_names).fillna(df['profile'])

    #     # Sort by ate_days ascending so largest delays appear at top of Y-axis
    #     df = df.sort_values(by='ate_days', ascending=True).reset_index(drop=True)

    #     # Color definitions
    #     colors = []
    #     for idx, row in df.iterrows():
    #         if row['profile'] == 'unknown':
    #             colors.append('#dc2626') # Vivid crimson for unknown
    #         elif row['ci_low'] > 0:
    #             colors.append('#e05638') # Coral red for statistically significant positive
    #         elif row['ci_high'] < 0:
    #             colors.append('#0d9488') # Teal for statistically significant negative
    #         else:
    #             colors.append('#94a3b8') # Slate gray for not significant

    #     # Bubble sizing proportional to admission volume in NHS cohort
    #     prop_pct = df['proportion'] * 100
    #     sizes = 75 + (prop_pct / prop_pct.max()) * 520

    #     # Layout dimensions optimized for A0 poster card width
    #     fig, ax = plt.subplots(figsize=(11.5, 14.5), dpi=300)
    #     fig.patch.set_facecolor('#ffffff')
    #     ax.set_facecolor('#f8fafc')

    #     y_pos = np.arange(len(df))

    #     # Subtle background gridlines
    #     for y in y_pos:
    #         ax.axhline(y, color='#e2e8f0', linestyle='-', linewidth=0.9, zorder=1)

    #     # Solid vertical zero line
    #     ax.axvline(0, color='#475569', linestyle='-', linewidth=2.2, zorder=2)

    #     # 95% Confidence Interval error bars
    #     xerr_low = df['ate_days'] - df['ci_low']
    #     xerr_high = df['ci_high'] - df['ate_days']

    #     ax.errorbar(
    #         df['ate_days'], y_pos,
    #         xerr=[xerr_low, xerr_high],
    #         fmt='none',
    #         ecolor='#334155',
    #         elinewidth=2.0,
    #         capsize=4.5,
    #         capthick=1.8,
    #         zorder=3
    #     )

    #     # Scatter points (bubbles)
    #     ax.scatter(
    #         df['ate_days'], y_pos,
    #         s=sizes,
    #         c=colors,
    #         edgecolors='#0f172a',
    #         linewidth=1.4,
    #         alpha=0.94,
    #         zorder=4
    #     )

    #     # Soft red highlight band behind the Unknown/Uncoded diagnosis row
    #     unknown_idx = df[df['profile'] == 'unknown'].index[0]
    #     ax.axhspan(unknown_idx - 0.48, unknown_idx + 0.48, color='#fee2e2', alpha=0.65, zorder=0)

    #     # Y-axis labels with increased font size for poster readability
    #     ax.set_yticks(y_pos)
    #     labels = df['clinical_name'].tolist()
    #     ax.set_yticklabels(labels, fontsize=12.2, fontweight='medium', color='#0f172a')

    #     # Distinctive styling for Unknown Diagnosis label
    #     ax.get_yticklabels()[unknown_idx].set_fontweight('bold')
    #     ax.get_yticklabels()[unknown_idx].set_color('#b91c1c')
    #     ax.get_yticklabels()[unknown_idx].set_fontsize(13.2)

    #     # X-axis configuration
    #     ax.set_xlim(-8.5, 16.8)
    #     ax.set_xlabel('Causal Effect on Hospital Stay (Average Treatment Effect, Days)', fontsize=14.0, fontweight='bold', color='#0a2540', labelpad=12)
    #     ax.tick_params(axis='x', labelsize=12.5, colors='#1e293b')

    #     # Direction indicators
    #     ax.annotate('← Shorter Hospital Stay', xy=(-4.5, len(df)-0.1), xytext=(-4.5, len(df)+0.75),
    #                 ha='center', fontsize=12.5, fontweight='bold', color='#0d9488',
    #                 annotation_clip=False)
    #     ax.annotate('Extends Hospital Stay (Days) →', xy=(8.0, len(df)-0.1), xytext=(8.0, len(df)+0.75),
    #                 ha='center', fontsize=12.5, fontweight='bold', color='#e05638',
    #                 annotation_clip=False)

    #     # Header Title
    #     ax.set_title('Causal Effect of Diagnoses on Inpatient Length of Stay (LoS)', fontsize=16.0, fontweight='bold', color='#0a2540', pad=28)
    #     ax.text(0.5, 1.018, 'Bubble size = Volume (% of NHS admissions) | Error bars = 95% Confidence Interval (DML-IRM)',
    #             transform=ax.transAxes, ha='center', fontsize=11.5, color='#475569', style='italic')

    #     # Informative callout for Diagnostic Ambiguity
    #     ax.annotate('Diagnostic Ambiguity:\n+2.53 days causal delay\n(28.0% of all admissions)',
    #                 xy=(df.loc[unknown_idx, 'ate_days'], unknown_idx),
    #                 xytext=(df.loc[unknown_idx, 'ate_days'] + 4.2, unknown_idx - 3.8),
    #                 fontsize=11.0, fontweight='bold', color='#b91c1c',
    #                 arrowprops=dict(arrowstyle='->', color='#b91c1c', lw=1.8, connectionstyle='arc3,rad=-0.2'),
    #                 bbox=dict(boxstyle='round,pad=0.55', facecolor='#ffffff', edgecolor='#ef4444', lw=1.5),
    #                 zorder=5)

    #     # Informative callout for HE11
    #     he11_idx = df[df['profile'] == 'HE11'].index[0]
    #     ax.annotate('Hip/Femur Fractures:\n+14.09 days delay',
    #                 xy=(df.loc[he11_idx, 'ate_days'], he11_idx),
    #                 xytext=(df.loc[he11_idx, 'ate_days'] - 6.2, he11_idx - 1.8),
    #                 fontsize=10.5, fontweight='bold', color='#e05638',
    #                 arrowprops=dict(arrowstyle='->', color='#e05638', lw=1.5, connectionstyle='arc3,rad=0.15'),
    #                 bbox=dict(boxstyle='round,pad=0.45', facecolor='#ffffff', edgecolor='#e05638', lw=1.2),
    #                 zorder=5)

    #     # Frame lines
    #     for spine in ['top', 'right']:
    #         ax.spines[spine].set_visible(False)
    #     ax.spines['left'].set_color('#94a3b8')
    #     ax.spines['left'].set_linewidth(1.2)
    #     ax.spines['bottom'].set_color('#94a3b8')
    #     ax.spines['bottom'].set_linewidth(1.2)

    #     plt.tight_layout()
    #     plt.show()
    #     # fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
    #     # fig.savefig(output_path_pdf, bbox_inches='tight')
    #     # plt.close(fig)
    #     # print(f'Successfully generated: {output_path_png}')
    # generate_plot()
    return


if __name__ == "__main__":
    app.run()
