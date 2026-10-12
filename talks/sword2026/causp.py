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

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    layout_file="layouts/causp.slides.json",
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
    # scale with the slide width (Quarto reveal does this via its 1920x1080 canvas)
    _scale = """<style>
      .ts-wrap { container-type: inline-size; width: 100%; overflow: hidden; }
      .ts-wrap .title-slide-custom { font-size: 2.1cqw; }
      .ts-wrap .title-slide-custom .logo-bar { margin-bottom: 4cqw; }
      .ts-wrap .title-slide-custom .logo-bar img { height: 6cqw !important; width: auto !important; max-width: none; }
    </style>"""
    mo.Html(f'{_scale}<div class="ts-wrap">{_html}</div>')
    return (re,)


@app.cell(hide_code=True)
def _(mo):
    _items = [
        "The Challenge: Why Patients Arrive and Why Some Stay Longer?",
        "Why is It So Important?",
        # "Grouping diagnosis with Healthcare Resource Groups (HRGs)",
        "Methodology",
        "Key Results",
        "Next"
    ]

    def outline(active=None):
        def _color(i):
            if active is None:
                return "#999999"
            return "#20808D" if i == active else "#999999"

        _li = "".join(
            f'<li style="color:{_color(i)};font-size:1em;line-height:1.9">{t}</li>' # points + bullet size
            for i, t in enumerate(_items)
        )
        return mo.Html(
            '<style>.ol-list li::marker { font-size: 1.2em; }</style>'   # bullet dot only
            '<div style="container-type:inline-size;width:100%">'
            '<div style="font-size:3cqw;padding:2cqw 4cqw">' # master size (everything)
            '<h2 style="font-size:1.6em;font-weight:700;color:#091717;margin:0 0 0.6em 0">Outline</h2>'  # title only
            f'<ul class="ol-list" style="font-weight:700;margin:0;'
            f'padding-left:1.2em;list-style:disc">{_li}</ul>'
            '</div></div>'
        )

    return (outline,)


@app.cell(hide_code=True)
def _(outline):
    outline()
    return


@app.cell(hide_code=True)
def _(outline):
    outline(0)
    return


@app.cell(hide_code=True)
def _(mo, re):
    def fragment_md(markdown):
        markdown = re.sub(
            r"(?m)^- (.+)$",
            lambda match: f'<span class="fragment">• {match.group(1)}</span>',
            markdown,
        )
        return mo.md(markdown)

    _image = mo.Html(
        '<img '
        'src="https://raw.githubusercontent.com/mustafaslanCoto/mustafaslanCoto.github.io/main/talks/images/hospital_bed.jpg" '
        'alt="Hospital bed" '
        'style="display:block;width:100%;height:600px;object-position:center;border-radius:8px">'
        # 'style="display:block;width:100%;height:auto;max-height:65vh;border-radius:8px">'
    )
    _text = fragment_md("""

    ## Problem

    - **Hidden Clinical Drivers:** The true, isolated impact of specific diagnoses on Length of Stay (LoS) remains unknown

    - **Tangled Hospital Data:** Hospitals need to understand why some patients stay longer than expected, but raw records mix together clinical diagnoses, patient age, chronic illnesses, and ward congestion.

    - **Correlations vs. Causes:** Standard machine learning predicts correlations, not causes: it cannot tell whether a diagnosis actively causes a delay or simply reflects why a patient arrived at the hospital in the first place.

    - **Reclaiming Bed Capacity:** Identifying clear, diagnosis-specific discharge protocols early in admission is essential to taking proactive action, preventing avoidable delays, and freeing up acute hospital beds.

    """)

    # _text = mo.md("""
    # ## Why does it matter?

    # <ul>
    #   <li class="fragment"><strong>Hidden Clinical Drivers:</strong> The true, isolated impact of specific diagnoses on Length of Stay (LoS) remains unknown.</li>
    #   <li class="fragment"><strong>Tangled Hospital Data:</strong> Hospitals need to understand why some patients stay longer than expected, but raw records mix together clinical diagnoses, patient age, chronic illnesses, and ward congestion.</li>
    #   <li class="fragment"><strong>Correlations vs. Causes:</strong> Standard machine learning predicts correlations, not causes: it cannot tell whether a diagnosis actively causes a delay or simply reflects why a patient arrived at the hospital in the first place.</li>
    #   <li class="fragment"><strong>Reclaiming Bed Capacity:</strong> Identifying clear, diagnosis-specific discharge protocols early in admission is essential to taking proactive action, preventing avoidable delays, and freeing up acute hospital beds.</li>
    # </ul>
    # """)
    mo.hstack([_image, _text], widths=[0.45, 0.55], align="center", gap=1)
    return (fragment_md,)


@app.cell
def _(outline):
    outline(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Advancing NHS Wales *Six Goals* for Urgent Care

    - **Goal 4 (Rapid Assessment & Cohorting):** Frontloading diagnostic clarification during the critical first 48 hours of admission.
    - **Goal 5 (Optimal Hospital Care & Timely Discharge):** Activating diagnosis-tailored discharge readiness protocols before discharge.
    - **Goal 6 (Home First / Step-Down):** Early social care and pharmacy coordination, avoiding costly hospital deconditioning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tailoring Discharge to Clinical Need

    - **Diagnosis-Tailored Discharge Bundles:** Shifting ward practice from one-size-fits-all planning to targeted, multidisciplinary pathways aligned with the patient's primary diagnosis.
    - **First 48-Hour Protocol:** Frontloading diagnostic clarification early in the admission window to resolve clinical ambiguity before avoidable discharge delays build up.
    """)
    return


@app.cell
def _(outline):
    outline(2)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## Assumed Causal Structure for Diagnosis and Length of Stay"),
        mo.mermaid(
            """
            flowchart LR
                X["Observed Confounders <br/> (Control Variables) <br/><i style='font-family:serif;font-size:4em;'>X</i>"]
                D["Diagnosis<br/><i style='font-family:serif;font-size:4em;'>D</i>"]
                Y["Length of Stay<br/><i style='font-family:serif;font-size:4em;'>LoS</i>"]

                X --> D
                X --> Y
                D --> Y

                classDef covariate fill:#dcefee,stroke:#20808D,stroke-width:2px,color:#091717;
                classDef diagnosis fill:#b7d5da,stroke:#20808D,stroke-width:2px,color:#091717;

                class X,Y covariate;
                class D diagnosis;
            """,
            theme="base",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <!-- ## Double Machine Learning (DML): A Robust Approach to Causal Inference

    - **Treatment:** For each diagnosis, define its presence as $(D=1)$ and absence as $(D=0)$. The outcome $Y$ is Length of Stay (LoS).

    - **Adjust for confounding:** Use observed patient and admission characteristics $(X)$ to account for factors associated with both diagnosis and LoS.

    - **Learn with machine learning:** Estimate the expected LoS, $(g(D,X)=\mathbb{E}[Y\mid D,X])$, and the diagnosis propensity, $(m(X)=\Pr(D=1\mid X))$.

    - **Estimate the effect:** Combine these models in a doubly robust, orthogonal score, then average the predicted contrast between diagnosis present and absent across patients.

    - **Repeat for each diagnosis** to obtain diagnosis-specific average effects on LoS.

    **Interpretation:** These are adjusted population-average effects, not individual patient-level causal effects. A causal interpretation relies on adequate confounder adjustment and overlap between diagnosis groups. -->
    """)
    return


@app.cell
def _():
    import io as _io
    from urllib.request import urlopen as _urlopen

    import pandas as pd
    import numpy as np
    # import seaborn as sns

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
    order_df = pd.read_csv(fetch(f"{data_url}/diagnosis_order.csv"))
    diags = pd.read_csv(fetch(f"{data_url}/caus_hrg_lgb.csv"))
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

    confounders = pd.read_csv(fetch(f"{data_url}/confounders.csv"))
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
    return


@app.cell
def _():
    # mo.ui.table(diags.round(2), pagination=True, page_size=15)
    # cleand_df[cleand_df["code"]!="$$X"]["code"].value_counts(normalize=True).cumsum()
    return


@app.cell
def _(confounders, mo):
    _names = confounders["confounder"].dropna().astype(str).tolist()
    _mid = (len(_names) + 1) // 2

    _left = mo.md("\n".join(f"- `{name}`" for name in _names[:_mid]))
    _right = mo.md("\n".join(f"- `{name}`" for name in _names[_mid:]))

    mo.vstack([
        mo.md(f"## Confounders ({len(_names)})\n\nCovariates included in $X$"),
        mo.hstack([_left, _right], widths=[0.5, 0.5], align="start", gap=1),
    ])
    return


app._unparsable_cell(
    r"""
    ## Double Machine Learning (DML): A Robust Approach to Causal Inference

    For each diagnosis, let $D=1$ indicate its presence and $D=0$ its absence. Let $LoS$ denote Length of Stay, and $X$ the observed confounders.

    The average predictive effect, $\theta_0$, compares the expected $LoS$ with the diagnosis present versus absent, averaged over patients' observed characteristics:

    $$
    \theta_0
    =
    \mathbb{E}\!\left[g_0(1,X)-g_0(0,X)\right].
    $$

    The outcome and diagnosis are modelled using the following regression equations:

    $$
    \begin{aligned}
    LoS &= g_0(D,X) + \varepsilon,
    &\quad \mathbb{E}[\varepsilon \mid X,D] &= 0, \\
    D &= m_0(X) + \widetilde{D},
    &\quad \mathbb{E}[\widetilde{D} \mid X] &= 0.
    \end{aligned}
    $$

    Here, $g_0(D,X)$ is the regression function for $LoS$, while $m_0(X)$ is the classification function for the diagnosis given the confounders.

    Building on these two regression models, we construct an efficient estimator of the average predictive effect using the following orthogonal score:

    $$
    \theta_0 = \mathbb{E}\!\left[\varphi_0(W)\right],
    \qquad W=(LoS,D,X),
    $$

    where

    $$
    \varphi_0(W)
    =
    g_0(1,X)-g_0(0,X)
    +
    \bigl(LoS-g_0(D,X)\bigr)H_0,
    $$

    and

    $$
    H_0
    =
    \frac{\mathbf{1}(D=1)}{m_0(X)}
    -
    \frac{\mathbf{1}(D=0)}{1-m_0(X)}.
    $$

    Here, $\mathbf{1}(\cdot)$ is an indicator function, and $H_0$ is the Horvitz-Thompson transformation.
    """,
    column=None, disabled=False, hide_code=True, expand_output=False, name="_"
)


@app.cell
def _(outline):
    outline(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Causal Effect Sizes of Diagnosis on LoS
    """)
    return


@app.cell
def _(diags, mo, np, pd):
    import altair as alt

    clinical_names = {
        'HE11': 'Hip & Femur Fractures',
        'HE': 'Musculoskeletal Trauma',
        'DZ11': 'Severe Pneumonia & Lung Infection',
        'EB10': 'Heart Attack & Cardiac Shock',
        'WJ': 'Sepsis & Severe Systemic Infection',
        'HC': 'Spinal & Joint Disorders',
        'AA': 'Stroke & Acute Neurological',
        'unknown': '★ Diagnostic Ambiguity (Uncoded Stay)',
        'HD': 'Bone & Soft-Tissue Musculoskeletal',
        'FD11': 'Digestive & Oesophageal Cancers',
        'AA26': 'Cerebrovascular & Nerve Disorders',
        'LA04': 'Severe Kidney Infection / Pyelonephritis',
        'YQ50': 'Arterial Thrombosis & Vascular Disease',
        'EB14': 'Cardiomyopathy & Heart Disease',
        'EB': 'Valvular & Inflammatory Heart Disease',
        'DZ': 'Chronic Respiratory Conditions',
        'KC': 'Metabolic & Electrolyte Disorders',
        'LA': 'Kidney & Urinary Conditions',
        'EB07': 'Cardiac Arrhythmias & Conduction',
        'JD07': 'Skin & Subcutaneous Conditions',
        'GC17': 'Liver & Pancreatic Conditions',
        'WH': 'General Clinical Surveillance',
        'LB38': 'Urinary Symptoms & Haematuria',
        'MB': 'Gynaecological Conditions',
        'CB': 'Oral & Upper Respiratory',
        'FD': 'Gastrointestinal Infections',
        'LB': 'Bladder & Kidney Disorders',
        'FD10': 'General Intestinal Disorders',
        'FD02': 'Inflammatory Bowel Disease',
        'DZ65': 'Chronic Bronchitis & COPD',
        'FD03': 'Peptic Ulcer & GI Bleeding',
        'SA04': 'Iron Deficiency Anaemia',
        'DZ19': 'Wheezing & Bronchospasm',
        'EB12': 'Non-cardiac Chest Pain',
        'FD05': 'Non-specific Abdominal Pain',
        'WH52': 'Post-Surgical Cancer Surveillance'
    }

    def create_clinical_forest_plot(df: pd.DataFrame, top_n: int = 36) -> alt.Chart | mo.Html:
        if df.empty:
            return mo.md("*No effect size data available.*")

        required_cols = ["profile", "ate_days", "ci_low", "ci_high", "dominance"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return mo.md(f"*Missing required columns: `{list(missing_cols)}`*")

        # Clean, map names, identify intervals crossing zero, and sort
        clean_df = (
            df.dropna(subset=required_cols)
            .assign(
                abs_ate=lambda x: x["ate_days"].abs(),
                dominance_pct=lambda x: x["dominance"] * 100,
                clinical_name=lambda x: x["profile"].astype(str).map(clinical_names).fillna(x["profile"].astype(str)),
                display_label=lambda x: x.apply(
                    lambda r: f"{r['clinical_name']} ({r['profile']})" if r["clinical_name"] != str(r["profile"]) else str(r["profile"]),
                    axis=1,
                ),
                # Check whether 0 falls inside [ci_low, ci_high]
                crosses_zero=lambda x: (x["ci_low"] <= 0) & (x["ci_high"] >= 0),
                effect_dir=lambda x: np.select(
                    [
                        x["crosses_zero"],
                        x["ate_days"] > 0,
                    ],
                    [
                        "Not Significant (CI spans 0)",
                        "Increases LoS (+ Days)",
                    ],
                    default="Decreases LoS (- Days)",
                ),
            )
            .sort_values(by="abs_ate", ascending=False)
            .head(top_n)
            .sort_values(by="ate_days", ascending=False)  # Highest positive at top
            .reset_index(drop=True)
        )

        if clean_df.empty:
            return mo.md("*No valid numeric rows found after cleaning.*")

        y_order = clean_df["display_label"].tolist()

        # Y-encoding with horizontal grid and colored Ambiguity label
        y_encoding = alt.Y(
            "display_label:N",
            title=None,
            sort=y_order,
            axis=alt.Axis(
                labelFontSize=12,
                labelLimit=380,
                ticks=False,
                domainColor="#CBD5E1",
                grid=True,                              # Horizontal grid enabled
                gridColor="rgba(203, 213, 225, 0.45)",  # Soft neutral divider
                gridDash=[2, 2],                        # Subtle dashed style
                gridWidth=1,
                labelColor=alt.expr(                    # Red label for Diagnostic Ambiguity
                    "indexof(datum.value, 'Diagnostic Ambiguity') >= 0 ? '#B91C1C' : '#334155'"
                ),
                labelFontWeight=alt.expr(               # Bold text for Diagnostic Ambiguity
                    "indexof(datum.value, 'Diagnostic Ambiguity') >= 0 ? 'bold' : 'normal'"
                ),
            ),
        )

        # 3-color palette: Red (increases), Teal (decreases), Grey (spans zero)
        color_scale = alt.Scale(
            domain=[
                "Increases LoS (+ Days)",
                "Decreases LoS (- Days)",
                "Not Significant (CI spans 0)",
            ],
            range=["#E53E3E", "#319795", "#94A3B8"],
        )

        base = alt.Chart(clean_df)

        # 0. Horizontal shaded band behind the Diagnostic Ambiguity row
        ambiguity_band = base.transform_filter(
            alt.datum.profile == "unknown"
        ).mark_rule(
            color="#FEE2E2",
            strokeWidth=24,
            opacity=0.75,
        ).encode(
            y=y_encoding
        )

        # 1. Zero dashed baseline rule
        zero_line = alt.Chart().mark_rule(
            color="#64748B",
            strokeWidth=1.5,
            strokeDash=[5, 5],
        ).encode(x=alt.datum(0))

        # 2. 95% Confidence Interval error bars
        error_bars = base.mark_rule(
            strokeWidth=2.2,
        ).encode(
            x=alt.X("ci_low:Q"),
            x2="ci_high:Q",
            y=y_encoding,
            color=alt.Color("effect_dir:N", scale=color_scale, legend=None),
        )

        # 3. Effect size circles scaled by Dominance %
        points = base.mark_circle(
            stroke="#1E293B",
            strokeWidth=1.2,
            opacity=0.95,
        ).encode(
            x=alt.X(
                "ate_days:Q",
                title="Causal Effect Sizes of Diagnosis on Hospital Length of Stay in Days (95% CI)",
                axis=alt.Axis(
                    orient="top",
                    grid=True,                          # Vertical grid
                    gridColor="rgba(160, 174, 192, 0.25)",
                    titleFontSize=13,
                    labelFontSize=11,
                    titlePadding=12,
                ),
            ),
            y=y_encoding,
            size=alt.Size(
                "dominance_pct:Q",
                title="Dominance (%)",
                scale=alt.Scale(range=[80, 520]),
                legend=alt.Legend(titleFontSize=12, labelFontSize=11, orient="right"),
            ),
            color=alt.Color(
                "effect_dir:N",
                title="Causal Significance",
                scale=color_scale,
                legend=alt.Legend(titleFontSize=12, labelFontSize=11, orient="right"),
            ),
            tooltip=[
                alt.Tooltip("clinical_name:N", title="Clinical Group"),
                alt.Tooltip("profile:N", title="HRG Code"),
                alt.Tooltip("ate_days:Q", title="ATE (Days)", format="+.2f"),
                alt.Tooltip("ci_low:Q", title="95% CI Lower", format=".2f"),
                alt.Tooltip("ci_high:Q", title="95% CI Upper", format=".2f"),
                alt.Tooltip("dominance_pct:Q", title="Dominance (%)", format=".2f"),
                alt.Tooltip("effect_dir:N", title="Significance"),
            ],
        )

        # 4. Inline text note right beside the Diagnostic Ambiguity whisker
        ambiguity_note = base.transform_filter(
            alt.datum.profile == "unknown"
        ).mark_text(
            align="left",
            baseline="middle",
            dx=16,
            fontSize=11,
            fontWeight="bold",
            color="#B91C1C",
            text="← +2.53 days delay (28.0% of all admissions)",
        ).encode(
            x="ci_high:Q",
            y=y_encoding,
        )

        return (
            (ambiguity_band + zero_line + error_bars + points + ambiguity_note)
            .properties(
                width=880,
                height=max(360, len(clean_df) * 22),
                background="transparent",
            )
            .configure_view(stroke=None)
        )

    create_clinical_forest_plot(diags, top_n=36)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attributable bed-days from the uncoded group and codes increasing LoS
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

    # Clinical name mapping
    _clin_map = globals().get("clinical_names", {})

    # Use concise labels so badges never crowd the row
    my_options = {}
    for _, row in pos_diags.iterrows():
        prof = str(row["profile"])
        icon = "📁" if prof == "unknown" else "🏥"

        if prof == "unknown":
            short_label = "📁 Uncoded"
        else:
            clin = _clin_map.get(prof, prof)
            short_label = (
                f"{icon} {prof} ({clin.split('&')[0].split('/')[0].strip()[:9]})"
            )

        my_options[short_label] = prof

    label_style = "font-size: 13px; font-weight: 600; color: #475569; display: block; margin-bottom: 4px;"

    default_key = next(
        (k for k, v in my_options.items() if v == "unknown"),
        list(my_options.keys())[0],
    )

    # 1. Multi-select (no inline label)
    group_select = mo.ui.multiselect(
        options=my_options,
        value=[default_key],
        label="",
    )

    # 2. Causal Estimator Dropdown (no inline label)
    metric_options = {
        "📊 Point Estimate (ATE)": "ate_days",
        "📉 Conservative (CI Low)": "ci_low",
        "📈 Aggressive (CI High)": "ci_high",
    }

    metric_select = mo.ui.dropdown(
        options=metric_options,
        value="📊 Point Estimate (ATE)",
        label="",
    )

    # 3. Success Rate Slider (no inline label)
    success_rate = mo.ui.slider(
        start=0,
        stop=100,
        step=5,
        value=100,
        label="",
    )

    # Place each control in an isolated column with its label on top
    col1 = mo.vstack(
        [
            mo.md(f"<span style='{label_style}'>Select Diagnoses:</span>"),
            group_select,
        ],
        gap=0.3,
    ).style({"flex": "1.3", "min-width": "0", "overflow": "hidden"})

    col2 = mo.vstack(
        [mo.md(f"<span style='{label_style}'>Estimator:</span>"), metric_select],
        gap=0.3,
    ).style({"flex": "0.9", "min-width": "0", "overflow": "hidden"})

    col3 = mo.vstack(
        [
            mo.md(f"<span style='{label_style}'>Resolution Rate (%):</span>"),
            success_rate,
        ],
        gap=0.3,
    ).style({"flex": "1.0", "min-width": "0"})

    _custom_css = mo.Html("""
    <style>
        /* Prevent multiselect from breaking its column boundary */
        marimo-multiselect {
            width: 100% !important;
            display: block !important;
        }
        marimo-multiselect div[role="combobox"],
        marimo-multiselect button {
            max-height: 38px !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            white-space: nowrap !important;
            flex-wrap: nowrap !important;
        }
        select { 
            width: 100% !important;
            font-size: 13px !important; 
            padding: 5px 8px !important; 
        }
        input[type=range] { 
            cursor: pointer;
            width: 100% !important;
        }
    </style>
    """)

    controls_display = mo.hstack(
        [col1, col2, col3], justify="space-between", align="start", gap=3
    ).style(
        style={
            "padding": "12px 18px",
            "background-color": "#f8fafc",
            "border-radius": "10px",
            "border": "1px solid #e2e8f0",
            "width": "100%",
        }
    )

    mo.vstack([_custom_css, controls_display])
    return group_select, metric_select, success_rate


@app.cell
def _():
    # ## filter rows where there is no 0 between ci_low and ci_high
    # suc_rate = 0.2
    # _test_cut = "2025-03-01"
    # _clean_test = cleand_df[cleand_df["HS_START_DATE"] >= test_cut]
    # sign_diaglist = diags[(~((diags['ci_low'] <= 0) & (diags['ci_high'] >= 0))) & (diags['ate_days'] > 0) ]["profile"].tolist()
    # sign_diagdf = _clean_test[_clean_test["group"].isin(sign_diaglist)]
    # sign_diagdf = sign_diagdf.merge(diags[["profile", "ate_days"]], left_on="group", right_on="profile", how="left").drop(columns=["profile"])


    # ## calcualate total bed-days and excess bed-days caused by the significant diagnosis groups
    # total_bed_days = cleand_df["spell_los"].sum()
    # excess_bed_days = sign_diagdf["ate_days"].sum()*suc_rate
    # ## calculate the percentage of excess bed-days caused by the significant diagnosis groups
    # excess_percentage = (excess_bed_days / total_bed_days) * 100
    return


@app.cell
def _(cleand_df, diags, group_select, metric_select, mo, success_rate):
    # Counterfactual calculations
    test_cut = "2025-03-01"
    clean_test = cleand_df[cleand_df["HS_START_DATE"] >= test_cut]
    total_admits = clean_test.shape[0]

    positive_groups = diags[diags["ate_days"] > 0]["profile"].tolist()

    selected_groups = group_select.value
    if not isinstance(selected_groups, list):
        selected_groups = [selected_groups]

    if not selected_groups:
        selected_groups = positive_groups

    active_diag = clean_test[clean_test["group"].isin(selected_groups)]
    num_admits = active_diag.shape[0]

    _clin_map = globals().get("clinical_names", {})
    if len(selected_groups) == 1:
        _p0 = selected_groups[0]
        group_label = (
            "Uncoded Group"
            if _p0 == "unknown"
            else _clin_map.get(_p0, f"Group '{_p0}'")
        )
        banner_context = f"**{group_label}**"
    elif set(selected_groups) == set(positive_groups):
        group_label = "All Positive Groups"
        banner_context = f"across **All {len(selected_groups)} Positive Groups**"
    else:
        group_label = f"{len(selected_groups)} Selected Groups"
        banner_context = f"across **{len(selected_groups)} Selected Groups**"

    excess_los = 0.0
    _metric_col = metric_select.value

    for _p in selected_groups:
        _sub_df = active_diag[active_diag["group"] == _p]
        if not _sub_df.empty:
            _baseline_match = diags.loc[diags["profile"] == _p, _metric_col]
            if not _baseline_match.empty:
                _baseline_val = _baseline_match.values[0]
                excess_los += _sub_df.shape[0] * _baseline_val

    admits_pct = (num_admits / total_admits * 100.0) if total_admits > 0 else 0.0
    total_group_los = active_diag["spell_los"].sum()
    beds_occupied_year = total_group_los / 365.0

    rate = success_rate.value / 100.0
    savings_bed_days = excess_los * rate
    beds_freed_year = savings_bed_days / 365.0

    pct_saved_group = (
        (savings_bed_days / total_group_los * 100.0) if total_group_los > 0 else 0.0
    )
    total_test_cohort_los = clean_test["spell_los"].sum()
    pct_saved_total = (
        (savings_bed_days / total_test_cohort_los * 100.0)
        if total_test_cohort_los > 0
        else 0.0
    )

    _style_base = {
        "padding": "14px 12px",
        "border-radius": "10px",
        "text-align": "center",
        "flex": "1",
        "min-width": "190px",
    }

    # 1. Total Group Bed-Days & Beds Occupied
    card_total_group = mo.md(
        f"""
        ### 📁 **{total_group_los:,.0f}**
        **Total {group_label} Bed-Days**  
        <span style="font-size: 13px; font-weight: 600; color: #475569;">(~{beds_occupied_year:.0f} beds occupied / year)</span>
        """
    ).style(
        style={
            **_style_base,
            "border": "2px solid #8c8c8c",
            "background-color": "#fafafa",
            "color": "#1a1a1a",
        }
    )

    # 2. Bed-Days Saved & Beds Freed
    card_savings = mo.md(
        f"""
        ### 🛏️ **{savings_bed_days:,.0f}** <span style="font-size: 15px; font-weight: 600; opacity: 0.9;">({pct_saved_group:.0f}%)</span>
        **Bed-Days Saved**  
        <span style="font-size: 13px; font-weight: 600; color: #0369a1;">(~{beds_freed_year:.0f} beds freed / year)</span>
        """
    ).style(
        style={
            **_style_base,
            "border": "2px solid #007bc0",
            "background-color": "#f4f9fc",
            "color": "#004d7a",
        }
    )

    # 3. Whole Hospital Relief
    card_total = mo.md(
        f"""
        ### 📈 **{pct_saved_total:.0f}%**
        **Saved (Entire Hospital Capacity)**
        """
    ).style(
        style={
            **_style_base,
            "border": "2px solid #16a34a",
            "background-color": "#f0fdf4",
            "color": "#14532d",
        }
    )

    # Natural English phrasing for the header
    card_admits = mo.md(
        f"""
        ### 🏥 **Patients Affected: {num_admits:,} Hospital Stays** ({banner_context})
        **Accounts for {admits_pct:.0f}% of all admissions over the year (out of {total_admits:,} total)**
        """
    ).style(
        style={
            **_style_base,
            "border": "2px solid #6366f1",
            "background-color": "#f5f3ff",
            "color": "#4338ca",
            "flex": "1",
        }
    )

    dashboard_header = mo.md(
        f"#### 🔍 Counterfactual Scenario Results for **{group_label}**"
    )

    dashboard_row = mo.hstack(
        [card_total_group, card_savings, card_total],
        justify="space-between",
        align="stretch",
        gap=1.2,
    )

    mo.vstack([dashboard_header, card_admits, dashboard_row])
    return


@app.cell
def _(outline):
    outline(4)
    return


@app.cell
def _(fragment_md):
    fragment_md("""

    ## Next Steps

    - **Co-Designing Discharge Protocols & Teams**: Collaborating directly with CVUHB clinicians and managers to determine how specialized discharge teams should be structured and to establish tailored protocols targeting delay-inducing diagnoses identified by our causal model.

    - **Interactive Decision-Support Tool:** Designing and deploying an interactive web application for clinicians and operational managers to prioritize high-impact diagnoses, and measure the real-time bed capacity impact of tailored discharge protocols.

    - **Ward-Level Pilot Implementation:** Running a prospective pilot study within priority acute wards to test whether early diagnostic clarification and proactive multidisciplinary reviews translate into measured reductions in Length of Stay.

    - **Academic & Practitioner Dissemination:** Presenting findings at leading Operations Research conferences (e.g., EJOR, POMS, INFORMS) and NHS executive seminars to gather peer feedback and evaluate whether the framework generalizes across other NHS trusts and hospital settings.

    """
              )           
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


@app.cell(hide_code=True)
def _(mo):
    import sys as _sys

    _RAW = "https://raw.githubusercontent.com/mustafaslanCoto/mustafaslanCoto.github.io/main/talks/images/follow_us.png"
    if _sys.platform == "emscripten": ## Use raw URL for Emscripten environment, meaning we can't rely on local file paths
        _src = _RAW
    else:
        import base64 as _b64 # For encoding the local image file as a base64 string

        _p = mo.notebook_dir().parent / "images" / "follow_us.png"
        _src = "data:image/png;base64," + _b64.b64encode(_p.read_bytes()).decode()

    _text = mo.md("# Any questions or thoughts? 💬")
    # _text = mo.vstack([_thanks], align="center")
    _image = mo.Html(
        '<div style="container-type:inline-size;width:100%">'
        f'<img src="{_src}" style="display:block;margin:0 auto;max-height:55vh;max-width:100%">'
        '</div>'
    )

    mo.vstack([_text, _image], align = "center")
    return


if __name__ == "__main__":
    app.run()
