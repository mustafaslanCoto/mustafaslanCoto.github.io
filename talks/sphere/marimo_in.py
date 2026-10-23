import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", layout_file="layouts/marimo_in.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Specifications
    """)
    return


@app.cell
def _():
    import io
    import pickle
    from urllib.request import urlopen
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    import pandas as pd
    import numpy as np

    # Load precomputed model results directly from GitHub raw repository (or local fallback)
    _DATA_URL = "https://raw.githubusercontent.com/mustafaslanCoto/mustafaslanCoto.github.io/main/talks/sphere/ms_arr_model_results.pkl"
    try:
        with urlopen(_DATA_URL) as _resp:
            model_res = pickle.load(io.BytesIO(_resp.read()))
    except Exception:
        with open("talks/sphere/ms_arr_model_results.pkl", "rb") as _f:
            model_res = pickle.load(_f)

    hmm_df = model_res["hmm_df"]
    res = model_res.get("res", {})
    cv_dates = model_res.get("cv_dates", [])
    p_stay_regime0 = model_res.get("p_stay_regime0", [])
    p_stay_regime1 = model_res.get("p_stay_regime1", [])

    # Load dataset
    all_ead = "https://raw.githubusercontent.com/mustafaslanCoto/mustafaslanCoto.github.io/main/talks/sphere/ead_data.csv"
    ead = pd.read_csv(all_ead)
    ead["date"] = pd.to_datetime(ead["date"])
    ead.set_index("date", inplace=True)
    return (
        LineCollection,
        LinearSegmentedColormap,
        cv_dates,
        hmm_df,
        np,
        p_stay_regime0,
        p_stay_regime1,
        plt,
    )


@app.cell
def _(hmm_df, mo):
    date_picker = mo.ui.date_range(
        start=hmm_df.index.min().date(),
        stop=hmm_df.index.max().date(),
        value=(hmm_df.iloc[-180:].index.min().date(), hmm_df.index.max().date()),
        label="📅 **Select Date Range for Regime History:**"
    )
    date_picker
    return (date_picker,)


@app.cell
def _(
    LineCollection,
    LinearSegmentedColormap,
    date_picker,
    history_idx_slider,
    hmm_df,
    mo,
    np,
    plt,
):
    # Filter dataset based on selected date range
    _start_date, _stop_date = date_picker.value
    _filtered_df = hmm_df.loc[str(_start_date):str(_stop_date)].copy()

    if len(_filtered_df) < 2:
        _display = mo.md(f"⚠️ *Please select a range containing at least 2 days (currently {len(_filtered_df)} selected).*")
    else:
        _dates = _filtered_df.index
        _target = _filtered_df['ead'].values
        _prob_surge = _filtered_df['regime_2'].values
        _prob_base = _filtered_df['regime_1'].values

        # Active index from history slider
        _sel_i = min(history_idx_slider.value, len(_dates) - 1)
        _cur_date = _dates[_sel_i]
        _cur_target = _target[_sel_i]
        _cur_surge = _prob_surge[_sel_i]
        _cur_base = _prob_base[_sel_i]

        # Colormap from teal (0 = Normal) to amber (0.5) to coral/crimson (1 = Surge)
        _cmap = LinearSegmentedColormap.from_list('regime_cmap', ['#20808D', '#D97706', '#C93B2B'])

        _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(24, 9), sharex=True, gridspec_kw={'height_ratios': [1.15, 1]})
        _fig.patch.set_facecolor('#FBFAF4')
        _ax1.set_facecolor('#FBFAF4')
        _ax2.set_facecolor('#FBFAF4')

        # Subplot 1: Time Series colored by continuous probability
        _norm_dates = np.arange(len(_dates))
        _ax1.fill_between(_norm_dates, 0, _target, color='#20808D', alpha=0.06)

        _points = np.array([_norm_dates, _target]).T.reshape(-1, 1, 2)
        _segments = np.concatenate([_points[:-1], _points[1:]], axis=1)

        _lc = LineCollection(_segments, cmap=_cmap, linewidth=3.5, zorder=5)
        _lc.set_array(_prob_surge)
        _ax1.add_collection(_lc)
        _ax1.set_xlim(0, max(len(_dates) - 1, 1))
        _ax1.set_ylim(0, max(_target) * 1.22 if len(_target) > 0 and max(_target) > 0 else 1)

        _ax1.scatter(_norm_dates, _target, c=_prob_surge, cmap=_cmap, s=40, zorder=6, edgecolor='white', linewidth=0.8)

        # Vertical inspection line across Subplot 1
        _ax1.axvline(_sel_i, color='#1e293b', linestyle='-', linewidth=2.0, alpha=0.9, zorder=10)
        _ax1.scatter([_sel_i], [_cur_target], color='#0f172a', s=110, edgecolor='white', linewidth=2, zorder=11)

        # Tooltip callout on Subplot 1
        _ha1 = "right" if _sel_i / max(len(_dates) - 1, 1) > 0.68 else "left"
        _off1 = -1.5 if _ha1 == "right" else 1.5
        _tip1 = (
            f"Date: {_cur_date.strftime('%b %d, %Y')}\n"
            f"Avoidable Deaths: {_cur_target:.1f}\n"
            f"P(Surge): {_cur_surge:.3f}\n"
            f"P(Normal): {_cur_base:.3f}"
        )
        _ax1.text(
            _sel_i + _off1, max(_target) * 1.15, _tip1,
            ha=_ha1, va='top', fontsize=11, fontweight='bold', color='#0f172a',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#0284c7', linewidth=1.5, alpha=0.95),
            zorder=12
        )

        _ax1.set_ylabel('Avoidable Deaths', fontsize=18, fontweight='bold', color='#0B363C')
        _ax1.set_title(
            f'Estimated Avoidable Deaths ({len(_filtered_df)} Days: {_start_date.strftime("%b %d, %Y")} – {_stop_date.strftime("%b %d, %Y")}) with Regime Changes (Teal = Normal, Red = Peak Risk)',
            fontsize=20, fontweight='bold', color='#0B363C', pad=12
        )
        _ax1.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
        _ax1.tick_params(labelsize=14)

        # Subplot 2: Continuous Posterior Probability P(Surge)
        _ax2.plot(_norm_dates, _prob_surge, color='#C93B2B', linewidth=3.0, zorder=5, label='Filtered Probability P(Surge | Past Observations)')
        _ax2.plot(_norm_dates, _prob_base, color='#20808D', linewidth=2.5, linestyle='--', zorder=4, label='P(Normal | Past Observations)')
        _ax2.fill_between(_norm_dates, 0, _prob_surge, color='#C93B2B', alpha=0.25, zorder=2)
        _ax2.fill_between(_norm_dates, 0, _prob_base, color='#20808D', alpha=0.08, zorder=2)

        # Vertical inspection line across Subplot 2
        _ax2.axvline(_sel_i, color='#1e293b', linestyle='-', linewidth=2.0, alpha=0.9, zorder=10)
        _ax2.scatter([_sel_i], [_cur_surge], color='#C93B2B', s=100, edgecolor='white', linewidth=1.8, zorder=11)
        _ax2.scatter([_sel_i], [_cur_base], color='#20808D', s=100, edgecolor='white', linewidth=1.8, zorder=11)

        _n_ticks = min(7, len(_dates))
        _tick_locs = np.linspace(0, len(_dates) - 1, _n_ticks, dtype=int)
        _ax2.set_xticks(_tick_locs)
        _ax2.set_xticklabels([_dates[i].strftime('%b %d, %Y') for i in _tick_locs], fontsize=14)

        _ax2.set_ylabel('Posterior Probability', fontsize=18, fontweight='bold', color='#0B363C')
        _ax2.set_xlabel('Date', fontsize=18, fontweight='bold', color='#0B363C')
        _ax2.set_ylim(-0.02, 1.05)
        _ax2.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
        _ax2.tick_params(labelsize=14)
        _ax2.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=14)

        # Colorbar
        _cbar_ax = _fig.add_axes([0.915, 0.54, 0.015, 0.35])
        _cbar = _fig.colorbar(_lc, cax=_cbar_ax)
        _cbar.set_label('Peak Risk Probability', fontsize=14, fontweight='bold', color='#0B363C')
        _cbar.ax.tick_params(labelsize=12)

        plt.subplots_adjust(left=0.07, right=0.90, hspace=0.14, top=0.92, bottom=0.11)
        _display = plt.gca()

    _display
    return


@app.cell(hide_code=True)
def _(date_picker, hmm_df, mo):
    _start_d, _stop_d = date_picker.value
    filtered_df = hmm_df.loc[str(_start_d):str(_stop_d)].copy()
    n_days = max(len(filtered_df), 1)

    history_idx_slider = mo.ui.slider(
        start=0,
        stop=n_days - 1,
        value=n_days - 1,
        step=1,
        label="🔍 **Inspect Exact Date on History Plot (Move vertical cursor):**",
        full_width=True
    )
    history_idx_slider
    return (history_idx_slider,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How transition matrix change
    """)
    return


@app.cell(hide_code=True)
def _(cv_dates, cv_idx_slider, np, p_stay_regime0, p_stay_regime1, plt):
    import matplotlib.dates as mdates

    _n_pts = min(len(cv_dates), len(p_stay_regime0), len(p_stay_regime1))
    fig_trans, ax_trans = plt.subplots(figsize=(11, 4.8))

    if _n_pts == 0:
        ax_trans.text(0.5, 0.5, "No CV persistence data available.", ha="center", va="center")
    else:
        _valid_dates = cv_dates[:_n_pts]
        _valid_p00 = p_stay_regime0[:_n_pts]
        _valid_p11 = p_stay_regime1[:_n_pts]

        # Safely clamp selected index to available data range
        _idx = max(0, min(cv_idx_slider.value, _n_pts - 1))
        _sel_date = _valid_dates[_idx]
        _sel_p00 = _valid_p00[_idx]
        _sel_p11 = _valid_p11[_idx]

        ax_trans.plot(
            _valid_dates, _valid_p00,
            label=r"Regime 0 Persistence ($P_{00}$ - Baseline)",
            color="#006699", linewidth=2.2, marker="o", markersize=3.5, alpha=0.9
        )
        ax_trans.plot(
            _valid_dates, _valid_p11,
            label=r"Regime 1 Persistence ($P_{11}$ - Surge)",
            color="#d95f02", linewidth=2.2, marker="s", markersize=3.5, alpha=0.9
        )

        # Reference benchmark / mean lines
        mean_p00 = np.mean(_valid_p00)
        mean_p11 = np.mean(_valid_p11)
        ax_trans.axhline(mean_p00, color="#006699", linestyle="--", alpha=0.45, label=f"Mean $P_{{00}}$ ({mean_p00:.2f})")
        ax_trans.axhline(mean_p11, color="#d95f02", linestyle="--", alpha=0.45, label=f"Mean $P_{{11}}$ ({mean_p11:.2f})")

        # Set Y limits accommodating target range (0.80 to 0.95) and data
        _y_data = _valid_p00 + _valid_p11
        _y_min = min(0.80, min(_y_data) - 0.02)
        _y_max = max(0.95, max(_y_data) + 0.02)
        ax_trans.set_ylim(_y_min, _y_max)

        # Vertical inspection line
        ax_trans.axvline(_sel_date, color="#1e293b", linestyle="-", linewidth=2.0, alpha=0.9, zorder=10)

        # Highlight points at selected index
        ax_trans.scatter([_sel_date], [_sel_p00], color="#006699", s=90, edgecolor="white", linewidth=1.5, zorder=11)
        ax_trans.scatter([_sel_date], [_sel_p11], color="#d95f02", s=90, edgecolor="white", linewidth=1.5, zorder=11)

        # Floating callout text box next to vertical line
        _x_lim = ax_trans.get_xlim()
        _date_num = mdates.date2num(_sel_date)
        _x_span = _x_lim[1] - _x_lim[0]

        # If cursor is near the right edge, place text on the left; otherwise right
        if _x_span > 0 and (_date_num - _x_lim[0]) / _x_span > 0.68:
            _ha = "right"
            _offset = -0.015 * _x_span
        else:
            _ha = "left"
            _offset = 0.015 * _x_span if _x_span > 0 else 0.5

        _text_str = (
            f"Cutoff: {_sel_date.strftime('%b %d, %Y')}\n"
            f"P₀₀ (Baseline): {_sel_p00:.4f}\n"
            f"P₁₁ (Surge):     {_sel_p11:.4f}"
        )

        _y_lim = ax_trans.get_ylim()
        _text_y = _y_lim[1] - 0.03 * (_y_lim[1] - _y_lim[0])

        ax_trans.text(
            _date_num + _offset, _text_y, _text_str,
            ha=_ha, va="top", fontsize=10, fontweight="bold", color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#0284c7", linewidth=1.5, alpha=0.95),
            zorder=12
        )

        ax_trans.set_title("Stability of Regime Persistence Across CV Training Windows", fontsize=13, fontweight="bold", pad=12)
        ax_trans.set_xlabel("CV Window End Date (Training Cutoff)", fontsize=11, fontweight="medium")
        ax_trans.set_ylabel("Probability of Staying in Regime ($P_{ii}$)", fontsize=11, fontweight="medium")
        ax_trans.grid(True, linestyle=":", alpha=0.6)
        ax_trans.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=9.5, loc="lower right", ncol=2)

    for spine in ax_trans.spines.values():
        spine.set_color("#cccccc")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(cv_dates, mo):
    # Extract transition probabilities and split cutoffs from precomputed results
    cv_idx_slider = mo.ui.slider(
        start=0,
        stop=max(len(cv_dates) - 1, 0),
        value=max(len(cv_dates) - 1, 0),
        step=1,
        label="🔍 **Inspect CV Split / Date:**",
        full_width=True
    )
    cv_idx_slider
    return (cv_idx_slider,)


if __name__ == "__main__":
    app.run()
