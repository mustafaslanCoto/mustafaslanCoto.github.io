import marimo

__generated_with = "1.10.18"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import kendalltau, norm, spearmanr

    return kendalltau, mo, norm, np, plt, spearmanr


@app.cell
def _(mo):
    mo.md(
        r"""
        # Copulas: separating marginals from dependence

        A copula describes how random variables depend on one another, separately
        from the shape of each variable's distribution. This is useful when two
        variables have very different marginal distributions but their outcomes
        still move together.

        **Sklar's theorem** says a joint distribution can be written as
        \(F_{X,Y}(x,y) = C(F_X(x), F_Y(y))\), where \(F_X\) and \(F_Y\) are the
        marginal CDFs and \(C\) is a copula.

        Use the controls to change the dependence in a Gaussian copula. The
        resulting variables keep an exponential and a Weibull marginal,
        respectively.
        """
    )
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(
        start=-0.95,
        stop=0.95,
        step=0.05,
        value=0.65,
        label="Latent Gaussian correlation",
    )
    n_samples = mo.ui.slider(
        start=200,
        stop=3000,
        step=200,
        value=1000,
        label="Number of observations",
    )
    mo.hstack([rho, n_samples], justify="start", gap=2)
    return n_samples, rho


@app.cell
def _(kendalltau, mo, n_samples, norm, np, plt, rho, spearmanr):
    _latent = np.random.default_rng(2026).multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, rho.value], [rho.value, 1.0]],
        size=n_samples.value,
    )
    _uniform = norm.cdf(_latent)

    # Inverse CDF transforms change the marginals without changing the copula.
    _x = -np.log1p(-_uniform[:, 0])
    _y = (-np.log1p(-_uniform[:, 1])) ** (1 / 1.5)

    _fig, _axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
        gridspec_kw={"width_ratios": [3, 1, 1]},
    )
    _axes[0].scatter(_x, _y, alpha=0.45, s=16, edgecolors="none")
    _axes[0].set(
        xlabel="X (Exponential, rate 1)",
        ylabel="Y (Weibull, shape 1.5)",
        title="Joint sample",
    )
    _axes[1].hist(_x, bins=25, density=True, color="#3973ac", alpha=0.8)
    _axes[1].set(title="X marginal", xlabel="X", ylabel="Density")
    _axes[2].hist(_y, bins=25, density=True, color="#d17a22", alpha=0.8)
    _axes[2].set(title="Y marginal", xlabel="Y")
    _fig.tight_layout()

    _tau = kendalltau(_x, _y).statistic
    _rho_s = spearmanr(_x, _y).statistic
    _theoretical_tau = 2 / np.pi * np.arcsin(rho.value)
    figure = _fig
    dependence_summary = mo.md(
        f"""
        **Dependence summary** — sample Kendall's \\(\\tau\\): **{_tau:.3f}**
        (Gaussian-copula value: **{_theoretical_tau:.3f}**);
        sample Spearman's \\(\\rho_s\\): **{_rho_s:.3f}**.

        Changing the copula parameter changes the association, not the
        marginal distribution families.
        """
    )
    mo.vstack([figure, dependence_summary])
    return dependence_summary, figure


if __name__ == "__main__":
    app.run()
