"""
Correlation analysis between NIFTY 50 intraday movement, gaps, and India VIX.

Approaches:
1. Pearson correlation (linear)
2. Spearman rank correlation (monotonic, non-linear)
3. Kendall tau correlation (rank-based, robust to outliers)
4. Rolling correlations (time-varying relationships)
5. Lagged cross-correlations (does one lead the other?)
6. Mutual information (captures non-linear dependencies)
7. Regime-based analysis (correlations in high vs low volatility)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import mutual_info_score
from pathlib import Path
import json

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_prepare():
    """Load data and compute all metrics."""
    nifty = pd.read_csv("data/nifty50.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv("data/indiavix.csv", index_col=0, parse_dates=True)

    # Intraday movement
    is_positive = nifty["Close"] >= nifty["Open"]
    intraday_pos = (
        (nifty["Open"] - nifty["Low"])
        + (nifty["High"] - nifty["Low"])
        + (nifty["High"] - nifty["Close"])
    )
    intraday_neg = (
        (nifty["High"] - nifty["Open"])
        + (nifty["High"] - nifty["Low"])
        + (nifty["Close"] - nifty["Low"])
    )
    nifty["Intraday_Pct"] = (
        intraday_pos.where(is_positive, intraday_neg) / nifty["Open"] * 100
    )

    # Gap
    nifty["Prev_Close"] = nifty["Close"].shift(1)
    nifty["Gap_Pct"] = (nifty["Open"] - nifty["Prev_Close"]) / nifty["Prev_Close"] * 100
    nifty["Abs_Gap_Pct"] = nifty["Gap_Pct"].abs()

    # Daily return
    nifty["Return_Pct"] = nifty["Close"].pct_change() * 100

    # VIX close aligned
    nifty["VIX"] = vix["Close"].reindex(nifty.index)

    nifty = nifty.dropna(subset=["Intraday_Pct", "Gap_Pct", "VIX"])
    return nifty


def discretize(series, n_bins=30):
    """Discretize a continuous series into bins for mutual information."""
    return pd.cut(series, bins=n_bins, labels=False).dropna().astype(int)


# ─────────────────────────────────────────────────────────────
# Analysis 1: Pearson, Spearman, Kendall correlations
# ─────────────────────────────────────────────────────────────
def static_correlations(df):
    pairs = [
        ("Intraday_Pct", "VIX", "Intraday Movement vs VIX"),
        ("Intraday_Pct", "Abs_Gap_Pct", "Intraday Movement vs Abs Gap"),
        ("Intraday_Pct", "Gap_Pct", "Intraday Movement vs Gap (signed)"),
        ("Abs_Gap_Pct", "VIX", "Abs Gap vs VIX"),
        ("VIX", "Gap_Pct", "VIX vs Gap (signed)"),
        ("Intraday_Pct", "Return_Pct", "Intraday Movement vs Daily Return"),
    ]

    results = []
    for col_a, col_b, label in pairs:
        a, b = df[col_a], df[col_b]
        pearson_r, pearson_p = stats.pearsonr(a, b)
        spearman_r, spearman_p = stats.spearmanr(a, b)
        kendall_r, kendall_p = stats.kendalltau(a, b)

        results.append({
            "Pair": label,
            "Pearson r": round(pearson_r, 4),
            "Pearson p": f"{pearson_p:.2e}",
            "Spearman ρ": round(spearman_r, 4),
            "Spearman p": f"{spearman_p:.2e}",
            "Kendall τ": round(kendall_r, 4),
            "Kendall p": f"{kendall_p:.2e}",
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "static_correlations.csv", index=False)
    print("\n══════ Static Correlations ══════")
    print(results_df.to_string(index=False))
    return results_df


# ─────────────────────────────────────────────────────────────
# Analysis 2: Correlation matrix heatmap
# ─────────────────────────────────────────────────────────────
def correlation_heatmap(df):
    cols = ["Intraday_Pct", "Abs_Gap_Pct", "Gap_Pct", "VIX", "Return_Pct"]
    for method in ["pearson", "spearman", "kendall"]:
        corr = df[cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        labels = ["Intraday %", "Abs Gap %", "Gap %", "VIX", "Return %"]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.values[i, j]:.3f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(corr.values[i, j]) > 0.5 else "black")

        plt.colorbar(im, ax=ax)
        plt.title(f"Correlation Matrix ({method.title()})")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"heatmap_{method}.png", dpi=150)
    print("Saved correlation heatmaps")


# ─────────────────────────────────────────────────────────────
# Analysis 3: Rolling correlations (60-day and 252-day windows)
# ─────────────────────────────────────────────────────────────
def rolling_correlations(df):
    pairs = [
        ("Intraday_Pct", "VIX", "Intraday vs VIX"),
        ("Intraday_Pct", "Abs_Gap_Pct", "Intraday vs Abs Gap"),
        ("Abs_Gap_Pct", "VIX", "Abs Gap vs VIX"),
    ]
    windows = [60, 252]

    fig, axes = plt.subplots(len(pairs), 1, figsize=(20, 4 * len(pairs)), sharex=True)
    for idx, (col_a, col_b, label) in enumerate(pairs):
        ax = axes[idx]
        for w in windows:
            rolling_corr = df[col_a].rolling(w).corr(df[col_b])
            ax.plot(df.index, rolling_corr, linewidth=0.8, label=f"{w}-day rolling", alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Correlation")
        ax.set_title(f"Rolling Correlation: {label}")
        ax.legend(loc="lower right")
        ax.set_ylim(-1, 1)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rolling_correlations.png", dpi=150)
    print("Saved rolling_correlations.png")


# ─────────────────────────────────────────────────────────────
# Analysis 4: Lagged cross-correlation
# ─────────────────────────────────────────────────────────────
def lagged_correlations(df):
    pairs = [
        ("VIX", "Intraday_Pct", "VIX → Intraday Movement"),
        ("Intraday_Pct", "VIX", "Intraday Movement → VIX"),
        ("Abs_Gap_Pct", "Intraday_Pct", "Abs Gap → Intraday Movement"),
        ("VIX", "Abs_Gap_Pct", "VIX → Abs Gap"),
    ]
    max_lag = 20

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    results = []
    for idx, (leader, follower, label) in enumerate(pairs):
        ax = axes[idx // 2][idx % 2]
        lags = list(range(-max_lag, max_lag + 1))
        corrs = []
        leader_vals = df[leader].values
        follower_vals = df[follower].values
        n = len(leader_vals)
        for lag in lags:
            if lag >= 0:
                corr = np.corrcoef(leader_vals[:n - lag], follower_vals[lag:])[0, 1] if lag < n else 0
            else:
                corr = np.corrcoef(leader_vals[-lag:], follower_vals[:n + lag])[0, 1] if -lag < n else 0
            corrs.append(corr)

        ax.bar(list(lags), corrs, color="steelblue", alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel(f"Lag (days) — positive = {leader.replace('_Pct','')} leads")
        ax.set_ylabel("Pearson r")
        ax.set_title(label)

        best_lag = list(lags)[np.argmax(np.abs(corrs))]
        best_corr = corrs[np.argmax(np.abs(corrs))]
        results.append({"Pair": label, "Best Lag": best_lag, "Correlation at Best Lag": round(best_corr, 4)})

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lagged_correlations.png", dpi=150)

    lag_df = pd.DataFrame(results)
    lag_df.to_csv(OUTPUT_DIR / "lagged_correlations.csv", index=False)
    print("\n══════ Lagged Cross-Correlations ══════")
    print(lag_df.to_string(index=False))
    print("Saved lagged_correlations.png")


# ─────────────────────────────────────────────────────────────
# Analysis 5: Mutual information (non-linear dependency)
# ─────────────────────────────────────────────────────────────
def mutual_information_analysis(df):
    cols = ["Intraday_Pct", "Abs_Gap_Pct", "Gap_Pct", "VIX", "Return_Pct"]
    labels = ["Intraday %", "Abs Gap %", "Gap %", "VIX", "Return %"]
    n = len(cols)
    mi_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = discretize(df[cols[i]])
            b = discretize(df[cols[j]])
            common = a.index.intersection(b.index)
            mi_matrix[i, j] = mutual_info_score(a.loc[common], b.loc[common])

    # Normalize to [0, 1] using NMI-like scaling
    mi_normalized = mi_matrix / mi_matrix.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mi_normalized, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mi_normalized[i, j]:.3f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if mi_normalized[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Normalized MI")
    plt.title("Mutual Information (Normalized) — Non-linear Dependencies")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "mutual_information.png", dpi=150)

    mi_df = pd.DataFrame(mi_normalized, index=labels, columns=labels).round(4)
    mi_df.to_csv(OUTPUT_DIR / "mutual_information.csv")
    print("\n══════ Mutual Information (Normalized) ══════")
    print(mi_df.to_string())
    print("Saved mutual_information.png")


# ─────────────────────────────────────────────────────────────
# Analysis 6: Regime-based correlations (high vs low VIX)
# ─────────────────────────────────────────────────────────────
def regime_analysis(df):
    vix_median = df["VIX"].median()
    vix_75 = df["VIX"].quantile(0.75)

    regimes = {
        f"Low VIX (< {vix_median:.1f})": df[df["VIX"] < vix_median],
        f"High VIX (>= {vix_median:.1f})": df[df["VIX"] >= vix_median],
        f"Very High VIX (>= {vix_75:.1f})": df[df["VIX"] >= vix_75],
    }

    pairs = [
        ("Intraday_Pct", "VIX"),
        ("Intraday_Pct", "Abs_Gap_Pct"),
        ("Abs_Gap_Pct", "VIX"),
    ]

    results = []
    for regime_name, regime_df in regimes.items():
        row = {"Regime": regime_name, "N": len(regime_df)}
        for col_a, col_b in pairs:
            r, p = stats.spearmanr(regime_df[col_a], regime_df[col_b])
            row[f"{col_a} vs {col_b} (ρ)"] = round(r, 4)
        results.append(row)

    regime_df = pd.DataFrame(results)
    regime_df.to_csv(OUTPUT_DIR / "regime_correlations.csv", index=False)
    print("\n══════ Regime-Based Correlations (Spearman) ══════")
    print(regime_df.to_string(index=False))

    # Box plots of intraday movement by VIX regime
    df["VIX_Regime"] = pd.cut(
        df["VIX"],
        bins=[0, 12, 16, 20, 30, 100],
        labels=["<12", "12-16", "16-20", "20-30", ">30"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df.boxplot(column="Intraday_Pct", by="VIX_Regime", ax=axes[0])
    axes[0].set_title("Intraday Movement % by VIX Regime")
    axes[0].set_xlabel("VIX Range")
    axes[0].set_ylabel("Intraday Movement %")
    plt.sca(axes[0])
    plt.title("Intraday Movement % by VIX Regime")

    df.boxplot(column="Abs_Gap_Pct", by="VIX_Regime", ax=axes[1])
    axes[1].set_title("Abs Gap % by VIX Regime")
    axes[1].set_xlabel("VIX Range")
    axes[1].set_ylabel("Abs Gap %")
    plt.sca(axes[1])
    plt.title("Abs Gap % by VIX Regime")

    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "regime_boxplots.png", dpi=150)
    print("Saved regime_boxplots.png")


# ─────────────────────────────────────────────────────────────
# Analysis 7: Scatter plots with regression lines
# ─────────────────────────────────────────────────────────────
def scatter_analysis(df):
    pairs = [
        ("VIX", "Intraday_Pct", "VIX", "Intraday Movement %"),
        ("VIX", "Abs_Gap_Pct", "VIX", "Abs Gap %"),
        ("Abs_Gap_Pct", "Intraday_Pct", "Abs Gap %", "Intraday Movement %"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, (col_x, col_y, lbl_x, lbl_y) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(df[col_x], df[col_y], alpha=0.15, s=5, color="steelblue")

        # Linear regression
        slope, intercept, r, p, se = stats.linregress(df[col_x], df[col_y])
        x_line = np.linspace(df[col_x].min(), df[col_x].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1.5,
                label=f"r={r:.3f}, slope={slope:.3f}")

        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)
        ax.set_title(f"{lbl_x} vs {lbl_y}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_regression.png", dpi=150)
    print("Saved scatter_regression.png")


# ─────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────
def generate_summary(df, static_df):
    summary = []
    summary.append(f"Data period: {df.index.min().date()} to {df.index.max().date()}")
    summary.append(f"Total trading days analyzed: {len(df)}")
    summary.append(f"")
    summary.append(f"Mean Intraday Movement: {df['Intraday_Pct'].mean():.2f}%")
    summary.append(f"Mean Abs Gap: {df['Abs_Gap_Pct'].mean():.2f}%")
    summary.append(f"Mean VIX: {df['VIX'].mean():.2f}")
    summary.append(f"Median VIX: {df['VIX'].median():.2f}")

    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write("\n".join(summary))
    return summary


def main():
    print("Loading data...")
    df = load_and_prepare()
    print(f"Analyzing {len(df)} trading days from {df.index.min().date()} to {df.index.max().date()}")

    static_df = static_correlations(df)
    correlation_heatmap(df)
    rolling_correlations(df)
    lagged_correlations(df)
    mutual_information_analysis(df)
    regime_analysis(df)
    scatter_analysis(df)
    summary = generate_summary(df, static_df)

    print("\n══════ Summary ══════")
    for line in summary:
        print(line)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
