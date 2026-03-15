"""
Breakout continuation analysis for NIFTY 50.

When the market closes below the previous day's low (bearish breakout)
or above the previous day's high (bullish breakout), does it continue
in that direction? And if so, for how long?

Also measures the velocity (magnitude / time) of breakout moves before
consolidation or mean reversion.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    nifty = pd.read_csv("data/nifty50.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv("data/indiavix.csv", index_col=0, parse_dates=True)
    nifty["Prev_High"] = nifty["High"].shift(1)
    nifty["Prev_Low"] = nifty["Low"].shift(1)
    nifty["Prev_Close"] = nifty["Close"].shift(1)
    nifty["VIX"] = vix["Close"].reindex(nifty.index)
    nifty = nifty.dropna()
    return nifty


def identify_breakouts(df):
    """Tag days where close breaks previous day's range."""
    df["Bullish_Breakout"] = df["Close"] > df["Prev_High"]
    df["Bearish_Breakout"] = df["Close"] < df["Prev_Low"]
    return df


def measure_continuation(df, breakout_col, direction):
    """
    For each breakout day, look forward and measure:
    1. Streak: consecutive days closing in the breakout direction
    2. Peak excursion: max favorable move (using intraday high/low) before mean reversion
    3. Velocity: peak excursion / days to reach peak

    For bullish: continuation = next day closes above current day's close
    For bearish: continuation = next day closes below current day's close
    """
    breakout_days = df.index[df[breakout_col]]
    results = []

    for day in breakout_days:
        loc = df.index.get_loc(day)
        streak = 0
        base_close = df.iloc[loc]["Close"]
        base_open = df.iloc[loc]["Open"]
        prev_close = base_close

        # --- Streak (close-to-close) ---
        for fwd in range(1, min(61, len(df) - loc)):
            next_row = df.iloc[loc + fwd]
            if direction == "bull":
                continues = next_row["Close"] > prev_close
            else:
                continues = next_row["Close"] < prev_close

            if continues:
                streak += 1
                prev_close = next_row["Close"]
            else:
                break

        total_continuation_return = (prev_close - base_close) / base_close * 100

        # --- Peak excursion (using intraday highs/lows) ---
        # Look forward up to 60 days, track the most extreme favorable price
        # before 2 consecutive closes back inside the breakout level
        peak_price = base_close
        peak_day_offset = 0
        reversal_count = 0
        look_ahead = min(61, len(df) - loc)

        for fwd in range(1, look_ahead):
            row = df.iloc[loc + fwd]
            if direction == "bull":
                # Track highest intraday high
                if row["High"] > peak_price:
                    peak_price = row["High"]
                    peak_day_offset = fwd
                    reversal_count = 0
                # Mean reversion: close drops below breakout day's close
                if row["Close"] < base_close:
                    reversal_count += 1
                else:
                    reversal_count = 0
            else:
                # Track lowest intraday low
                if row["Low"] < peak_price:
                    peak_price = row["Low"]
                    peak_day_offset = fwd
                    reversal_count = 0
                # Mean reversion: close rises above breakout day's close
                if row["Close"] > base_close:
                    reversal_count += 1
                else:
                    reversal_count = 0

            # Stop after 2 consecutive reversals
            if reversal_count >= 2:
                break

        peak_excursion_pct = abs(peak_price - base_close) / base_close * 100
        days_to_peak = peak_day_offset
        velocity = peak_excursion_pct / days_to_peak if days_to_peak > 0 else peak_excursion_pct

        # --- Breakout magnitude on the breakout day itself ---
        if direction == "bull":
            breakout_magnitude = (base_close - df.iloc[loc]["Prev_High"]) / df.iloc[loc]["Prev_High"] * 100
        else:
            breakout_magnitude = (df.iloc[loc]["Prev_Low"] - base_close) / df.iloc[loc]["Prev_Low"] * 100

        # --- Fixed forward returns ---
        fwd_returns = {}
        for n in [1, 2, 3, 5, 10, 20]:
            if loc + n < len(df):
                fwd_close = df.iloc[loc + n]["Close"]
                fwd_returns[f"Fwd_{n}d_Ret"] = (fwd_close - base_close) / base_close * 100
            else:
                fwd_returns[f"Fwd_{n}d_Ret"] = np.nan

        # --- VIX on breakout day and forward ---
        vix_day0 = df.iloc[loc]["VIX"]
        vix_fwd = {}
        for n in [1, 2, 3, 5, 10]:
            if loc + n < len(df):
                vix_fwd[f"VIX_Day{n}"] = df.iloc[loc + n]["VIX"]
            else:
                vix_fwd[f"VIX_Day{n}"] = np.nan

        results.append({
            "Date": day,
            "Close": base_close,
            "Streak": streak,
            "Continuation_Return_Pct": total_continuation_return,
            "Breakout_Magnitude_Pct": breakout_magnitude,
            "Peak_Excursion_Pct": peak_excursion_pct,
            "Days_To_Peak": days_to_peak,
            "Velocity_Pct_Per_Day": velocity,
            "VIX": vix_day0,
            **vix_fwd,
            **fwd_returns,
        })

    return pd.DataFrame(results)


def print_stats(label, cont_df, direction):
    """Print summary statistics for continuation."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total breakout days: {len(cont_df)}")
    print(f"  Mean streak length: {cont_df['Streak'].mean():.2f} days")
    print(f"  Median streak length: {cont_df['Streak'].median():.1f} days")
    print(f"  Max streak: {cont_df['Streak'].max()} days")
    print()

    # Streak distribution
    streak_counts = cont_df["Streak"].value_counts().sort_index()
    print("  Streak Distribution:")
    for s in range(min(11, streak_counts.index.max() + 1)):
        count = streak_counts.get(s, 0)
        pct = count / len(cont_df) * 100
        bar = "#" * int(pct)
        print(f"    {s:2d} days: {count:4d} ({pct:5.1f}%) {bar}")
    if streak_counts.index.max() >= 11:
        count_11plus = cont_df[cont_df["Streak"] >= 11].shape[0]
        pct = count_11plus / len(cont_df) * 100
        print(f"   11+ days: {count_11plus:4d} ({pct:5.1f}%)")

    # Velocity / magnitude stats
    print()
    print("  Breakout Magnitude & Velocity:")
    print(f"    Breakout day magnitude:  mean={cont_df['Breakout_Magnitude_Pct'].mean():.3f}%  "
          f"median={cont_df['Breakout_Magnitude_Pct'].median():.3f}%")
    print(f"    Peak excursion:          mean={cont_df['Peak_Excursion_Pct'].mean():.3f}%  "
          f"median={cont_df['Peak_Excursion_Pct'].median():.3f}%")
    print(f"    Days to peak:            mean={cont_df['Days_To_Peak'].mean():.1f}  "
          f"median={cont_df['Days_To_Peak'].median():.1f}")
    print(f"    Velocity (%/day):        mean={cont_df['Velocity_Pct_Per_Day'].mean():.3f}  "
          f"median={cont_df['Velocity_Pct_Per_Day'].median():.3f}")

    # Forward returns
    sign = 1 if direction == "bull" else -1
    print()
    print("  Forward Returns (mean):")
    for n in [1, 2, 3, 5, 10, 20]:
        col = f"Fwd_{n}d_Ret"
        mean_ret = cont_df[col].mean()
        median_ret = cont_df[col].median()
        win_rate = (cont_df[col] * sign > 0).mean() * 100
        print(f"    {n:2d}-day: mean={mean_ret:+.3f}%  median={median_ret:+.3f}%  "
              f"win_rate={win_rate:.1f}%")


def plot_streak_distribution(bull_df, bear_df):
    """Plot streak length distributions side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cont_df, label, color in [
        (axes[0], bull_df, "Bullish Breakout\n(Close > Prev High)", "forestgreen"),
        (axes[1], bear_df, "Bearish Breakout\n(Close < Prev Low)", "crimson"),
    ]:
        max_streak = min(15, cont_df["Streak"].max())
        bins = range(0, max_streak + 2)
        streaks_capped = cont_df["Streak"].clip(upper=max_streak)
        ax.hist(streaks_capped, bins=bins, color=color, alpha=0.7, edgecolor="white",
                align="left", rwidth=0.8)
        ax.set_xlabel("Consecutive Days of Continuation")
        ax.set_ylabel("Frequency")
        ax.set_title(label)
        ax.set_xticks(range(0, max_streak + 1))

        mean_s = cont_df["Streak"].mean()
        median_s = cont_df["Streak"].median()
        ax.axvline(mean_s, color="black", linestyle="--", linewidth=1, label=f"Mean={mean_s:.1f}")
        ax.axvline(median_s, color="orange", linestyle="--", linewidth=1, label=f"Median={median_s:.1f}")
        ax.legend()

    plt.suptitle("NIFTY 50 — Breakout Continuation Streak Distribution", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "continuation_streaks.png", dpi=150)
    print("\nSaved continuation_streaks.png")


def plot_forward_returns(bull_df, bear_df):
    """Plot average forward returns after breakouts."""
    periods = [1, 2, 3, 5, 10, 20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cont_df, label, color in [
        (axes[0], bull_df, "After Bullish Breakout", "forestgreen"),
        (axes[1], bear_df, "After Bearish Breakout", "crimson"),
    ]:
        means = [cont_df[f"Fwd_{n}d_Ret"].mean() for n in periods]
        medians = [cont_df[f"Fwd_{n}d_Ret"].median() for n in periods]
        x = range(len(periods))

        ax.bar([i - 0.15 for i in x], means, width=0.3, color=color, alpha=0.7, label="Mean")
        ax.bar([i + 0.15 for i in x], medians, width=0.3, color=color, alpha=0.4, label="Median")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}d" for n in periods])
        ax.set_xlabel("Forward Period")
        ax.set_ylabel("Return %")
        ax.set_title(label)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()

    plt.suptitle("NIFTY 50 — Forward Returns After Breakouts", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "continuation_forward_returns.png", dpi=150)
    print("Saved continuation_forward_returns.png")


def plot_win_rates(bull_df, bear_df):
    """Plot win rates (% of times continuation holds) at each forward period."""
    periods = [1, 2, 3, 5, 10, 20]

    fig, ax = plt.subplots(figsize=(10, 6))

    bull_wr = [(bull_df[f"Fwd_{n}d_Ret"] > 0).mean() * 100 for n in periods]
    bear_wr = [(bear_df[f"Fwd_{n}d_Ret"] < 0).mean() * 100 for n in periods]

    x = range(len(periods))
    ax.plot(x, bull_wr, "o-", color="forestgreen", linewidth=2, markersize=8, label="Bullish (% closing higher)")
    ax.plot(x, bear_wr, "o-", color="crimson", linewidth=2, markersize=8, label="Bearish (% closing lower)")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% (random)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}d" for n in periods])
    ax.set_xlabel("Forward Period")
    ax.set_ylabel("Win Rate %")
    ax.set_ylim(30, 70)
    ax.set_title("NIFTY 50 — Continuation Win Rate After Breakouts")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "continuation_win_rates.png", dpi=150)
    print("Saved continuation_win_rates.png")


def plot_breakout_timeline(df, bull_df, bear_df):
    """Plot NIFTY price with breakout days marked."""
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(df.index, df["Close"], color="gray", linewidth=0.5, alpha=0.7, label="NIFTY Close")

    bull_dates = pd.to_datetime(bull_df["Date"])
    bear_dates = pd.to_datetime(bear_df["Date"])
    ax.scatter(bull_dates, df.loc[bull_dates, "Close"],
               color="forestgreen", s=3, alpha=0.5, label=f"Bullish Breakout (n={len(bull_df)})", zorder=3)
    ax.scatter(bear_dates, df.loc[bear_dates, "Close"],
               color="crimson", s=3, alpha=0.5, label=f"Bearish Breakout (n={len(bear_df)})", zorder=3)

    ax.set_ylabel("NIFTY Close")
    ax.set_xlabel("Date")
    ax.set_title("NIFTY 50 — Breakout Days (Close > Prev High / Close < Prev Low)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "continuation_timeline.png", dpi=150)
    print("Saved continuation_timeline.png")


def plot_streak_vs_return(bull_df, bear_df):
    """Show how continuation return scales with streak length."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cont_df, label, color in [
        (axes[0], bull_df, "Bullish Breakout", "forestgreen"),
        (axes[1], bear_df, "Bearish Breakout", "crimson"),
    ]:
        # Group by streak and show mean continuation return
        max_s = min(10, cont_df["Streak"].max())
        grouped = cont_df[cont_df["Streak"] <= max_s].groupby("Streak")
        means = grouped["Continuation_Return_Pct"].mean()
        counts = grouped["Continuation_Return_Pct"].count()

        ax.bar(means.index, means.values, color=color, alpha=0.7, edgecolor="white")
        ax.set_xlabel("Streak Length (days)")
        ax.set_ylabel("Mean Continuation Return %")
        ax.set_title(f"{label}: Return by Streak Length")
        ax.set_xticks(range(0, max_s + 1))

        # Annotate counts
        for s, m, c in zip(means.index, means.values, counts.values):
            ax.text(s, m + (0.05 if m >= 0 else -0.15), f"n={c}", ha="center", fontsize=8)

    plt.suptitle("NIFTY 50 — Continuation Return by Streak Length", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "continuation_streak_vs_return.png", dpi=150)
    print("Saved continuation_streak_vs_return.png")


def plot_velocity_distribution(bull_df, bear_df):
    """Plot distributions of breakout magnitude, peak excursion, days to peak, and velocity."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("Breakout_Magnitude_Pct", "Breakout Day Magnitude %", axes[0, 0]),
        ("Peak_Excursion_Pct", "Peak Excursion % (before mean reversion)", axes[0, 1]),
        ("Days_To_Peak", "Days to Peak", axes[1, 0]),
        ("Velocity_Pct_Per_Day", "Velocity (% per day)", axes[1, 1]),
    ]

    for col, label, ax in metrics:
        bull_data = bull_df[col].clip(upper=bull_df[col].quantile(0.95))
        bear_data = bear_df[col].clip(upper=bear_df[col].quantile(0.95))

        ax.hist(bull_data, bins=50, alpha=0.6, color="forestgreen", label="Bullish", density=True)
        ax.hist(bear_data, bins=50, alpha=0.6, color="crimson", label="Bearish", density=True)
        ax.axvline(bull_df[col].median(), color="forestgreen", linestyle="--", linewidth=1.5,
                   label=f"Bull median={bull_df[col].median():.2f}")
        ax.axvline(bear_df[col].median(), color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Bear median={bear_df[col].median():.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("NIFTY 50 — Breakout Velocity & Magnitude Distributions", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "velocity_distributions.png", dpi=150)
    print("Saved velocity_distributions.png")


def plot_magnitude_vs_continuation(bull_df, bear_df):
    """Does a larger initial breakout magnitude predict longer continuation or bigger peak excursion?"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col_idx, (cont_df, label, color) in enumerate([
        (bull_df, "Bullish", "forestgreen"),
        (bear_df, "Bearish", "crimson"),
    ]):
        # Magnitude vs Peak Excursion
        ax = axes[0, col_idx]
        ax.scatter(cont_df["Breakout_Magnitude_Pct"], cont_df["Peak_Excursion_Pct"],
                   alpha=0.15, s=8, color=color)
        from scipy import stats as sp_stats
        slope, intercept, r, p, se = sp_stats.linregress(
            cont_df["Breakout_Magnitude_Pct"], cont_df["Peak_Excursion_Pct"])
        x_line = np.linspace(0, cont_df["Breakout_Magnitude_Pct"].quantile(0.95), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5,
                label=f"r={r:.3f}")
        ax.set_xlabel("Breakout Day Magnitude %")
        ax.set_ylabel("Peak Excursion %")
        ax.set_title(f"{label}: Magnitude vs Peak Excursion")
        ax.legend()
        ax.set_xlim(0, cont_df["Breakout_Magnitude_Pct"].quantile(0.95))
        ax.set_ylim(0, cont_df["Peak_Excursion_Pct"].quantile(0.95))

        # Magnitude vs Streak
        ax = axes[1, col_idx]
        # Bin magnitude into quintiles and show mean streak
        cont_df_copy = cont_df.copy()
        cont_df_copy["Mag_Bin"] = pd.qcut(cont_df_copy["Breakout_Magnitude_Pct"], q=5, duplicates="drop")
        grouped = cont_df_copy.groupby("Mag_Bin", observed=True).agg(
            Mean_Streak=("Streak", "mean"),
            Mean_Peak=("Peak_Excursion_Pct", "mean"),
            Mean_Velocity=("Velocity_Pct_Per_Day", "mean"),
            Count=("Streak", "count"),
        ).reset_index()

        x = range(len(grouped))
        bars = ax.bar(x, grouped["Mean_Streak"], color=color, alpha=0.7, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b.left:.2f}-{b.right:.2f}" for b in grouped["Mag_Bin"]],
                           rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Breakout Magnitude % (quintiles)")
        ax.set_ylabel("Mean Streak (days)")
        ax.set_title(f"{label}: Breakout Size vs Continuation Length")
        for i, (s, c) in enumerate(zip(grouped["Mean_Streak"], grouped["Count"])):
            ax.text(i, s + 0.02, f"n={c}", ha="center", fontsize=8)

    plt.suptitle("NIFTY 50 — Does Breakout Size Predict Continuation?", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "magnitude_vs_continuation.png", dpi=150)
    print("Saved magnitude_vs_continuation.png")


def plot_velocity_by_streak(bull_df, bear_df):
    """How does velocity change with streak length?"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cont_df, label, color in [
        (axes[0], bull_df, "Bullish Breakout", "forestgreen"),
        (axes[1], bear_df, "Bearish Breakout", "crimson"),
    ]:
        # Only streaks with at least 1 day of continuation
        moving = cont_df[cont_df["Streak"] >= 1].copy()
        max_s = min(8, moving["Streak"].max())
        grouped = moving[moving["Streak"] <= max_s].groupby("Streak")

        means_vel = grouped["Velocity_Pct_Per_Day"].mean()
        means_peak = grouped["Peak_Excursion_Pct"].mean()
        counts = grouped["Velocity_Pct_Per_Day"].count()

        x = np.arange(len(means_vel))
        width = 0.35
        ax.bar(x - width / 2, means_vel, width, color=color, alpha=0.7, label="Velocity (%/day)")
        ax.bar(x + width / 2, means_peak, width, color=color, alpha=0.35, label="Peak Excursion %")
        ax.set_xticks(x)
        ax.set_xticklabels(means_vel.index)
        ax.set_xlabel("Streak Length (days)")
        ax.set_ylabel("% ")
        ax.set_title(f"{label}")
        ax.legend()

        for i, c in enumerate(counts):
            ax.text(i, max(means_vel.iloc[i], means_peak.iloc[i]) + 0.05,
                    f"n={c}", ha="center", fontsize=8)

    plt.suptitle("NIFTY 50 — Velocity & Peak Excursion by Streak Length", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "velocity_by_streak.png", dpi=150)
    print("Saved velocity_by_streak.png")


def print_velocity_summary(bull_df, bear_df):
    """Print a concise velocity summary table."""
    print(f"\n{'='*70}")
    print("  VELOCITY SUMMARY")
    print(f"{'='*70}")

    header = f"  {'Metric':<35} {'Bullish':>12} {'Bearish':>12}"
    print(header)
    print("  " + "-" * 59)

    metrics = [
        ("Breakout day magnitude (median)", "Breakout_Magnitude_Pct", "median"),
        ("Breakout day magnitude (mean)", "Breakout_Magnitude_Pct", "mean"),
        ("Peak excursion (median)", "Peak_Excursion_Pct", "median"),
        ("Peak excursion (mean)", "Peak_Excursion_Pct", "mean"),
        ("Days to peak (median)", "Days_To_Peak", "median"),
        ("Days to peak (mean)", "Days_To_Peak", "mean"),
        ("Velocity %/day (median)", "Velocity_Pct_Per_Day", "median"),
        ("Velocity %/day (mean)", "Velocity_Pct_Per_Day", "mean"),
    ]

    for label, col, agg in metrics:
        bull_val = getattr(bull_df[col], agg)()
        bear_val = getattr(bear_df[col], agg)()
        fmt = ".3f" if "Days" not in label else ".1f"
        print(f"  {label:<35} {bull_val:>12{fmt}}% {bear_val:>12{fmt}}%"
              if "Days" not in label else
              f"  {label:<35} {bull_val:>12{fmt}} {bear_val:>12{fmt}}")


def measure_recovery(df, cont_df, direction, max_lookforward=252):
    """
    For each breakout, measure how many days until price recovers back
    to the breakout-day close level.

    For bearish: recovery = close >= breakdown day close
    For bullish: recovery = close <= breakout day close (i.e. gives back the gains)

    Returns the cont_df with added columns:
    - Days_To_Recovery: trading days until recovery (NaN if not recovered within max_lookforward)
    - Max_Drawdown_Pct: worst adverse move before recovery (for bearish: how much further it fell)
    """
    recovery_days = []
    max_adverse = []

    for _, row in cont_df.iterrows():
        date = row["Date"]
        base_close = row["Close"]
        loc = df.index.get_loc(date)

        recovered = False
        worst = 0.0
        look_ahead = min(max_lookforward + 1, len(df) - loc)

        for fwd in range(1, look_ahead):
            fwd_row = df.iloc[loc + fwd]
            if direction == "bear":
                # Track how much further it fell
                drawdown = (fwd_row["Low"] - base_close) / base_close * 100
                worst = min(worst, drawdown)
                # Recovery: close back at or above breakdown level
                if fwd_row["Close"] >= base_close:
                    recovery_days.append(fwd)
                    max_adverse.append(abs(worst))
                    recovered = True
                    break
            else:
                # Track how much further it ran up
                excursion = (fwd_row["High"] - base_close) / base_close * 100
                worst = max(worst, excursion)
                # Recovery (giveback): close back at or below breakout level
                if fwd_row["Close"] <= base_close:
                    recovery_days.append(fwd)
                    max_adverse.append(abs(worst))
                    recovered = True
                    break

        if not recovered:
            recovery_days.append(np.nan)
            max_adverse.append(abs(worst) if direction == "bear" else abs(worst))

    cont_df = cont_df.copy()
    cont_df["Days_To_Recovery"] = recovery_days
    cont_df["Max_Adverse_Pct"] = max_adverse
    return cont_df


def print_recovery_stats(cont_df, label):
    """Print recovery statistics for bearish breakdowns."""
    recovered = cont_df.dropna(subset=["Days_To_Recovery"])
    not_recovered = cont_df[cont_df["Days_To_Recovery"].isna()]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total breakdowns: {len(cont_df)}")
    print(f"  Recovered within 252 days: {len(recovered)} ({len(recovered)/len(cont_df)*100:.1f}%)")
    print(f"  Not recovered within 252 days: {len(not_recovered)} ({len(not_recovered)/len(cont_df)*100:.1f}%)")
    print()

    if len(recovered) > 0:
        days = recovered["Days_To_Recovery"]
        print("  Days to Recovery (among recovered):")
        print(f"    Mean:   {days.mean():.1f} days")
        print(f"    Median: {days.median():.1f} days")
        print(f"    P25:    {days.quantile(0.25):.0f} days")
        print(f"    P75:    {days.quantile(0.75):.0f} days")
        print(f"    P90:    {days.quantile(0.90):.0f} days")
        print(f"    Max:    {days.max():.0f} days")
        print()

        # Recovery buckets
        buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 60), (61, 252)]
        print("  Recovery Time Distribution:")
        for lo, hi in buckets:
            count = ((days >= lo) & (days <= hi)).sum()
            pct = count / len(cont_df) * 100
            label_range = f"{lo}d" if lo == hi else f"{lo}-{hi}d"
            bar = "#" * int(pct)
            print(f"    {label_range:>8}: {count:4d} ({pct:5.1f}%) {bar}")

        print()
        print(f"  Max adverse move before recovery (median): {recovered['Max_Adverse_Pct'].median():.3f}%")
        print(f"  Max adverse move before recovery (mean):   {recovered['Max_Adverse_Pct'].mean():.3f}%")


def plot_recovery_distribution(bear_df):
    """Plot distribution of recovery times after bearish breakdowns."""
    recovered = bear_df.dropna(subset=["Days_To_Recovery"])
    not_recovered_count = bear_df["Days_To_Recovery"].isna().sum()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Histogram of recovery days
    ax = axes[0, 0]
    days = recovered["Days_To_Recovery"]
    ax.hist(days, bins=50, color="crimson", alpha=0.7, edgecolor="white")
    ax.axvline(days.median(), color="black", linestyle="--", linewidth=1.5,
               label=f"Median={days.median():.0f} days")
    ax.axvline(days.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean={days.mean():.0f} days")
    ax.set_xlabel("Days to Recovery")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Recovery Time Distribution (n={len(recovered)}, {not_recovered_count} never recovered)")
    ax.legend()

    # 2. CDF of recovery
    ax = axes[0, 1]
    sorted_days = np.sort(days.values)
    cdf = np.arange(1, len(sorted_days) + 1) / len(bear_df)  # denominator includes non-recovered
    ax.plot(sorted_days, cdf * 100, color="crimson", linewidth=2)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(75, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.5)
    # Mark key percentiles
    for pct_target in [50, 75, 90]:
        idx = np.searchsorted(cdf * 100, pct_target)
        if idx < len(sorted_days):
            ax.annotate(f"{pct_target}% by day {sorted_days[idx]:.0f}",
                        xy=(sorted_days[idx], pct_target),
                        xytext=(sorted_days[idx] + 15, pct_target - 5),
                        arrowprops=dict(arrowstyle="->", color="black"),
                        fontsize=9)
    ax.set_xlabel("Days After Breakdown")
    ax.set_ylabel("% of Breakdowns Recovered")
    ax.set_title("Cumulative Recovery Rate")
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 105)

    # 3. Recovery time vs breakdown magnitude
    ax = axes[1, 0]
    ax.scatter(recovered["Breakout_Magnitude_Pct"], recovered["Days_To_Recovery"],
               alpha=0.2, s=10, color="crimson")
    from scipy import stats as sp_stats
    slope, intercept, r, p, se = sp_stats.linregress(
        recovered["Breakout_Magnitude_Pct"], recovered["Days_To_Recovery"])
    x_line = np.linspace(0, recovered["Breakout_Magnitude_Pct"].quantile(0.95), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5,
            label=f"r={r:.3f}")
    ax.set_xlabel("Breakdown Magnitude %")
    ax.set_ylabel("Days to Recovery")
    ax.set_title("Does Breakdown Size Predict Recovery Time?")
    ax.set_xlim(0, recovered["Breakout_Magnitude_Pct"].quantile(0.95))
    ax.set_ylim(0, min(260, recovered["Days_To_Recovery"].quantile(0.95)))
    ax.legend()

    # 4. Max adverse move vs recovery time
    ax = axes[1, 1]
    ax.scatter(recovered["Days_To_Recovery"], recovered["Max_Adverse_Pct"],
               alpha=0.2, s=10, color="crimson")
    ax.set_xlabel("Days to Recovery")
    ax.set_ylabel("Max Adverse Move % (further drop before recovery)")
    ax.set_title("How Deep Before Recovery?")
    ax.set_xlim(0, min(260, recovered["Days_To_Recovery"].quantile(0.95)))
    ax.set_ylim(0, recovered["Max_Adverse_Pct"].quantile(0.95))

    plt.suptitle("NIFTY 50 — Recovery After Bearish Breakdowns", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bearish_recovery.png", dpi=150)
    print("Saved bearish_recovery.png")


def plot_recovery_by_streak(bear_df):
    """How does initial streak length affect recovery time?"""
    recovered = bear_df.dropna(subset=["Days_To_Recovery"]).copy()
    max_s = min(6, recovered["Streak"].max())
    grouped = recovered[recovered["Streak"] <= max_s].groupby("Streak")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Mean recovery time by streak
    ax = axes[0]
    means = grouped["Days_To_Recovery"].mean()
    medians = grouped["Days_To_Recovery"].median()
    counts = grouped["Days_To_Recovery"].count()
    x = np.arange(len(means))
    ax.bar(x - 0.15, means, width=0.3, color="crimson", alpha=0.7, label="Mean")
    ax.bar(x + 0.15, medians, width=0.3, color="crimson", alpha=0.4, label="Median")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_xlabel("Initial Streak Length (days)")
    ax.set_ylabel("Days to Recovery")
    ax.set_title("Recovery Time by Streak Length")
    ax.legend()
    for i, c in enumerate(counts):
        ax.text(i, max(means.iloc[i], medians.iloc[i]) + 1, f"n={c}", ha="center", fontsize=8)

    # % not recovered by streak
    ax = axes[1]
    not_recovered_pct = []
    streak_vals = range(0, max_s + 1)
    for s in streak_vals:
        subset = bear_df[bear_df["Streak"] == s]
        if len(subset) > 0:
            nr = subset["Days_To_Recovery"].isna().sum() / len(subset) * 100
        else:
            nr = 0
        not_recovered_pct.append(nr)
    ax.bar(list(streak_vals), not_recovered_pct, color="darkred", alpha=0.7)
    ax.set_xlabel("Initial Streak Length (days)")
    ax.set_ylabel("% Not Recovered within 252 days")
    ax.set_title("Non-Recovery Rate by Streak Length")
    ax.set_xticks(list(streak_vals))

    plt.suptitle("NIFTY 50 — Recovery Dynamics by Bearish Streak Length", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "recovery_by_streak.png", dpi=150)
    print("Saved recovery_by_streak.png")


def save_recovery_csv(bear_df):
    """Save recovery data to CSV."""
    cols = ["Date", "Close", "Streak", "Breakout_Magnitude_Pct", "Peak_Excursion_Pct",
            "Days_To_Peak", "Velocity_Pct_Per_Day", "Days_To_Recovery", "Max_Adverse_Pct",
            "VIX"]
    bear_df[cols].to_csv(OUTPUT_DIR / "bearish_recovery_data.csv", index=False)
    print("Saved bearish_recovery_data.csv")


# ─────────────────────────────────────────────────────────────
# VIX and Breakout/Breakdown Analysis
# ─────────────────────────────────────────────────────────────

def vix_breakout_analysis(df, bull_df, bear_df):
    """Analyze the relationship between VIX and breakout/breakdown behaviour."""
    from scipy import stats as sp_stats

    # --- 1. VIX levels on breakout days vs normal days ---
    normal_mask = ~df["Bullish_Breakout"] & ~df["Bearish_Breakout"]
    vix_normal = df.loc[normal_mask, "VIX"]
    vix_bull = bull_df["VIX"]
    vix_bear = bear_df["VIX"]

    print(f"\n{'='*70}")
    print("  VIX LEVELS BY DAY TYPE")
    print(f"{'='*70}")
    print(f"  {'Day Type':<30} {'Mean':>8} {'Median':>8} {'N':>6}")
    print("  " + "-" * 52)
    print(f"  {'Normal (no breakout)':<30} {vix_normal.mean():>8.2f} {vix_normal.median():>8.2f} {len(vix_normal):>6}")
    print(f"  {'Bullish Breakout':<30} {vix_bull.mean():>8.2f} {vix_bull.median():>8.2f} {len(vix_bull):>6}")
    print(f"  {'Bearish Breakdown':<30} {vix_bear.mean():>8.2f} {vix_bear.median():>8.2f} {len(vix_bear):>6}")

    # --- 2. Does VIX predict continuation vs reversal? ---
    print(f"\n{'='*70}")
    print("  DOES VIX PREDICT CONTINUATION?")
    print(f"{'='*70}")

    for label, cont_df, direction in [
        ("Bullish", bull_df, "bull"),
        ("Bearish", bear_df, "bear"),
    ]:
        continued = cont_df[cont_df["Streak"] >= 1]
        reversed_ = cont_df[cont_df["Streak"] == 0]
        print(f"\n  {label} Breakouts:")
        print(f"    Continued (streak>=1): VIX mean={continued['VIX'].mean():.2f}, "
              f"median={continued['VIX'].median():.2f}, n={len(continued)}")
        print(f"    Reversed  (streak=0):  VIX mean={reversed_['VIX'].mean():.2f}, "
              f"median={reversed_['VIX'].median():.2f}, n={len(reversed_)}")

        # T-test
        t_stat, p_val = sp_stats.ttest_ind(continued["VIX"], reversed_["VIX"])
        print(f"    T-test: t={t_stat:.3f}, p={p_val:.4f} "
              f"{'(significant)' if p_val < 0.05 else '(not significant)'}")

    # --- 3. VIX change in the days after breakout ---
    print(f"\n{'='*70}")
    print("  VIX TRAJECTORY AFTER BREAKOUTS")
    print(f"{'='*70}")

    for label, cont_df, direction in [
        ("Bullish", bull_df, "bull"),
        ("Bearish", bear_df, "bear"),
    ]:
        print(f"\n  {label} — Mean VIX change from breakout day:")
        for n in [1, 2, 3, 5, 10]:
            col = f"VIX_Day{n}"
            vix_change = cont_df[col] - cont_df["VIX"]
            valid = vix_change.dropna()
            print(f"    Day {n:>2}: {valid.mean():+.2f} (median {valid.median():+.2f}), n={len(valid)}")

    # --- 4. Correlations: VIX vs streak, peak excursion, velocity ---
    print(f"\n{'='*70}")
    print("  VIX CORRELATIONS WITH BREAKOUT METRICS")
    print(f"{'='*70}")
    print(f"  {'Pair':<45} {'Bullish r':>10} {'Bearish r':>10}")
    print("  " + "-" * 65)

    metrics = [
        ("VIX vs Streak", "VIX", "Streak"),
        ("VIX vs Peak Excursion", "VIX", "Peak_Excursion_Pct"),
        ("VIX vs Velocity", "VIX", "Velocity_Pct_Per_Day"),
        ("VIX vs Breakout Magnitude", "VIX", "Breakout_Magnitude_Pct"),
    ]

    corr_results = []
    for label, col_a, col_b in metrics:
        r_bull = sp_stats.spearmanr(bull_df[col_a], bull_df[col_b])[0]
        r_bear = sp_stats.spearmanr(bear_df[col_a], bear_df[col_b])[0]
        print(f"  {label:<45} {r_bull:>10.3f} {r_bear:>10.3f}")
        corr_results.append({"Pair": label, "Bullish_Spearman": round(r_bull, 3),
                             "Bearish_Spearman": round(r_bear, 3)})

    # VIX vs recovery time for bearish
    if "Days_To_Recovery" in bear_df.columns:
        recovered = bear_df.dropna(subset=["Days_To_Recovery"])
        r_rec = sp_stats.spearmanr(recovered["VIX"], recovered["Days_To_Recovery"])[0]
        print(f"  {'VIX vs Recovery Time (bearish)':<45} {'--':>10} {r_rec:>10.3f}")
        corr_results.append({"Pair": "VIX vs Recovery Time (bearish)", "Bullish_Spearman": None,
                             "Bearish_Spearman": round(r_rec, 3)})

    pd.DataFrame(corr_results).to_csv(OUTPUT_DIR / "vix_breakout_correlations.csv", index=False)
    return vix_normal, vix_bull, vix_bear


def plot_vix_breakout(df, bull_df, bear_df, vix_normal, vix_bull, vix_bear):
    """Plot VIX relationships with breakouts."""
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # --- 1. VIX distribution by day type ---
    ax = axes[0, 0]
    bins = np.linspace(5, 60, 50)
    ax.hist(vix_normal, bins=bins, alpha=0.5, density=True, color="gray", label="Normal")
    ax.hist(vix_bull, bins=bins, alpha=0.5, density=True, color="forestgreen", label="Bullish Breakout")
    ax.hist(vix_bear, bins=bins, alpha=0.5, density=True, color="crimson", label="Bearish Breakdown")
    ax.set_xlabel("India VIX")
    ax.set_ylabel("Density")
    ax.set_title("VIX Distribution by Day Type")
    ax.legend(fontsize=8)

    # --- 2. VIX on breakout day: continued vs reversed ---
    ax = axes[0, 1]
    bull_cont = bull_df[bull_df["Streak"] >= 1]["VIX"]
    bull_rev = bull_df[bull_df["Streak"] == 0]["VIX"]
    bear_cont = bear_df[bear_df["Streak"] >= 1]["VIX"]
    bear_rev = bear_df[bear_df["Streak"] == 0]["VIX"]

    positions = [1, 2, 4, 5]
    bp = ax.boxplot([bull_cont, bull_rev, bear_cont, bear_rev],
                    positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False)
    colors = ["forestgreen", "lightgreen", "crimson", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Bull\nCont.", "Bull\nRev.", "Bear\nCont.", "Bear\nRev."], fontsize=9)
    ax.set_ylabel("VIX")
    ax.set_title("VIX: Continuation vs Reversal")

    # --- 3. VIX trajectory after breakouts (mean VIX change) ---
    ax = axes[0, 2]
    days_fwd = [0, 1, 2, 3, 5, 10]
    for cont_df, label, color in [
        (bull_df, "Bullish", "forestgreen"),
        (bear_df, "Bearish", "crimson"),
    ]:
        vix_means = [0]  # day 0 = 0 change
        for n in [1, 2, 3, 5, 10]:
            vix_change = (cont_df[f"VIX_Day{n}"] - cont_df["VIX"]).dropna()
            vix_means.append(vix_change.mean())
        ax.plot(days_fwd, vix_means, "o-", color=color, linewidth=2, markersize=6, label=label)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Days After Breakout")
    ax.set_ylabel("Mean VIX Change")
    ax.set_title("VIX Trajectory After Breakouts")
    ax.legend()
    ax.set_xticks(days_fwd)

    # --- 4. VIX quintiles vs continuation rate ---
    ax = axes[1, 0]
    for cont_df, label, color in [
        (bull_df, "Bullish", "forestgreen"),
        (bear_df, "Bearish", "crimson"),
    ]:
        cont_df_copy = cont_df.copy()
        cont_df_copy["VIX_Q"] = pd.qcut(cont_df_copy["VIX"], q=5, duplicates="drop")
        grouped = cont_df_copy.groupby("VIX_Q", observed=True).agg(
            cont_rate=("Streak", lambda x: (x >= 1).mean() * 100),
            count=("Streak", "count"),
        ).reset_index()
        x = range(len(grouped))
        ax.plot(list(x), grouped["cont_rate"].values, "o-", color=color, linewidth=2, markersize=6,
                label=f"{label} (n per bin ~{grouped['count'].mean():.0f})")
    ax.set_xlabel("VIX Quintile (low → high)")
    ax.set_ylabel("% That Continue (streak >= 1)")
    ax.set_title("Continuation Rate by VIX Level")
    ax.legend(fontsize=8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.5)

    # --- 5. VIX vs peak excursion scatter ---
    ax = axes[1, 1]
    ax.scatter(bull_df["VIX"], bull_df["Peak_Excursion_Pct"],
               alpha=0.15, s=8, color="forestgreen", label="Bullish")
    ax.scatter(bear_df["VIX"], bear_df["Peak_Excursion_Pct"],
               alpha=0.15, s=8, color="crimson", label="Bearish")
    # Regression lines
    for cont_df, color in [(bull_df, "forestgreen"), (bear_df, "crimson")]:
        slope, intercept, r, p, se = sp_stats.linregress(cont_df["VIX"], cont_df["Peak_Excursion_Pct"])
        x_line = np.linspace(cont_df["VIX"].min(), cont_df["VIX"].quantile(0.95), 100)
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=1.5,
                linestyle="--", alpha=0.8)
    ax.set_xlabel("VIX on Breakout Day")
    ax.set_ylabel("Peak Excursion %")
    ax.set_title("VIX vs Peak Excursion")
    ax.set_ylim(0, bull_df["Peak_Excursion_Pct"].quantile(0.90))
    ax.legend(fontsize=8)

    # --- 6. VIX vs recovery time (bearish only) ---
    ax = axes[1, 2]
    if "Days_To_Recovery" in bear_df.columns:
        recovered = bear_df.dropna(subset=["Days_To_Recovery"])
        # Bin VIX into quintiles and show mean/median recovery
        rec_copy = recovered.copy()
        rec_copy["VIX_Q"] = pd.qcut(rec_copy["VIX"], q=5, duplicates="drop")
        grouped = rec_copy.groupby("VIX_Q", observed=True).agg(
            mean_rec=("Days_To_Recovery", "mean"),
            median_rec=("Days_To_Recovery", "median"),
            count=("Days_To_Recovery", "count"),
            not_rec_overall=("Days_To_Recovery", "count"),  # placeholder
        ).reset_index()

        # Also compute non-recovery rate per VIX quintile from full bear_df
        bear_copy = bear_df.copy()
        bear_copy["VIX_Q"] = pd.qcut(bear_copy["VIX"], q=5, duplicates="drop")
        nr_rates = bear_copy.groupby("VIX_Q", observed=True).apply(
            lambda g: g["Days_To_Recovery"].isna().mean() * 100,
        ).values

        x = np.arange(len(grouped))
        ax.bar(x - 0.15, grouped["mean_rec"], width=0.3, color="crimson", alpha=0.7, label="Mean")
        ax.bar(x + 0.15, grouped["median_rec"], width=0.3, color="crimson", alpha=0.4, label="Median")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(grouped))], fontsize=9)
        ax.set_xlabel("VIX Quintile (Q1=low, Q5=high)")
        ax.set_ylabel("Days to Recovery")
        ax.set_title("Bearish Recovery Time by VIX Level")
        ax.legend(fontsize=8)

        for i, (m, c, nr) in enumerate(zip(grouped["mean_rec"], grouped["count"], nr_rates)):
            ax.text(i, m + 1, f"n={c}\n{nr:.0f}% nr", ha="center", fontsize=7)

    plt.suptitle("NIFTY 50 — India VIX and Breakout/Breakdown Dynamics", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "vix_breakout_analysis.png", dpi=150)
    print("Saved vix_breakout_analysis.png")


def save_summary(bull_df, bear_df):
    """Save summary CSV."""
    periods = [1, 2, 3, 5, 10, 20]
    rows = []
    for label, cont_df, direction in [
        ("Bullish", bull_df, "bull"),
        ("Bearish", bear_df, "bear"),
    ]:
        sign = 1 if direction == "bull" else -1
        row = {
            "Type": label,
            "Count": len(cont_df),
            "Mean_Streak": round(cont_df["Streak"].mean(), 2),
            "Median_Streak": round(cont_df["Streak"].median(), 1),
            "Max_Streak": cont_df["Streak"].max(),
            "Mean_Breakout_Mag_Pct": round(cont_df["Breakout_Magnitude_Pct"].mean(), 3),
            "Median_Peak_Excursion_Pct": round(cont_df["Peak_Excursion_Pct"].median(), 3),
            "Mean_Peak_Excursion_Pct": round(cont_df["Peak_Excursion_Pct"].mean(), 3),
            "Median_Days_To_Peak": round(cont_df["Days_To_Peak"].median(), 1),
            "Mean_Velocity_Pct_Per_Day": round(cont_df["Velocity_Pct_Per_Day"].mean(), 3),
            "Median_Velocity_Pct_Per_Day": round(cont_df["Velocity_Pct_Per_Day"].median(), 3),
        }
        for n in periods:
            col = f"Fwd_{n}d_Ret"
            row[f"Fwd_{n}d_Mean"] = round(cont_df[col].mean(), 3)
            row[f"Fwd_{n}d_WinRate"] = round((cont_df[col] * sign > 0).mean() * 100, 1)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "continuation_summary.csv", index=False)
    print("Saved continuation_summary.csv")


def main():
    df = load_data()
    df = identify_breakouts(df)

    bull_count = df["Bullish_Breakout"].sum()
    bear_count = df["Bearish_Breakout"].sum()
    total = len(df)
    print(f"Total trading days: {total}")
    print(f"Bullish breakouts (Close > Prev High): {bull_count} ({bull_count/total*100:.1f}%)")
    print(f"Bearish breakouts (Close < Prev Low):  {bear_count} ({bear_count/total*100:.1f}%)")

    bull_df = measure_continuation(df, "Bullish_Breakout", "bull")
    bear_df = measure_continuation(df, "Bearish_Breakout", "bear")

    print_stats("BULLISH BREAKOUT CONTINUATION (Close > Prev High)", bull_df, "bull")
    print_stats("BEARISH BREAKOUT CONTINUATION (Close < Prev Low)", bear_df, "bear")

    plot_streak_distribution(bull_df, bear_df)
    plot_forward_returns(bull_df, bear_df)
    plot_win_rates(bull_df, bear_df)
    plot_breakout_timeline(df, bull_df, bear_df)
    plot_streak_vs_return(bull_df, bear_df)
    plot_velocity_distribution(bull_df, bear_df)
    plot_magnitude_vs_continuation(bull_df, bear_df)
    plot_velocity_by_streak(bull_df, bear_df)
    print_velocity_summary(bull_df, bear_df)

    # Recovery analysis
    bear_df = measure_recovery(df, bear_df, "bear")
    print_recovery_stats(bear_df, "BEARISH BREAKDOWN RECOVERY (days to return to breakdown level)")
    plot_recovery_distribution(bear_df)
    plot_recovery_by_streak(bear_df)
    save_recovery_csv(bear_df)

    # VIX and breakout analysis
    vix_normal, vix_bull, vix_bear = vix_breakout_analysis(df, bull_df, bear_df)
    plot_vix_breakout(df, bull_df, bear_df, vix_normal, vix_bull, vix_bear)

    save_summary(bull_df, bear_df)

    print("\nDone! All outputs in analysis/")


if __name__ == "__main__":
    main()
