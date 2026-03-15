"""
Breakout continuation analysis for NIFTY 50.

When the market closes below the previous day's low (bearish breakout)
or above the previous day's high (bullish breakout), does it continue
in that direction? And if so, for how long?
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
    nifty["Prev_High"] = nifty["High"].shift(1)
    nifty["Prev_Low"] = nifty["Low"].shift(1)
    nifty["Prev_Close"] = nifty["Close"].shift(1)
    nifty = nifty.dropna()
    return nifty


def identify_breakouts(df):
    """Tag days where close breaks previous day's range."""
    df["Bullish_Breakout"] = df["Close"] > df["Prev_High"]
    df["Bearish_Breakout"] = df["Close"] < df["Prev_Low"]
    return df


def measure_continuation(df, breakout_col, direction):
    """
    For each breakout day, look forward and count how many consecutive days
    continue in the breakout direction.

    For bullish: continuation = next day closes above current day's close
    For bearish: continuation = next day closes below current day's close

    Also track cumulative return over the continuation streak.
    """
    breakout_days = df.index[df[breakout_col]]
    results = []

    for day in breakout_days:
        loc = df.index.get_loc(day)
        streak = 0
        cum_return = 0.0
        base_close = df.iloc[loc]["Close"]
        prev_close = base_close

        for fwd in range(1, min(61, len(df) - loc)):  # look up to 60 days ahead
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

        # Also measure fixed forward returns regardless of streak
        fwd_returns = {}
        for n in [1, 2, 3, 5, 10, 20]:
            if loc + n < len(df):
                fwd_close = df.iloc[loc + n]["Close"]
                fwd_returns[f"Fwd_{n}d_Ret"] = (fwd_close - base_close) / base_close * 100
            else:
                fwd_returns[f"Fwd_{n}d_Ret"] = np.nan

        total_continuation_return = (prev_close - base_close) / base_close * 100

        results.append({
            "Date": day,
            "Close": base_close,
            "Streak": streak,
            "Continuation_Return_Pct": total_continuation_return,
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
    save_summary(bull_df, bear_df)

    print("\nDone! All outputs in analysis/")


if __name__ == "__main__":
    main()
