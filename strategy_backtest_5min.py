"""
Backtest breakout/breakdown strategies on NIFTY 50 using 5-minute intraday data.

Compares two entry modes:
1. EOD: Enter at day close when Close > Prev Day High or Close < Prev Day Low
2. Intraday: Enter at first 5-min candle open that breaches PDH/PDL

Both modes tested with futures and ITM options (Black-Scholes pricing with VIX).
Optuna optimization for stop loss, trailing SL, holding period, and targets.

Data: 5-minute NIFTY 50 candles (2015-2026) from Kaggle.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from pathlib import Path
from dataclasses import dataclass
import optuna
import warnings
import json

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

LOT_SIZE_CHANGES = [
    (pd.Timestamp("2007-01-01"), 50),
    (pd.Timestamp("2015-10-29"), 75),
    (pd.Timestamp("2024-11-25"), 25),
]


def get_lot_size(date):
    lot = 50
    for change_date, size in LOT_SIZE_CHANGES:
        if date >= change_date:
            lot = size
    return lot


# ─────────────────────────────────────────────────────────────
# Black-Scholes pricing
# ─────────────────────────────────────────────────────────────

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1


def find_itm_strike(spot, vix, option_type, target_delta=0.6, r=0.06, T=7/365):
    sigma = vix / 100
    if option_type == "call":
        lo, hi = spot * 0.85, spot * 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            d = bs_delta(spot, mid, T, r, sigma, "call")
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        strike = (lo + hi) / 2
        premium = bs_call_price(spot, strike, T, r, sigma)
    else:
        lo, hi = spot * 1.0, spot * 1.15
        for _ in range(50):
            mid = (lo + hi) / 2
            d = abs(bs_delta(spot, mid, T, r, sigma, "put"))
            if d > target_delta:
                hi = mid
            else:
                lo = mid
        strike = (lo + hi) / 2
        premium = bs_put_price(spot, strike, T, r, sigma)
    delta = bs_delta(spot, strike, T, r, sigma, option_type)
    return strike, premium, delta


# ─────────────────────────────────────────────────────────────
# Trade dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    pnl_points: float = 0.0
    pnl_rupees: float = 0.0
    lot_size: int = 75
    exit_reason: str = ""
    entry_mode: str = ""
    option_type: str = ""
    strike: float = 0.0
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    delta: float = 0.0
    option_pnl_points: float = 0.0
    option_pnl_rupees: float = 0.0


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_data():
    """Load 5-min data, build daily OHLC, merge VIX, compute all signals."""
    print("Loading 5-minute data...")
    raw = pd.read_csv("data/NIFTY 50_5minute.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["trading_date"] = raw["date"].dt.normalize()

    # Build daily OHLC from 5-min candles
    daily = raw.groupby("trading_date").agg(
        Open=("open", "first"),
        High=("high", "max"),
        Low=("low", "min"),
        Close=("close", "last"),
    )
    daily.index.name = None
    daily.sort_index(inplace=True)
    print(f"  Daily bars: {len(daily)} days ({daily.index.min().date()} to {daily.index.max().date()})")

    # Merge VIX (daily)
    vix = pd.read_csv("data/indiavix.csv", index_col=0, parse_dates=True)
    daily["VIX"] = vix["Close"].reindex(daily.index).ffill()

    # Previous day levels
    daily["Prev_High"] = daily["High"].shift(1)
    daily["Prev_Low"] = daily["Low"].shift(1)
    daily["Prev_Close"] = daily["Close"].shift(1)

    # EOD breakout signals
    daily["Bullish_Breakout"] = daily["Close"] > daily["Prev_High"]
    daily["Bearish_Breakout"] = daily["Close"] < daily["Prev_Low"]

    # Intraday entry signals from 5-min candles
    print("Computing intraday breach signals from 5-min candles...")
    daily = _compute_intraday_signals(raw, daily)

    daily = daily.dropna(subset=["Prev_High", "Prev_Low", "VIX"])
    print(f"  Usable days: {len(daily)}")

    bull_eod = daily["Bullish_Breakout"].sum()
    bear_eod = daily["Bearish_Breakout"].sum()
    bull_intra = daily["Intra_Bull_Signal"].sum()
    bear_intra = daily["Intra_Bear_Signal"].sum()
    print(f"  EOD signals:      {bull_eod} bullish, {bear_eod} bearish")
    print(f"  Intraday signals: {bull_intra} bullish, {bear_intra} bearish")

    return daily


def _compute_intraday_signals(raw, daily):
    """Find first 5-min candle open that breaches PDH/PDL for each day.

    Adds to daily: Intra_{Bull,Bear}_{Signal,Entry,PostHigh,PostLow,DayClose,Time}
    """
    for prefix in ["Intra_Bull", "Intra_Bear"]:
        daily[f"{prefix}_Signal"] = False
        for suffix in ["_Entry", "_PostHigh", "_PostLow", "_DayClose"]:
            daily[f"{prefix}{suffix}"] = np.nan
        daily[f"{prefix}_Time"] = pd.NaT

    for date, group in raw.groupby("trading_date"):
        if date not in daily.index:
            continue
        pdh = daily.at[date, "Prev_High"]
        pdl = daily.at[date, "Prev_Low"]
        if pd.isna(pdh) or pd.isna(pdl):
            continue

        group = group.sort_values("date")
        day_close = group["close"].iloc[-1]

        # Bullish: first candle with open > PDH
        bull_mask = group["open"] > pdh
        if bull_mask.any():
            first_idx = bull_mask.idxmax()
            post = group.loc[first_idx:]
            daily.at[date, "Intra_Bull_Signal"] = True
            daily.at[date, "Intra_Bull_Entry"] = group.at[first_idx, "open"]
            daily.at[date, "Intra_Bull_PostHigh"] = post["high"].max()
            daily.at[date, "Intra_Bull_PostLow"] = post["low"].min()
            daily.at[date, "Intra_Bull_DayClose"] = day_close
            daily.at[date, "Intra_Bull_Time"] = group.at[first_idx, "date"]

        # Bearish: first candle with open < PDL
        bear_mask = group["open"] < pdl
        if bear_mask.any():
            first_idx = bear_mask.idxmax()
            post = group.loc[first_idx:]
            daily.at[date, "Intra_Bear_Signal"] = True
            daily.at[date, "Intra_Bear_Entry"] = group.at[first_idx, "open"]
            daily.at[date, "Intra_Bear_PostHigh"] = post["high"].max()
            daily.at[date, "Intra_Bear_PostLow"] = post["low"].min()
            daily.at[date, "Intra_Bear_DayClose"] = day_close
            daily.at[date, "Intra_Bear_Time"] = group.at[first_idx, "date"]

    return daily


# ─────────────────────────────────────────────────────────────
# Unified backtest
# ─────────────────────────────────────────────────────────────

def run_backtest(df, entry_mode="eod", instrument="futures",
                 sl_pct=1.0, trailing_sl_pct=None, max_hold_days=5,
                 target_pct=None, trade_type="both", target_delta=0.6):
    """
    Backtest breakout strategy.

    entry_mode: "eod" (enter at close on breakout day) or
                "intraday" (enter at first 5-min candle open breaching PDH/PDL)
    instrument: "futures" or "options"
    """
    trades = []
    r = 0.06
    T_entry = 7 / 365

    i = 0
    while i < len(df):
        row = df.iloc[i]
        date = df.index[i]

        # ── Determine signal ──
        if entry_mode == "eod":
            is_bull = bool(row["Bullish_Breakout"])
            is_bear = bool(row["Bearish_Breakout"])
        else:
            is_bull = bool(row["Intra_Bull_Signal"])
            is_bear = bool(row["Intra_Bear_Signal"])

        if not is_bull and not is_bear:
            i += 1
            continue

        # Both signals on same day: take the one that triggered first
        if is_bull and is_bear:
            if entry_mode == "intraday":
                bull_time = row["Intra_Bull_Time"]
                bear_time = row["Intra_Bear_Time"]
                if bull_time <= bear_time:
                    is_bear = False
                else:
                    is_bull = False
            else:
                # EOD: can't have both (Close > PrevHigh AND Close < PrevLow is impossible)
                i += 1
                continue

        if is_bull and trade_type == "short":
            i += 1
            continue
        if is_bear and trade_type == "long":
            i += 1
            continue

        direction = "long" if is_bull else "short"

        # ── Entry price ──
        if entry_mode == "intraday":
            prefix = "Intra_Bull" if is_bull else "Intra_Bear"
            entry_price = row[f"{prefix}_Entry"]
        else:
            entry_price = row["Close"]

        if pd.isna(entry_price):
            i += 1
            continue

        lot_size = get_lot_size(date)
        entry_date = date

        # ── Initial stop loss ──
        if direction == "long":
            sl = entry_price * (1 - sl_pct / 100)
            peak = entry_price
        else:
            sl = entry_price * (1 + sl_pct / 100)
            trough = entry_price

        exit_price = None
        exit_date = None
        exit_reason = ""
        days_held = 0

        # For intraday entry, check entry day itself (j=0) using post-entry stats
        start_j = 0 if entry_mode == "intraday" else 1

        for j in range(start_j, min(max_hold_days + 1, len(df) - i)):
            if j == 0 and entry_mode == "intraday":
                # Entry day: use post-entry high/low (only candles after entry)
                prefix = "Intra_Bull" if is_bull else "Intra_Bear"
                day_high = row[f"{prefix}_PostHigh"]
                day_low = row[f"{prefix}_PostLow"]
            else:
                day = df.iloc[i + j]
                day_high = day["High"]
                day_low = day["Low"]

            if direction == "long":
                if day_low <= sl:
                    exit_price = sl
                    exit_date = df.index[i + j]
                    exit_reason = "stop_loss"
                    days_held = j
                    break
                if day_high > peak:
                    peak = day_high
                    if trailing_sl_pct is not None:
                        sl = max(sl, peak * (1 - trailing_sl_pct / 100))
                if target_pct is not None and day_high >= entry_price * (1 + target_pct / 100):
                    exit_price = entry_price * (1 + target_pct / 100)
                    exit_date = df.index[i + j]
                    exit_reason = "target"
                    days_held = j
                    break
            else:
                if day_high >= sl:
                    exit_price = sl
                    exit_date = df.index[i + j]
                    exit_reason = "stop_loss"
                    days_held = j
                    break
                if day_low < trough:
                    trough = day_low
                    if trailing_sl_pct is not None:
                        sl = min(sl, trough * (1 + trailing_sl_pct / 100))
                if target_pct is not None and day_low <= entry_price * (1 - target_pct / 100):
                    exit_price = entry_price * (1 - target_pct / 100)
                    exit_date = df.index[i + j]
                    exit_reason = "target"
                    days_held = j
                    break

        # Time exit
        if exit_price is None:
            last_j = min(max_hold_days, len(df) - i - 1)
            if last_j >= (1 if entry_mode == "eod" else 0):
                exit_price = df.iloc[i + last_j]["Close"]
                exit_date = df.index[i + last_j]
                exit_reason = "time_exit"
                days_held = last_j
            else:
                i += 1
                continue

        # ── P&L ──
        if direction == "long":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        trade = Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            pnl_points=pnl_points,
            pnl_rupees=pnl_points * lot_size,
            lot_size=lot_size,
            exit_reason=exit_reason,
            entry_mode=entry_mode,
        )

        # ── Options pricing ──
        if instrument == "options":
            vix = row["VIX"]
            sigma = vix / 100
            opt_type = "call" if direction == "long" else "put"
            strike, entry_premium, delta = find_itm_strike(
                entry_price, vix, opt_type, target_delta, r, T_entry)
            T_exit = max(T_entry - days_held / 365, 1 / 365)
            if opt_type == "call":
                exit_premium = bs_call_price(exit_price, strike, T_exit, r, sigma)
            else:
                exit_premium = bs_put_price(exit_price, strike, T_exit, r, sigma)

            trade.option_type = opt_type
            trade.strike = round(strike, 2)
            trade.entry_premium = round(entry_premium, 2)
            trade.exit_premium = round(exit_premium, 2)
            trade.delta = round(delta, 3)
            trade.option_pnl_points = round(exit_premium - entry_premium, 2)
            trade.option_pnl_rupees = round((exit_premium - entry_premium) * lot_size, 2)

        trades.append(trade)

        # Skip to after exit
        exit_loc = df.index.get_loc(exit_date)
        i = exit_loc + 1

    return trades


# ─────────────────────────────────────────────────────────────
# Trade analysis
# ─────────────────────────────────────────────────────────────

def analyze_trades(trades, label="", pnl_field="pnl_rupees", pts_field="pnl_points"):
    if not trades:
        print(f"  {label}: No trades")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    total = len(df)
    winners = (df[pts_field] > 0).sum()
    losers = (df[pts_field] < 0).sum()
    win_rate = winners / total * 100

    total_pnl = df[pnl_field].sum()
    avg_win = df.loc[df[pts_field] > 0, pnl_field].mean() if winners > 0 else 0
    avg_loss = df.loc[df[pts_field] < 0, pnl_field].mean() if losers > 0 else 0
    gross_loss = abs(df.loc[df[pts_field] < 0, pnl_field].sum())
    profit_factor = df.loc[df[pts_field] > 0, pnl_field].sum() / gross_loss if gross_loss > 0 else float("inf")

    cum_pnl = df[pnl_field].cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    by_reason = df.groupby("exit_reason")[pnl_field].agg(["count", "sum", "mean"])
    by_dir = df.groupby("direction")[pnl_field].agg(["count", "sum", "mean"])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total trades: {total}")
    print(f"  Winners: {winners} ({win_rate:.1f}%)")
    print(f"  Losers: {losers} ({100-win_rate:.1f}%)")
    print(f"  Total P&L: Rs {total_pnl:,.0f}")
    print(f"  Avg Win: Rs {avg_win:,.0f}")
    print(f"  Avg Loss: Rs {avg_loss:,.0f}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Max Drawdown: Rs {max_dd:,.0f}")
    print(f"\n  By Exit Reason:")
    print(f"  {by_reason.to_string()}")
    print(f"\n  By Direction:")
    print(f"  {by_dir.to_string()}")

    return {
        "total_trades": total, "win_rate": win_rate,
        "total_pnl": total_pnl, "profit_factor": profit_factor,
        "max_drawdown": max_dd, "avg_win": avg_win, "avg_loss": avg_loss,
    }


def trades_to_df(trades):
    records = []
    for t in trades:
        records.append({
            "Entry Date": t.entry_date.strftime("%Y-%m-%d"),
            "Exit Date": t.exit_date.strftime("%Y-%m-%d"),
            "Direction": t.direction,
            "Entry Mode": t.entry_mode,
            "Entry": round(t.entry_price, 2),
            "Exit": round(t.exit_price, 2),
            "SL": round(t.stop_loss, 2),
            "P&L (pts)": round(t.pnl_points, 2),
            "P&L (Rs)": round(t.pnl_rupees, 2),
            "Lot": t.lot_size,
            "Exit Reason": t.exit_reason,
            "Option": t.option_type,
            "Strike": t.strike,
            "Entry Prem": t.entry_premium,
            "Exit Prem": t.exit_premium,
            "Delta": t.delta,
            "Opt P&L (pts)": t.option_pnl_points,
            "Opt P&L (Rs)": t.option_pnl_rupees,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# Optuna optimization
# ─────────────────────────────────────────────────────────────

def run_optimization(df, entry_mode, instrument, n_trials=300, trade_type="both"):
    def objective(trial):
        sl_pct = trial.suggest_float("sl_pct", 0.3, 3.0, step=0.1)
        use_trailing = trial.suggest_categorical("use_trailing", [True, False])
        trailing_sl_pct = trial.suggest_float("trailing_sl_pct", 0.3, 3.0, step=0.1) if use_trailing else None
        if instrument == "options":
            max_hold = trial.suggest_int("max_hold_days", 1, 10)
        else:
            max_hold = trial.suggest_int("max_hold_days", 1, 20)
        use_target = trial.suggest_categorical("use_target", [True, False])
        target_pct = trial.suggest_float("target_pct", 0.5, 5.0, step=0.1) if use_target else None

        kwargs = dict(
            entry_mode=entry_mode, instrument=instrument,
            sl_pct=sl_pct, trailing_sl_pct=trailing_sl_pct,
            max_hold_days=max_hold, target_pct=target_pct, trade_type=trade_type,
        )
        if instrument == "options":
            kwargs["target_delta"] = trial.suggest_float("target_delta", 0.55, 0.75, step=0.05)

        trades = run_backtest(df, **kwargs)
        if len(trades) < 30:
            return -1e9

        if instrument == "options":
            pnls = [t.option_pnl_rupees for t in trades]
            pts = [t.option_pnl_points for t in trades]
        else:
            pnls = [t.pnl_rupees for t in trades]
            pts = [t.pnl_points for t in trades]

        total_pnl = sum(pnls)
        win_rate = sum(1 for p in pts if p > 0) / len(pts)
        cum = np.cumsum(pnls)
        max_dd = (cum - np.maximum.accumulate(cum)).min()

        if max_dd == 0:
            return total_pnl
        calmar = total_pnl / abs(max_dd) if max_dd < 0 else total_pnl
        return calmar * np.sqrt(win_rate)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_equity_comparison(results_list, filename):
    """Plot multiple equity curves in subplots."""
    n = len(results_list)
    fig, axes = plt.subplots(n, 1, figsize=(20, 5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, trades, pnl_field) in zip(axes, results_list):
        if not trades:
            continue
        dates = [t.exit_date for t in trades]
        pnls = [getattr(t, pnl_field) for t in trades]
        cum_pnl = np.cumsum(pnls)

        ax.plot(dates, cum_pnl, linewidth=1.5, color="steelblue")
        ax.fill_between(dates, cum_pnl, 0,
                        where=np.array(cum_pnl) >= 0, color="forestgreen", alpha=0.3)
        ax.fill_between(dates, cum_pnl, 0,
                        where=np.array(cum_pnl) < 0, color="crimson", alpha=0.3)
        ax.set_ylabel("Cumulative P&L (Rs)")
        ax.set_title(label)
        ax.axhline(0, color="black", linewidth=0.5)
        if len(cum_pnl) > 0:
            ax.annotate(f"Rs {cum_pnl[-1]:,.0f}", xy=(dates[-1], cum_pnl[-1]),
                        fontsize=10, fontweight="bold",
                        color="forestgreen" if cum_pnl[-1] > 0 else "crimson")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {filename}.png")


def plot_optimization_results(study, label, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    trials = [t for t in study.trials if t.value is not None and t.value > -1e8]
    values = [t.value for t in trials]
    ax.scatter(range(len(values)), values, alpha=0.3, s=10, color="steelblue")
    running_best = np.maximum.accumulate(values)
    ax.plot(running_best, color="crimson", linewidth=1.5, label="Running best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective (Calmar * sqrt(WR))")
    ax.set_title(f"{label} — Optimization History")
    ax.legend()

    ax = axes[1]
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())[:8]
        imp_vals = [importances[p] for p in params]
        ax.barh(params, imp_vals, color="steelblue", alpha=0.7)
        ax.set_xlabel("Importance")
        ax.set_title(f"{label} — Parameter Importance")
    except Exception:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {filename}.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def _summary_row(label, trades, pnl_field, pts_field):
    """Compute summary stats for one strategy."""
    pnls = [getattr(t, pnl_field) for t in trades]
    pts = [getattr(t, pts_field) for t in trades]
    n = len(trades)
    total_pnl = sum(pnls)
    wr = sum(1 for p in pts if p > 0) / n * 100
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gw / gl if gl > 0 else float("inf")
    cum = np.cumsum(pnls)
    max_dd = (cum - np.maximum.accumulate(cum)).min()
    return f"  {label:<42} {total_pnl:>12,.0f} {wr:>6.1f}% {pf:>6.2f} {max_dd:>12,.0f} {n:>7}"


def main():
    df = load_data()

    all_results = {}

    for entry_mode in ["eod", "intraday"]:
        for instrument in ["futures", "options"]:
            key = f"{instrument}_{entry_mode}"
            pnl_field = "option_pnl_rupees" if instrument == "options" else "pnl_rupees"
            pts_field = "option_pnl_points" if instrument == "options" else "pnl_points"

            # ── Baseline ──
            base_trades = run_backtest(
                df, entry_mode=entry_mode, instrument=instrument,
                sl_pct=1.0, max_hold_days=5,
            )
            base_label = f"{instrument.title()} {entry_mode.upper()} Baseline (SL=1%, 5d)"
            analyze_trades(base_trades, base_label, pnl_field, pts_field)

            # ── Optimize ──
            print(f"\nOptimizing {instrument} {entry_mode.upper()} (300 trials)...")
            study = run_optimization(df, entry_mode, instrument, n_trials=300)
            best = study.best_params
            print(f"  Best: {json.dumps(best, indent=2, default=str)}")

            # ── Run optimized ──
            opt_trades = run_backtest(
                df, entry_mode=entry_mode, instrument=instrument,
                sl_pct=best["sl_pct"],
                trailing_sl_pct=best.get("trailing_sl_pct"),
                max_hold_days=best["max_hold_days"],
                target_pct=best.get("target_pct"),
                target_delta=best.get("target_delta", 0.6),
            )
            opt_label = f"{instrument.title()} {entry_mode.upper()} Optimized"
            stats = analyze_trades(opt_trades, opt_label, pnl_field, pts_field)

            all_results[key] = {
                "baseline": base_trades,
                "optimized": opt_trades,
                "best_params": best,
                "study": study,
                "stats": stats,
            }

            # Save trades
            trades_to_df(base_trades).to_csv(
                OUTPUT_DIR / f"trades_5min_{key}_baseline.csv", index=False)
            trades_to_df(opt_trades).to_csv(
                OUTPUT_DIR / f"trades_5min_{key}_optimized.csv", index=False)

            # Optuna plots
            plot_optimization_results(
                study, f"{instrument.title()} {entry_mode.upper()}",
                f"optuna_5min_{key}")

    # ── EOD vs Intraday comparison charts ──
    for instrument in ["futures", "options"]:
        pnl_field = "option_pnl_rupees" if instrument == "options" else "pnl_rupees"
        eod_trades = all_results[f"{instrument}_eod"]["optimized"]
        intra_trades = all_results[f"{instrument}_intraday"]["optimized"]
        plot_equity_comparison([
            (f"{instrument.title()} EOD Entry (Optimized)", eod_trades, pnl_field),
            (f"{instrument.title()} Intraday Entry (Optimized)", intra_trades, pnl_field),
        ], f"eod_vs_intraday_{instrument}")

    # All 4 optimized strategies
    plot_equity_comparison([
        ("Futures EOD (Optimized)", all_results["futures_eod"]["optimized"], "pnl_rupees"),
        ("Futures Intraday (Optimized)", all_results["futures_intraday"]["optimized"], "pnl_rupees"),
        ("Options EOD (Optimized)", all_results["options_eod"]["optimized"], "option_pnl_rupees"),
        ("Options Intraday (Optimized)", all_results["options_intraday"]["optimized"], "option_pnl_rupees"),
    ], "equity_curves_5min_all_optimized")

    # ── Summary comparison table ──
    print("\n" + "=" * 92)
    print("  STRATEGY COMPARISON — 5-minute data (2015-2026)")
    print("=" * 92)
    print(f"\n  {'Strategy':<42} {'P&L':>12} {'Win%':>7} {'PF':>6} {'MaxDD':>12} {'Trades':>7}")
    print("  " + "-" * 86)

    for key in ["futures_eod", "futures_intraday", "options_eod", "options_intraday"]:
        data = all_results[key]
        instrument, entry_mode = key.split("_")
        pnl_f = "option_pnl_rupees" if instrument == "options" else "pnl_rupees"
        pts_f = "option_pnl_points" if instrument == "options" else "pnl_points"

        for variant, trades in [("Baseline", data["baseline"]), ("Optimized", data["optimized"])]:
            if not trades:
                continue
            label = f"{instrument.title()} {entry_mode.upper()} {variant}"
            print(_summary_row(label, trades, pnl_f, pts_f))
        print("  " + "-" * 86)

    # ── Save best params ──
    best_params = {k: v["best_params"] for k, v in all_results.items()}
    with open(OUTPUT_DIR / "best_params_5min.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\nBest parameters saved to analysis/best_params_5min.json")
    print("Done! All 5-min outputs in analysis/")


if __name__ == "__main__":
    main()
