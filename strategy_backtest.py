"""
Backtest breakout/breakdown strategies on NIFTY 50.

1. Futures: long on bullish breakout, short on bearish breakdown
2. Options: buy ITM ~0.6 delta call/put (Black-Scholes pricing with VIX as IV)
3. Optuna optimization for stop loss / trailing SL / holding period
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import optuna
import warnings
import json

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# NIFTY lot sizes over time
# ~2007-2015: 50, 2015-2024: 75, 2024+: 25
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
# Black-Scholes for options pricing
# ─────────────────────────────────────────────────────────────

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def find_itm_strike(spot, vix, option_type, target_delta=0.6, r=0.06, T=7/365):
    """Find the strike price that gives approximately target_delta."""
    sigma = vix / 100  # VIX is annualized vol in %
    # Binary search for strike
    if option_type == "call":
        # ITM call: strike < spot
        lo, hi = spot * 0.85, spot * 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            d = bs_delta(spot, mid, T, r, sigma, "call")
            if d > target_delta:
                lo = mid  # need higher strike to reduce delta
            else:
                hi = mid
        strike = (lo + hi) / 2
        premium = bs_call_price(spot, strike, T, r, sigma)
    else:
        # ITM put: strike > spot
        lo, hi = spot * 1.0, spot * 1.15
        for _ in range(50):
            mid = (lo + hi) / 2
            d = abs(bs_delta(spot, mid, T, r, sigma, "put"))
            if d > target_delta:
                hi = mid  # need lower strike to reduce delta
            else:
                lo = mid
        strike = (lo + hi) / 2
        premium = bs_put_price(spot, strike, T, r, sigma)

    delta = bs_delta(spot, strike, T, r, sigma, option_type)
    return strike, premium, delta


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_data():
    nifty = pd.read_csv("data/nifty50.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv("data/indiavix.csv", index_col=0, parse_dates=True)
    nifty["Prev_High"] = nifty["High"].shift(1)
    nifty["Prev_Low"] = nifty["Low"].shift(1)
    nifty["Prev_Close"] = nifty["Close"].shift(1)
    nifty["VIX"] = vix["Close"].reindex(nifty.index)
    nifty = nifty.dropna()
    nifty["Bullish_Breakout"] = nifty["Close"] > nifty["Prev_High"]
    nifty["Bearish_Breakout"] = nifty["Close"] < nifty["Prev_Low"]
    return nifty


# ─────────────────────────────────────────────────────────────
# Futures strategy
# ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    stop_loss: float
    pnl_points: float = 0.0
    pnl_rupees: float = 0.0
    lot_size: int = 75
    exit_reason: str = ""
    # Options fields
    option_type: str = ""
    strike: float = 0.0
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    delta: float = 0.0
    option_pnl_points: float = 0.0
    option_pnl_rupees: float = 0.0


def run_futures_backtest(df, sl_pct=1.0, trailing_sl_pct=None, max_hold_days=5,
                          target_pct=None, trade_type="both"):
    """
    Backtest futures strategy.

    Parameters:
    - sl_pct: initial stop loss as % of entry price
    - trailing_sl_pct: trailing stop loss as % from peak (None = no trailing)
    - max_hold_days: maximum holding period
    - target_pct: profit target as % (None = no target)
    - trade_type: "both", "long", "short"
    """
    trades = []
    i = 0
    while i < len(df):
        row = df.iloc[i]
        is_bull = row["Bullish_Breakout"]
        is_bear = row["Bearish_Breakout"]

        if not is_bull and not is_bear:
            i += 1
            continue

        if is_bull and trade_type == "short":
            i += 1
            continue
        if is_bear and trade_type == "long":
            i += 1
            continue

        direction = "long" if is_bull else "short"
        entry_price = row["Close"]
        entry_date = df.index[i]
        lot_size = get_lot_size(entry_date)

        if direction == "long":
            sl = entry_price * (1 - sl_pct / 100)
            peak = entry_price
        else:
            sl = entry_price * (1 + sl_pct / 100)
            trough = entry_price

        exit_price = None
        exit_date = None
        exit_reason = ""

        for j in range(1, min(max_hold_days + 1, len(df) - i)):
            day = df.iloc[i + j]

            if direction == "long":
                # Check stop loss (intraday)
                if day["Low"] <= sl:
                    exit_price = sl
                    exit_date = df.index[i + j]
                    exit_reason = "stop_loss"
                    break

                # Update trailing SL
                if day["High"] > peak:
                    peak = day["High"]
                    if trailing_sl_pct is not None:
                        new_sl = peak * (1 - trailing_sl_pct / 100)
                        sl = max(sl, new_sl)

                # Check target
                if target_pct is not None and day["High"] >= entry_price * (1 + target_pct / 100):
                    exit_price = entry_price * (1 + target_pct / 100)
                    exit_date = df.index[i + j]
                    exit_reason = "target"
                    break

            else:  # short
                # Check stop loss (intraday)
                if day["High"] >= sl:
                    exit_price = sl
                    exit_date = df.index[i + j]
                    exit_reason = "stop_loss"
                    break

                # Update trailing SL
                if day["Low"] < trough:
                    trough = day["Low"]
                    if trailing_sl_pct is not None:
                        new_sl = trough * (1 + trailing_sl_pct / 100)
                        sl = min(sl, new_sl)

                # Check target
                if target_pct is not None and day["Low"] <= entry_price * (1 - target_pct / 100):
                    exit_price = entry_price * (1 - target_pct / 100)
                    exit_date = df.index[i + j]
                    exit_reason = "target"
                    break

        # Time exit
        if exit_price is None:
            last_j = min(max_hold_days, len(df) - i - 1)
            if last_j > 0:
                exit_price = df.iloc[i + last_j]["Close"]
                exit_date = df.index[i + last_j]
                exit_reason = "time_exit"
            else:
                i += 1
                continue

        if direction == "long":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        trades.append(Trade(
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
        ))

        # Skip to after exit
        exit_loc = df.index.get_loc(exit_date)
        i = exit_loc + 1

    return trades


# ─────────────────────────────────────────────────────────────
# Options strategy
# ─────────────────────────────────────────────────────────────

def run_options_backtest(df, sl_pct=1.0, trailing_sl_pct=None, max_hold_days=5,
                          target_pct=None, trade_type="both", target_delta=0.6):
    """
    Backtest options strategy. Buy ITM call on bullish breakout, ITM put on bearish.
    Use Black-Scholes with VIX as IV. Price options at entry and exit.
    SL/target applied on underlying, option P&L computed from repricing.
    """
    trades = []
    r = 0.06  # risk-free rate
    T_entry = 7 / 365  # assume weekly expiry, ~7 DTE

    i = 0
    while i < len(df):
        row = df.iloc[i]
        is_bull = row["Bullish_Breakout"]
        is_bear = row["Bearish_Breakout"]

        if not is_bull and not is_bear:
            i += 1
            continue

        if is_bull and trade_type == "short":
            i += 1
            continue
        if is_bear and trade_type == "long":
            i += 1
            continue

        direction = "long" if is_bull else "short"
        entry_price = row["Close"]
        entry_date = df.index[i]
        lot_size = get_lot_size(entry_date)
        vix = row["VIX"]
        sigma = vix / 100

        if direction == "long":
            opt_type = "call"
            sl_underlying = entry_price * (1 - sl_pct / 100)
            peak = entry_price
        else:
            opt_type = "put"
            sl_underlying = entry_price * (1 + sl_pct / 100)
            trough = entry_price

        strike, entry_premium, delta = find_itm_strike(
            entry_price, vix, opt_type, target_delta, r, T_entry)

        exit_underlying = None
        exit_date = None
        exit_reason = ""
        days_held = 0

        for j in range(1, min(max_hold_days + 1, len(df) - i)):
            day = df.iloc[i + j]

            if direction == "long":
                if day["Low"] <= sl_underlying:
                    exit_underlying = sl_underlying
                    exit_date = df.index[i + j]
                    exit_reason = "stop_loss"
                    days_held = j
                    break
                if day["High"] > peak:
                    peak = day["High"]
                    if trailing_sl_pct is not None:
                        new_sl = peak * (1 - trailing_sl_pct / 100)
                        sl_underlying = max(sl_underlying, new_sl)
                if target_pct is not None and day["High"] >= entry_price * (1 + target_pct / 100):
                    exit_underlying = entry_price * (1 + target_pct / 100)
                    exit_date = df.index[i + j]
                    exit_reason = "target"
                    days_held = j
                    break
            else:
                if day["High"] >= sl_underlying:
                    exit_underlying = sl_underlying
                    exit_date = df.index[i + j]
                    exit_reason = "stop_loss"
                    days_held = j
                    break
                if day["Low"] < trough:
                    trough = day["Low"]
                    if trailing_sl_pct is not None:
                        new_sl = trough * (1 + trailing_sl_pct / 100)
                        sl_underlying = min(sl_underlying, new_sl)
                if target_pct is not None and day["Low"] <= entry_price * (1 - target_pct / 100):
                    exit_underlying = entry_price * (1 - target_pct / 100)
                    exit_date = df.index[i + j]
                    exit_reason = "target"
                    days_held = j
                    break

        if exit_underlying is None:
            last_j = min(max_hold_days, len(df) - i - 1)
            if last_j > 0:
                exit_underlying = df.iloc[i + last_j]["Close"]
                exit_date = df.index[i + last_j]
                exit_reason = "time_exit"
                days_held = last_j
            else:
                i += 1
                continue

        # Reprice option at exit
        T_exit = max((T_entry - days_held / 365), 1 / 365)
        # Use same sigma (VIX doesn't change dramatically in a few days for pricing purposes)
        if opt_type == "call":
            exit_premium = bs_call_price(exit_underlying, strike, T_exit, r, sigma)
        else:
            exit_premium = bs_put_price(exit_underlying, strike, T_exit, r, sigma)

        option_pnl_points = exit_premium - entry_premium
        # Futures-equivalent P&L for comparison
        if direction == "long":
            futures_pnl = exit_underlying - entry_price
        else:
            futures_pnl = entry_price - exit_underlying

        trades.append(Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_underlying,
            stop_loss=sl_underlying,
            pnl_points=futures_pnl,
            pnl_rupees=futures_pnl * lot_size,
            lot_size=lot_size,
            exit_reason=exit_reason,
            option_type=opt_type,
            strike=round(strike, 2),
            entry_premium=round(entry_premium, 2),
            exit_premium=round(exit_premium, 2),
            delta=round(delta, 3),
            option_pnl_points=round(option_pnl_points, 2),
            option_pnl_rupees=round(option_pnl_points * lot_size, 2),
        ))

        exit_loc = df.index.get_loc(exit_date)
        i = exit_loc + 1

    return trades


# ─────────────────────────────────────────────────────────────
# Trade analysis
# ─────────────────────────────────────────────────────────────

def analyze_trades(trades, label=""):
    """Print trade statistics."""
    if not trades:
        print(f"  {label}: No trades")
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    total = len(df)
    winners = (df["pnl_points"] > 0).sum()
    losers = (df["pnl_points"] < 0).sum()
    win_rate = winners / total * 100

    total_pnl = df["pnl_rupees"].sum()
    avg_win = df.loc[df["pnl_points"] > 0, "pnl_rupees"].mean() if winners > 0 else 0
    avg_loss = df.loc[df["pnl_points"] < 0, "pnl_rupees"].mean() if losers > 0 else 0
    profit_factor = (df.loc[df["pnl_points"] > 0, "pnl_rupees"].sum() /
                     abs(df.loc[df["pnl_points"] < 0, "pnl_rupees"].sum())) if losers > 0 else float("inf")

    # Max drawdown on cumulative P&L
    cum_pnl = df["pnl_rupees"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # By exit reason
    by_reason = df.groupby("exit_reason")["pnl_rupees"].agg(["count", "sum", "mean"])

    # By direction
    by_dir = df.groupby("direction")["pnl_rupees"].agg(["count", "sum", "mean"])

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

    stats = {
        "total_trades": total, "win_rate": win_rate,
        "total_pnl": total_pnl, "profit_factor": profit_factor,
        "max_drawdown": max_dd, "avg_win": avg_win, "avg_loss": avg_loss,
    }

    if df["option_pnl_rupees"].any():
        opt_total = df["option_pnl_rupees"].sum()
        opt_winners = (df["option_pnl_points"] > 0).sum()
        opt_wr = opt_winners / total * 100
        print(f"\n  Options P&L: Rs {opt_total:,.0f}")
        print(f"  Options Win Rate: {opt_wr:.1f}%")
        stats["options_total_pnl"] = opt_total
        stats["options_win_rate"] = opt_wr

    return stats


def trades_to_df(trades):
    """Convert trades to DataFrame."""
    records = []
    for t in trades:
        records.append({
            "Entry Date": t.entry_date.strftime("%Y-%m-%d"),
            "Exit Date": t.exit_date.strftime("%Y-%m-%d"),
            "Direction": t.direction,
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

def optimize_futures(df, n_trials=300, trade_type="both"):
    """Use Optuna to find optimal SL/trailing SL/hold period/target."""

    def objective(trial):
        sl_pct = trial.suggest_float("sl_pct", 0.3, 3.0, step=0.1)
        use_trailing = trial.suggest_categorical("use_trailing", [True, False])
        trailing_sl_pct = trial.suggest_float("trailing_sl_pct", 0.3, 3.0, step=0.1) if use_trailing else None
        max_hold = trial.suggest_int("max_hold_days", 1, 20)
        use_target = trial.suggest_categorical("use_target", [True, False])
        target_pct = trial.suggest_float("target_pct", 0.5, 5.0, step=0.1) if use_target else None

        trades = run_futures_backtest(df, sl_pct, trailing_sl_pct, max_hold, target_pct, trade_type)
        if len(trades) < 50:
            return -1e9

        pnl = sum(t.pnl_rupees for t in trades)
        n = len(trades)
        win_rate = sum(1 for t in trades if t.pnl_points > 0) / n

        # Cumulative P&L for drawdown
        cum = np.cumsum([t.pnl_rupees for t in trades])
        max_dd = (cum - np.maximum.accumulate(cum)).min()

        # Penalize large drawdowns — use calmar-like ratio
        if max_dd == 0:
            return pnl
        calmar = pnl / abs(max_dd) if max_dd < 0 else pnl

        # Multi-objective: maximize P&L * sqrt(win_rate) / drawdown_penalty
        return calmar * np.sqrt(win_rate)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def optimize_options(df, n_trials=300, trade_type="both"):
    """Use Optuna to find optimal parameters for options strategy."""

    def objective(trial):
        sl_pct = trial.suggest_float("sl_pct", 0.3, 3.0, step=0.1)
        use_trailing = trial.suggest_categorical("use_trailing", [True, False])
        trailing_sl_pct = trial.suggest_float("trailing_sl_pct", 0.3, 3.0, step=0.1) if use_trailing else None
        max_hold = trial.suggest_int("max_hold_days", 1, 10)
        use_target = trial.suggest_categorical("use_target", [True, False])
        target_pct = trial.suggest_float("target_pct", 0.5, 5.0, step=0.1) if use_target else None
        target_delta = trial.suggest_float("target_delta", 0.55, 0.75, step=0.05)

        trades = run_options_backtest(df, sl_pct, trailing_sl_pct, max_hold, target_pct,
                                      trade_type, target_delta)
        if len(trades) < 50:
            return -1e9

        pnl = sum(t.option_pnl_rupees for t in trades)
        n = len(trades)
        win_rate = sum(1 for t in trades if t.option_pnl_points > 0) / n

        cum = np.cumsum([t.option_pnl_rupees for t in trades])
        max_dd = (cum - np.maximum.accumulate(cum)).min()

        if max_dd == 0:
            return pnl
        calmar = pnl / abs(max_dd) if max_dd < 0 else pnl
        return calmar * np.sqrt(win_rate)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_equity_curves(futures_trades, options_trades, label_suffix=""):
    """Plot cumulative P&L for both strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

    for ax, trades, label, pnl_field in [
        (axes[0], futures_trades, "Futures", "pnl_rupees"),
        (axes[1], options_trades, "Options (ITM ~0.6 delta)", "option_pnl_rupees"),
    ]:
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
        ax.set_title(f"{label} — Equity Curve{label_suffix}")
        ax.axhline(0, color="black", linewidth=0.5)

        # Annotate final P&L
        ax.annotate(f"Rs {cum_pnl[-1]:,.0f}", xy=(dates[-1], cum_pnl[-1]),
                    fontsize=10, fontweight="bold",
                    color="forestgreen" if cum_pnl[-1] > 0 else "crimson")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"equity_curves{label_suffix.replace(' ','_')}.png", dpi=150)
    print(f"Saved equity_curves{label_suffix.replace(' ','_')}.png")


def plot_optimization_results(study, label):
    """Plot Optuna optimization insights."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Optimization history
    ax = axes[0]
    trials = [t for t in study.trials if t.value is not None and t.value > -1e8]
    values = [t.value for t in trials]
    ax.scatter(range(len(values)), values, alpha=0.3, s=10, color="steelblue")
    # Running best
    running_best = np.maximum.accumulate(values)
    ax.plot(running_best, color="crimson", linewidth=1.5, label="Running best")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective (Calmar * sqrt(WR))")
    ax.set_title(f"{label} — Optimization History")
    ax.legend()

    # Parameter importance
    ax = axes[1]
    importances = optuna.importance.get_param_importances(study)
    params = list(importances.keys())[:8]
    imp_vals = [importances[p] for p in params]
    ax.barh(params, imp_vals, color="steelblue", alpha=0.7)
    ax.set_xlabel("Importance")
    ax.set_title(f"{label} — Parameter Importance")

    plt.tight_layout()
    suffix = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(OUTPUT_DIR / f"optuna_{suffix}.png", dpi=150)
    print(f"Saved optuna_{suffix}.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    df = load_data()
    print(f"Data: {df.index.min().date()} to {df.index.max().date()}, {len(df)} days")

    # ── 1. Baseline futures with simple parameters ──
    print("\n" + "="*60)
    print("  BASELINE STRATEGIES (SL=1%, hold=5 days, no trailing)")
    print("="*60)

    base_futures = run_futures_backtest(df, sl_pct=1.0, max_hold_days=5)
    base_options = run_options_backtest(df, sl_pct=1.0, max_hold_days=5)

    fut_stats = analyze_trades(base_futures, "Futures Baseline (SL=1%, 5-day hold)")
    opt_stats = analyze_trades(base_options, "Options Baseline (SL=1%, 5-day hold, 0.6 delta)")

    # Save trade tables
    trades_df_fut = trades_to_df(base_futures)
    trades_df_opt = trades_to_df(base_options)
    trades_df_fut.to_csv(OUTPUT_DIR / "trades_futures_baseline.csv", index=False)
    trades_df_opt.to_csv(OUTPUT_DIR / "trades_options_baseline.csv", index=False)
    print(f"\nSaved {len(base_futures)} futures trades and {len(base_options)} options trades to CSV")

    plot_equity_curves(base_futures, base_options, " (Baseline)")

    # ── 2. Optuna optimization for futures ──
    print("\n" + "="*60)
    print("  OPTIMIZING FUTURES STRATEGY WITH OPTUNA (300 trials)")
    print("="*60)

    fut_study = optimize_futures(df, n_trials=300)
    best_fut = fut_study.best_params
    print(f"\n  Best futures parameters: {json.dumps(best_fut, indent=2, default=str)}")

    # Run with best params
    opt_fut_trades = run_futures_backtest(
        df,
        sl_pct=best_fut["sl_pct"],
        trailing_sl_pct=best_fut.get("trailing_sl_pct"),
        max_hold_days=best_fut["max_hold_days"],
        target_pct=best_fut.get("target_pct"),
    )
    analyze_trades(opt_fut_trades, "Futures OPTIMIZED")
    plot_optimization_results(fut_study, "Futures")

    # ── 3. Optuna optimization for options ──
    print("\n" + "="*60)
    print("  OPTIMIZING OPTIONS STRATEGY WITH OPTUNA (300 trials)")
    print("="*60)

    opt_study = optimize_options(df, n_trials=300)
    best_opt = opt_study.best_params
    print(f"\n  Best options parameters: {json.dumps(best_opt, indent=2, default=str)}")

    # Run with best params
    opt_opt_trades = run_options_backtest(
        df,
        sl_pct=best_opt["sl_pct"],
        trailing_sl_pct=best_opt.get("trailing_sl_pct"),
        max_hold_days=best_opt["max_hold_days"],
        target_pct=best_opt.get("target_pct"),
        target_delta=best_opt.get("target_delta", 0.6),
    )
    analyze_trades(opt_opt_trades, "Options OPTIMIZED")
    plot_optimization_results(opt_study, "Options")

    # Save optimized trade tables
    trades_df_fut_opt = trades_to_df(opt_fut_trades)
    trades_df_opt_opt = trades_to_df(opt_opt_trades)
    trades_df_fut_opt.to_csv(OUTPUT_DIR / "trades_futures_optimized.csv", index=False)
    trades_df_opt_opt.to_csv(OUTPUT_DIR / "trades_options_optimized.csv", index=False)
    print(f"Saved {len(opt_fut_trades)} optimized futures trades")
    print(f"Saved {len(opt_opt_trades)} optimized options trades")

    plot_equity_curves(opt_fut_trades, opt_opt_trades, " (Optimized)")

    # ── 4. Summary comparison ──
    print("\n" + "="*60)
    print("  STRATEGY COMPARISON SUMMARY")
    print("="*60)
    print(f"\n  {'Strategy':<35} {'P&L':>12} {'Win%':>7} {'PF':>6} {'MaxDD':>12} {'Trades':>7}")
    print("  " + "-" * 79)

    for label, trades, pnl_field in [
        ("Futures Baseline", base_futures, "pnl_rupees"),
        ("Futures Optimized", opt_fut_trades, "pnl_rupees"),
        ("Options Baseline", base_options, "option_pnl_rupees"),
        ("Options Optimized", opt_opt_trades, "option_pnl_rupees"),
    ]:
        if not trades:
            continue
        pnls = [getattr(t, pnl_field) for t in trades]
        total_pnl = sum(pnls)
        n = len(trades)
        pnl_pts = [t.pnl_points if pnl_field == "pnl_rupees" else t.option_pnl_points for t in trades]
        wr = sum(1 for p in pnl_pts if p > 0) / n * 100
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        cum = np.cumsum(pnls)
        max_dd = (cum - np.maximum.accumulate(cum)).min()
        print(f"  {label:<35} {total_pnl:>12,.0f} {wr:>6.1f}% {pf:>6.2f} {max_dd:>12,.0f} {n:>7}")

    # Save best params
    best_params = {
        "futures": best_fut,
        "options": best_opt,
    }
    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\nBest parameters saved to analysis/best_params.json")

    print("\nDone! All outputs in analysis/")


if __name__ == "__main__":
    main()
