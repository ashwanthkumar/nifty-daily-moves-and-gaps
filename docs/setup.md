# Setup & Usage

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Running

```bash
uv run python main.py                 # download data + generate base charts
uv run python correlation_analysis.py  # run all correlation analyses
uv run python continuation_analysis.py # breakout continuation analysis
```

Data is cached locally in `data/` after the first download. Delete the CSV files to re-download fresh data.

## Scripts

### `main.py`
Downloads NIFTY 50 (`^NSEI`) and India VIX (`^INDIAVIX`) historical data from Yahoo Finance. Computes intraday movement and gap metrics. Generates two dual-axis charts:

| Chart | File |
|-------|------|
| Intraday movement (bars) + Gap lines (abs/pos/neg) on dual axis | `nifty_intraday_and_gaps.png` |
| Intraday movement (bars) + India VIX on dual axis | `nifty_intraday_vs_vix.png` |

### `correlation_analysis.py`
Runs 7 correlation approaches between intraday movement, gaps, and VIX. All outputs saved to `analysis/`.

Generated charts:
- `analysis/heatmap_pearson.png`, `analysis/heatmap_spearman.png`, `analysis/heatmap_kendall.png` -- correlation matrices
- `analysis/rolling_correlations.png` -- 60-day and 252-day rolling correlations
- `analysis/lagged_correlations.png` -- lagged cross-correlations (+/-20 days)
- `analysis/mutual_information.png` -- non-linear dependency heatmap
- `analysis/regime_boxplots.png` -- intraday movement and gap distributions by VIX regime
- `analysis/scatter_regression.png` -- scatter plots with regression lines

Generated CSVs:
- `analysis/static_correlations.csv`
- `analysis/lagged_correlations.csv`
- `analysis/mutual_information.csv`
- `analysis/regime_correlations.csv`
- `analysis/summary.txt`

### `continuation_analysis.py`
Analyzes breakout continuation patterns when the market closes above the previous day's high or below the previous day's low.

Generated charts:
- `analysis/continuation_streaks.png` -- streak length distributions
- `analysis/continuation_forward_returns.png` -- mean/median forward returns
- `analysis/continuation_win_rates.png` -- win rate curves over time
- `analysis/continuation_timeline.png` -- NIFTY price with breakout days marked
- `analysis/continuation_streak_vs_return.png` -- return by streak length

Generated CSVs:
- `analysis/continuation_summary.csv`
