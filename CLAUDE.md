# CLAUDE.md

## Project overview

NIFTY 50 market analysis project. Analyzes intraday price movement, overnight gaps, and their correlation with India VIX. Also studies breakout continuation patterns.

## Structure

- `main.py` -- data download + base charts
- `correlation_analysis.py` -- 7 correlation methods between metrics
- `continuation_analysis.py` -- breakout continuation analysis
- `strategy_backtest.py` -- futures & options breakout backtest with Optuna optimization (daily data)
- `strategy_backtest_5min.py` -- EOD vs intraday entry backtest using 5-min candles
- `data/` -- cached CSV data (NIFTY 50, India VIX)
- `analysis/` -- generated charts and CSVs
- `docs/` -- detailed documentation (setup, metrics, usage)
- `README.md` -- results and findings only

## Documentation convention

- `README.md` contains only results and key findings (no setup/usage instructions)
- `docs/` contains setup instructions (`docs/setup.md`) and metric definitions (`docs/metrics.md`)

## Running

```bash
uv run python main.py
uv run python correlation_analysis.py
uv run python continuation_analysis.py
uv run python strategy_backtest.py
uv run python strategy_backtest_5min.py
```
