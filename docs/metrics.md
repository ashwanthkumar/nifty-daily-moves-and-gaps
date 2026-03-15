# Metrics

## Intraday Movement %

The total price path traversed within a trading day, as a percentage of the opening price. This captures *how much the market moved*, not just the net change.

The path direction depends on whether the day closed positive or negative:

- **Positive day** (Close >= Open): `Open -> Low -> High -> Close`
  - Movement = (Open - Low) + (High - Low) + (High - Close)
- **Negative day** (Close < Open): `Open -> High -> Low -> Close`
  - Movement = (High - Open) + (High - Low) + (Close - Low)

Expressed as a percentage: `Movement / Open * 100`

### Why this metric?

A day that opens at 100, drops to 95, rises to 105, and closes at 100 has 0% net return but 20% intraday movement. This metric captures the actual volatility experienced by intraday traders.

## Gap %

The overnight gap between the previous day's close and today's open:

```
Gap % = (Today's Open - Yesterday's Close) / Yesterday's Close * 100
```

Decomposed into three views:
- **Abs Gap %**: `|Gap %|` -- magnitude regardless of direction
- **Positive Gap %**: `max(0, Gap %)` -- gap-up days only
- **Negative Gap %**: `min(0, Gap %)` -- gap-down days only

## Breakout

A day where the market closes outside the previous day's range:

- **Bullish breakout**: `Close > Previous Day's High`
- **Bearish breakout**: `Close < Previous Day's Low`

## Continuation Streak

After a breakout, the number of consecutive days the market continues in the breakout direction:
- After bullish breakout: each subsequent day closing higher than the prior day
- After bearish breakout: each subsequent day closing lower than the prior day

The streak ends when the first reversal occurs.

## Correlation Methods Used

| Method | Type | What it captures | Sensitivity |
|--------|------|-----------------|-------------|
| Pearson | Linear | Linear relationships | Sensitive to outliers |
| Spearman | Rank | Monotonic non-linear relationships | Robust to outliers |
| Kendall | Rank | Concordance of pairs | Most robust, best for small samples |
| Mutual Information | Information-theoretic | Any statistical dependency | Captures non-linear, non-monotonic |
