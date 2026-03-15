# NIFTY 50 — Intraday Movement & Gap Analysis

Analysis of NIFTY 50 intraday price movement, overnight gaps, and their correlation with India VIX.

**Data**: ~4,500 trading days from 2007-09-17 to 2026-03-13 (source: Yahoo Finance)

> For setup instructions, metric definitions, and script details see [docs/](docs/).

## Table of Contents

- [Correlation Results](#correlation-results)
  - [1. Static Correlations (Pearson, Spearman, Kendall)](#1-static-correlations-pearson-spearman-kendall)
  - [2. Rolling Correlations](#2-rolling-correlations-60-day--252-day-windows)
  - [3. Lagged Cross-Correlations](#3-lagged-cross-correlations---20-days)
  - [4. Mutual Information](#4-mutual-information-non-linear-dependencies)
  - [5. Regime-Based Analysis](#5-regime-based-analysis-vix-regimes)
  - [6. Scatter Plots with Linear Regression](#6-scatter-plots-with-linear-regression)
- [Correlation Summary](#correlation-summary)
- [Breakout Continuation Analysis](#breakout-continuation-analysis)
  - [Frequency](#frequency)
  - [Continuation Streaks](#continuation-streaks)
  - [Forward Returns After Breakouts](#forward-returns-after-breakouts)
  - [Continuation Return by Streak Length](#continuation-return-by-streak-length)
  - [Breakout Velocity and Magnitude](#breakout-velocity-and-magnitude)
  - [Does Breakout Size Predict Continuation?](#does-breakout-size-predict-continuation)
  - [Recovery After Bearish Breakdowns](#recovery-after-bearish-breakdowns)
- [Continuation Summary](#continuation-summary)

---

## Correlation Results

### 1. Static Correlations (Pearson, Spearman, Kendall)

Three classical correlation measures applied to all variable pairs:

| Pair | Pearson r | Spearman p | Kendall t |
|------|-----------|------------|-----------|
| **Intraday Movement vs VIX** | **0.707** | **0.684** | **0.499** |
| Intraday Movement vs Abs Gap | 0.168 | -0.007 | -0.000 |
| Intraday Movement vs Gap (signed) | -0.086 | -0.092 | -0.061 |
| Abs Gap vs VIX | 0.132 | -0.018 | -0.002 |
| VIX vs Gap (signed) | -0.048 | -0.075 | -0.046 |
| Intraday Movement vs Daily Return | -0.065 | -0.098 | -0.070 |

**Key finding**: Intraday movement and VIX are strongly correlated (~0.70) across all three methods, confirming VIX is an effective gauge of realized intraday volatility. All other pairs show weak correlations.

The Intraday-vs-Abs-Gap pair is notable: Pearson shows 0.168 but Spearman drops to -0.007 -- the linear correlation is driven by a few extreme outlier days (crisis periods) where both spiked together.

**Charts**: `analysis/heatmap_pearson.png`, `analysis/heatmap_spearman.png`, `analysis/heatmap_kendall.png`

---

### 2. Rolling Correlations (60-day & 252-day windows)

Time-varying correlations to see if relationships are stable or regime-dependent.

| Pair | 252-day rolling range | Observation |
|------|----------------------|-------------|
| Intraday vs VIX | 0.30 - 0.85 | Consistently positive, but strength varies significantly |
| Intraday vs Abs Gap | -0.40 - 0.60 | Unstable -- swings between positive and negative |
| Abs Gap vs VIX | -0.30 - 0.55 | Mostly weak, occasionally spikes during crisis periods |

**Key finding**: The Intraday-VIX relationship is the only stable one. The 60-day window reveals short-term regime shifts where even this relationship temporarily weakens (e.g., post-COVID normalization).

**Chart**: `analysis/rolling_correlations.png`

---

### 3. Lagged Cross-Correlations ( +/-20 days)

Tests whether one variable *leads* another.

| Pair | Best Lag | Correlation |
|------|----------|-------------|
| VIX -> Intraday Movement | -1 day | 0.710 |
| Intraday Movement -> VIX | +1 day | 0.710 |
| Abs Gap -> Intraday Movement | 0 days | 0.168 |
| VIX -> Abs Gap | -1 day | 0.136 |

**Key finding**: VIX and intraday movement are most correlated at lag +/-1 (r=0.71, slightly higher than lag-0 r=0.707). Yesterday's VIX is a marginally better predictor of today's intraday movement than today's VIX itself. The relationship is symmetric -- high intraday movement also predicts tomorrow's VIX staying elevated.

**Chart**: `analysis/lagged_correlations.png`

---

### 4. Mutual Information (Non-linear Dependencies)

Captures *any* statistical dependency, not just linear/monotonic. Values normalized to [0, 1].

| Pair | Normalized MI |
|------|--------------|
| **Intraday vs VIX** | **0.159** |
| Abs Gap vs Gap (signed) | 0.299 |
| Intraday vs Return | 0.077 |
| VIX vs Return | 0.065 |
| Intraday vs Abs Gap | 0.021 |

**Key finding**: The VIX-Intraday MI (0.159) confirms significant non-linear dependency beyond what Pearson captures. Intraday vs Abs Gap MI is very low (0.021), meaning there's genuinely no relationship, not just a non-linear one hiding from Pearson.

**Chart**: `analysis/mutual_information.png`

---

### 5. Regime-Based Analysis (VIX Regimes)

Correlations computed separately for different VIX environments:

| VIX Regime | N days | Intraday vs VIX (p) | Intraday vs Abs Gap (p) | Abs Gap vs VIX (p) |
|------------|--------|---------------------|------------------------|-------------------|
| Low VIX (< 17.1) | 2,196 | 0.382 | 0.087 | 0.104 |
| High VIX (>= 17.1) | 2,205 | 0.633 | -0.120 | -0.196 |
| Very High VIX (>= 22.1) | 1,102 | 0.605 | -0.165 | -0.276 |

**Key finding**: The Intraday-VIX correlation nearly *doubles* in high-VIX regimes (0.38 -> 0.63). The Abs Gap vs VIX correlation *flips negative* in high-VIX environments -- during crises, high VIX means large intraday swings, not necessarily large gaps.

Median intraday movement by VIX bucket:
- VIX < 12: ~1.1%
- VIX 12-16: ~1.5%
- VIX 16-20: ~1.9%
- VIX 20-30: ~2.6%
- VIX > 30: ~4.2%

**Chart**: `analysis/regime_boxplots.png`

---

### 6. Scatter Plots with Linear Regression

| Pair | r | Slope | Interpretation |
|------|---|-------|----------------|
| VIX vs Intraday % | 0.707 | 0.124 | Each 1-point VIX increase -> ~0.12% more intraday movement |
| VIX vs Abs Gap % | 0.132 | 0.007 | Weak: VIX barely predicts gap size |
| Abs Gap vs Intraday % | 0.168 | 0.563 | Weak but positive -- large gaps tend to come with volatile days |

**Chart**: `analysis/scatter_regression.png`

---

## Correlation Summary

1. **VIX is a strong predictor of intraday movement** (r=0.71). This is the dominant relationship in the dataset.
2. **Gaps are largely independent** of both VIX and intraday movement. They contain distinct information.
3. **The VIX-Intraday relationship is non-stationary** -- it's stronger during high-volatility regimes and weaker during calm markets.
4. **No meaningful lead-lag** exists between variables beyond 1 day. VIX at lag-1 is marginally better than contemporaneous VIX for predicting intraday movement.
5. **Negative gaps weakly predict larger intraday movement** (r=-0.086 for signed gap vs intraday), meaning gap-down days tend to be slightly more volatile intraday.
6. **Mutual information confirms** the VIX-Intraday link is the only meaningful non-linear dependency; gaps are genuinely independent.

---

## Breakout Continuation Analysis

When NIFTY closes above the previous day's high (bullish breakout) or below the previous day's low (bearish breakout), does the move continue? For how long?

### Frequency

| Type | Count | % of Trading Days |
|------|-------|-------------------|
| Bullish breakout (Close > Prev High) | 1,446 | 31.9% |
| Bearish breakout (Close < Prev Low) | 1,190 | 26.2% |
| Neither (Close within prev range) | ~42% | -- |

About 58% of trading days close outside the previous day's range -- breakouts are common, not exceptional.

### Continuation Streaks

How many consecutive days does the move continue (each day closing higher/lower than the prior)?

| Streak | Bullish | Bearish |
|--------|---------|---------|
| 0 (reverses next day) | 42.7% | 48.3% |
| 1 day | 25.0% | 24.5% |
| 2 days | 14.4% | 12.6% |
| 3 days | 7.5% | 7.0% |
| 4+ days | 10.5% | 7.7% |
| **Mean streak** | **1.3 days** | **1.1 days** |
| **Max streak** | **12 days** | **7 days** |

Roughly half of all breakouts reverse the very next day. Bullish breakouts have a slight continuation edge over bearish ones (57.3% vs 51.7% next-day win rate), and longer max streaks (12 vs 7 days).

**Chart**: `analysis/continuation_streaks.png`

### Forward Returns After Breakouts

Average returns over fixed forward periods following a breakout day:

**After Bullish Breakout:**

| Period | Mean Return | Median Return | Win Rate (% higher) |
|--------|------------|---------------|---------------------|
| 1 day | +0.117% | +0.114% | 57.3% |
| 2 days | +0.154% | +0.193% | 57.7% |
| 3 days | +0.209% | +0.316% | 57.5% |
| 5 days | +0.248% | +0.389% | 58.1% |
| 10 days | +0.510% | +0.694% | 59.1% |
| 20 days | +1.034% | +1.229% | 61.2% |

**After Bearish Breakout:**

| Period | Mean Return | Median Return | Win Rate (% lower) |
|--------|------------|---------------|---------------------|
| 1 day | -0.021% | -0.045% | 51.7% |
| 2 days | +0.028% | +0.030% | 49.4% |
| 3 days | +0.065% | +0.046% | 48.7% |
| 5 days | +0.201% | +0.242% | 46.1% |
| 10 days | +0.403% | +0.510% | 43.8% |
| 20 days | +0.608% | +0.985% | 39.4% |

**Charts**: `analysis/continuation_forward_returns.png`, `analysis/continuation_win_rates.png`

### Continuation Return by Streak Length

When a breakout does continue, how much return does each additional streak day add?

- Bullish: each additional day of streak adds roughly +0.5-0.7% cumulative return
- Bearish: each additional day adds roughly -1.0% cumulative return (but sample sizes get small beyond 5 days)

**Chart**: `analysis/continuation_streak_vs_return.png`

### Breakout Velocity and Magnitude

How far does the move go before consolidation/mean reversion, and how fast?

- **Breakout day magnitude**: how far beyond the previous day's high/low the close lands
- **Peak excursion**: the maximum favorable intraday move (using highs/lows) before 2 consecutive closes back past the breakout level
- **Days to peak**: trading days from breakout to peak excursion
- **Velocity**: peak excursion / days to peak (% per day)

| Metric | Bullish (median) | Bearish (median) |
|--------|-----------------|-----------------|
| Breakout day magnitude | 0.480% | 0.544% |
| Peak excursion | 1.364% | 1.306% |
| Days to peak | 3.0 | 2.0 |
| Velocity (% per day) | 0.311% | 0.460% |

**Key finding**: Bearish breakouts are *faster* -- they reach their peak excursion in 2 days (median) vs 3 for bullish, with 48% higher velocity (0.46%/day vs 0.31%/day). But both reach a similar peak magnitude (~1.3%). Bears move faster and exhaust sooner; bulls grind slower but persist longer.

The velocity by streak chart shows this clearly: as streaks extend, bullish peak excursion grows steadily while velocity stays flat (~0.3-0.5%/day). Bearish streaks maintain higher velocity (~1.0-1.5%/day) but cap out at 7 days.

**Charts**: `analysis/velocity_distributions.png`, `analysis/velocity_by_streak.png`

### Does Breakout Size Predict Continuation?

Breakout magnitude (how far beyond the prev high/low) is binned into quintiles to test if larger initial breakouts lead to longer streaks or bigger peak excursions.

| Observation | Bullish | Bearish |
|-------------|---------|---------|
| Magnitude vs Peak Excursion (r) | 0.173 | 0.177 |
| Magnitude vs Streak Length | Slight positive trend across quintiles | Flat / no trend |

**Key finding**: Initial breakout size is a *weak* predictor of how far the move goes (r~0.17 for both). Larger breakouts do tend to have slightly larger peak excursions, but the relationship is noisy. Breakout size does not meaningfully predict streak duration -- a small breakout can run for days, and a large one can reverse immediately.

**Chart**: `analysis/magnitude_vs_continuation.png`

### Recovery After Bearish Breakdowns

After a bearish breakdown (close < prev low), how long until the market closes back at or above the breakdown-day close?

**98% of breakdowns recover within 252 trading days (1 year).**

| Metric | Value |
|--------|-------|
| Recovered within 1 year | 98.0% (1,166 of 1,190) |
| Median recovery time | **2 days** |
| Mean recovery time | 10.2 days |
| P75 | 5 days |
| P90 | 22 days |

Recovery time distribution:

| Window | % of all breakdowns |
|--------|-------------------|
| Next day (1d) | 48.2% |
| 2 days | 12.5% |
| 3-5 days | 13.8% |
| 6-10 days | 8.2% |
| 11-20 days | 5.0% |
| 21-60 days | 6.1% |
| 61-252 days | 4.1% |
| Never (within 1 year) | 2.0% |

**Key finding**: Nearly half of all bearish breakdowns recover the very next day. 75% recover within 5 days. Only 2% fail to recover within a year -- these correspond to major crashes (2008 GFC, COVID 2020 crash).

**Recovery by streak length**: Longer bearish streaks take significantly longer to recover. Streak-0 breakdowns (immediate reversal) recover in 1 day median. Streak-4+ breakdowns take 14-22 days median and have a 7-13% chance of not recovering within a year.

**Breakdown size doesn't predict recovery time** (r=-0.012). A large single-day breakdown recovers just as fast as a small one. What matters is whether it *continues* falling (streak length), not how far it fell on day 1.

**Max adverse move**: Before recovering, the median further drop is only 0.81%. Even in breakdowns, the typical "pain" beyond the breakdown day is less than 1%.

**Charts**: `analysis/bearish_recovery.png`, `analysis/recovery_by_streak.png`

---

## Continuation Summary

1. **Bullish breakouts have genuine continuation**: 57% win rate at day 1, rising to 61% by day 20. The market's long-term upward drift (NIFTY ~5x over this period) provides a structural tailwind.

2. **Bearish breakouts are mean-reverting**: Only 52% continue on day 1, dropping to 39% by day 20. Closing below the previous low is actually a contrarian *buy* signal -- the market bounces back more often than it continues falling.

3. **The asymmetry is stark**: After a bullish breakout, you gain +1.03% over 20 days on average. After a bearish breakout, despite the breakdown, you still *gain* +0.61% over 20 days -- the bull bias overwhelms the breakdown signal.

4. **Streaks are short-lived**: Mean continuation is only 1.1-1.3 days. Most breakouts are noise, not the start of multi-day trends. The few that do extend (4+ days) deliver outsized returns but are rare (<11%).

5. **Bearish moves are more violent but shorter**: Max bearish streak is 7 days vs 12 for bullish. Bear moves exhaust themselves faster.

6. **Bearish breakouts are faster (higher velocity)**: 0.46%/day median velocity vs 0.31%/day for bullish. Bears reach peak excursion in 2 days (median) vs 3 for bulls. Same peak magnitude (~1.3%), but bears get there 50% faster.

7. **Breakout size is a poor predictor**: Initial magnitude only weakly predicts peak excursion (r~0.17) and barely predicts streak length. You can't tell from the breakout day alone whether it will continue.

8. **Bearish breakdowns recover fast**: Median recovery is just 2 days. 48% recover the next day, 75% within 5 days. Only 2% fail to recover within a year (major crashes). The breakdown magnitude doesn't predict recovery time -- what matters is whether the streak continues.

9. **Longer bearish streaks are harder to recover from**: Streak-0 recovers in 1 day. Streak-4+ takes 14-22 days median with up to 13% non-recovery risk within a year.
