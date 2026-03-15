# NIFTY 50 — Intraday Movement & Gap Analysis

Analysis of NIFTY 50 intraday price movement, overnight gaps, and their correlation with India VIX.

**Data**: ~4,400 trading days from 2008-03-03 to 2026-03-13 (source: Yahoo Finance)

## Metrics

### Intraday Movement %
The total price path traversed within a trading day, as a percentage of the opening price:
- **Positive day** (Close >= Open): path is `Open -> Low -> High -> Close`
- **Negative day** (Close < Open): path is `Open -> High -> Low -> Close`

This captures *how much the market moved*, not just the net change.

### Gap %
The overnight gap: `(Today's Open - Yesterday's Close) / Yesterday's Close * 100`

Decomposed into:
- **Abs Gap**: magnitude regardless of direction
- **Positive Gap**: gap-up days only (clipped at 0)
- **Negative Gap**: gap-down days only (clipped at 0)

## Setup & Run

```bash
uv run python main.py                # download data + generate base charts
uv run python correlation_analysis.py # run all correlation analyses
```

Data is cached locally in `data/` after the first download.

## Charts

### Base Charts (`main.py`)

| Chart | File |
|-------|------|
| Intraday movement (bars) + Gap lines (abs/pos/neg) on dual axis | `nifty_intraday_and_gaps.png` |
| Intraday movement (bars) + India VIX on dual axis | `nifty_intraday_vs_vix.png` |

### Correlation Analysis (`correlation_analysis.py`)

All outputs saved to `analysis/`.

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

**Why three methods?**
- **Pearson**: measures linear relationships; sensitive to outliers
- **Spearman**: rank-based; captures monotonic non-linear relationships
- **Kendall**: rank-based; more robust to outliers, better for small samples

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

### 3. Lagged Cross-Correlations (+/-20 days)

Tests whether one variable *leads* another.

| Pair | Best Lag | Correlation |
|------|----------|-------------|
| VIX -> Intraday Movement | -1 day | 0.710 |
| Intraday Movement -> VIX | +1 day | 0.710 |
| Abs Gap -> Intraday Movement | 0 days | 0.168 |
| VIX -> Abs Gap | -1 day | 0.136 |

**Key finding**: VIX and intraday movement are most correlated at lag +/-1 (r=0.71, slightly higher than lag-0 r=0.707). This is because VIX reflects *expected* volatility -- yesterday's VIX is a marginally better predictor of today's intraday movement than today's VIX itself. The relationship is symmetric, meaning high intraday movement also predicts tomorrow's VIX staying elevated.

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

**Key finding**: The VIX-Intraday MI (0.159) confirms significant non-linear dependency beyond what Pearson captures. The high MI between Abs Gap and signed Gap (0.299) is expected -- they're mathematically related. Intraday vs Abs Gap MI is very low (0.021), meaning there's genuinely no relationship, not just a non-linear one hiding from Pearson.

**Chart**: `analysis/mutual_information.png`

---

### 5. Regime-Based Analysis (VIX Regimes)

Correlations computed separately for different VIX environments:

| VIX Regime | N days | Intraday vs VIX (p) | Intraday vs Abs Gap (p) | Abs Gap vs VIX (p) |
|------------|--------|---------------------|------------------------|-------------------|
| Low VIX (< 17.1) | 2,196 | 0.382 | 0.087 | 0.104 |
| High VIX (>= 17.1) | 2,205 | 0.633 | -0.120 | -0.196 |
| Very High VIX (>= 22.1) | 1,102 | 0.605 | -0.165 | -0.276 |

**Key finding**: The Intraday-VIX correlation nearly *doubles* in high-VIX regimes (0.38 -> 0.63). During calm markets, VIX is a weaker predictor of intraday movement. The Abs Gap vs VIX correlation *flips negative* in high-VIX environments -- during crises, high VIX doesn't necessarily mean large gaps; it means large intraday swings.

Median intraday movement by VIX bucket:
- VIX < 12: ~1.1%
- VIX 12-16: ~1.5%
- VIX 16-20: ~1.9%
- VIX 20-30: ~2.6%
- VIX > 30: ~4.2%

**Chart**: `analysis/regime_boxplots.png`

---

### 6. Scatter Plots with Linear Regression

Visual confirmation of relationships with regression lines:

| Pair | r | Slope | Interpretation |
|------|---|-------|----------------|
| VIX vs Intraday % | 0.707 | 0.124 | Each 1-point VIX increase -> ~0.12% more intraday movement |
| VIX vs Abs Gap % | 0.132 | 0.007 | Weak: VIX barely predicts gap size |
| Abs Gap vs Intraday % | 0.168 | 0.563 | Weak but positive -- large gaps tend to come with volatile days |

**Chart**: `analysis/scatter_regression.png`

---

## Summary of Findings

1. **VIX is a strong predictor of intraday movement** (r=0.71). This is the dominant relationship in the dataset.
2. **Gaps are largely independent** of both VIX and intraday movement. They contain distinct information.
3. **The VIX-Intraday relationship is non-stationary** -- it's stronger during high-volatility regimes and weaker during calm markets.
4. **No meaningful lead-lag** exists between variables beyond 1 day. VIX at lag-1 is marginally better than contemporaneous VIX for predicting intraday movement.
5. **Negative gaps weakly predict larger intraday movement** (r=-0.086 for signed gap vs intraday), meaning gap-down days tend to be slightly more volatile intraday.
6. **Mutual information confirms** the VIX-Intraday link is the only meaningful non-linear dependency; gaps are genuinely independent, not just non-linearly related.
