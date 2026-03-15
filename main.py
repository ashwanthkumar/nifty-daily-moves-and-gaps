import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

NIFTY_CSV = DATA_DIR / "nifty50.csv"
VIX_CSV = DATA_DIR / "indiavix.csv"


def download_and_cache(ticker: str, csv_path: Path) -> pd.DataFrame:
    """Download from Yahoo Finance and cache locally as CSV."""
    if csv_path.exists():
        print(f"Loading cached data from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print(f"Downloading {ticker} from Yahoo Finance...")
        df = yf.download(ticker, period="max")
        # Flatten multi-level columns
        df.columns = df.columns.get_level_values(0)
        df.to_csv(csv_path)
        print(f"Saved to {csv_path}")
    return df


def compute_metrics(nifty: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday movement and gap metrics."""
    is_positive = nifty["Close"] >= nifty["Open"]

    # Positive day path: O → L → H → C
    intraday_pos = (
        (nifty["Open"] - nifty["Low"])
        + (nifty["High"] - nifty["Low"])
        + (nifty["High"] - nifty["Close"])
    )
    # Negative day path: O → H → L → C
    intraday_neg = (
        (nifty["High"] - nifty["Open"])
        + (nifty["High"] - nifty["Low"])
        + (nifty["Close"] - nifty["Low"])
    )

    nifty["Intraday"] = intraday_pos.where(is_positive, intraday_neg)
    nifty["Intraday_Pct"] = (nifty["Intraday"] / nifty["Open"]) * 100

    # Gap: today's open vs previous close
    nifty["Prev_Close"] = nifty["Close"].shift(1)
    nifty["Gap"] = nifty["Open"] - nifty["Prev_Close"]
    nifty["Gap_Pct"] = (nifty["Gap"] / nifty["Prev_Close"]) * 100
    nifty["Abs_Gap_Pct"] = nifty["Gap_Pct"].abs()
    nifty["Pos_Gap_Pct"] = nifty["Gap_Pct"].clip(lower=0)
    nifty["Neg_Gap_Pct"] = nifty["Gap_Pct"].clip(upper=0)

    nifty = nifty.dropna(subset=["Intraday_Pct", "Gap_Pct"])
    return nifty


def plot_intraday_and_gaps(nifty: pd.DataFrame):
    """Chart 1: Intraday movement bars + gap lines on secondary axis."""
    fig, ax1 = plt.subplots(figsize=(20, 8))

    ax1.bar(
        nifty.index, nifty["Intraday_Pct"],
        color="steelblue", alpha=0.6, label="Intraday Movement %", width=1.0,
    )
    ax1.set_ylabel("Intraday Movement %", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(nifty.index, nifty["Abs_Gap_Pct"], color="orange", linewidth=0.5, alpha=0.8, label="Abs Gap %")
    ax2.plot(nifty.index, nifty["Pos_Gap_Pct"], color="green", linewidth=0.5, alpha=0.8, label="Positive Gap %")
    ax2.plot(nifty.index, nifty["Neg_Gap_Pct"], color="red", linewidth=0.5, alpha=0.8, label="Negative Gap %")
    ax2.set_ylabel("Gap %", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    plt.title("NIFTY 50 — Intraday Movement & Gap Analysis")
    plt.tight_layout()
    fig.savefig("nifty_intraday_and_gaps.png", dpi=150)
    print("Saved nifty_intraday_and_gaps.png")


def plot_intraday_vs_vix(nifty: pd.DataFrame, vix: pd.DataFrame):
    """Chart 2: Intraday movement bars + India VIX on secondary axis."""
    vix_aligned = vix["Close"].reindex(nifty.index)

    fig, ax1 = plt.subplots(figsize=(20, 8))

    ax1.bar(
        nifty.index, nifty["Intraday_Pct"],
        color="steelblue", alpha=0.6, label="Intraday Movement %", width=1.0,
    )
    ax1.set_ylabel("Intraday Movement %", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(vix_aligned.index, vix_aligned.values, color="crimson", linewidth=0.7, alpha=0.8, label="India VIX")
    ax2.set_ylabel("India VIX", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    plt.title("NIFTY 50 — Intraday Movement vs India VIX")
    plt.tight_layout()
    fig.savefig("nifty_intraday_vs_vix.png", dpi=150)
    print("Saved nifty_intraday_vs_vix.png")


def main():
    nifty = download_and_cache("^NSEI", NIFTY_CSV)
    vix = download_and_cache("^INDIAVIX", VIX_CSV)

    nifty = compute_metrics(nifty)

    plot_intraday_and_gaps(nifty)
    plot_intraday_vs_vix(nifty, vix)

    plt.show()


if __name__ == "__main__":
    main()
