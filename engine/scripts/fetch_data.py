"""Download the raw inputs for Mirage and write them into data/.

Two sources, both free and keyless.

1. Yahoo Finance chart endpoint, for daily OHLCV and split/dividend adjusted
   closes on a liquid US equity universe plus SPY as benchmark.
2. Kenneth French Data Library, for daily research factor returns. These serve
   as an outside reference for the momentum control, so the survival result does
   not rest only on prices we assembled ourselves.

Stooq was the original intended price source. As of August 2026 its CSV
endpoints sit behind a proof-of-work browser challenge, so it is not usable from
a script. That is recorded in data/README.md.

Run once, commit the output, and never touch the network again:

    python -m engine.scripts.fetch_data

The universe below is a hand-picked set of large caps that were already listed in
2005. That selection is survivorship biased by construction. It is fine for this
project, whose subject is the statistics of strategy selection rather than the
level of any one backtest, but the bias is real and is disclosed in the README.
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

from engine.config import BENCHMARK, DATA_DIR, PRICES_DIR

START = date(2005, 1, 1)

# French history reaches back to 1926. Only the window that overlaps the price
# panel is kept, which trims about 1.9 MB of unused rows out of the repo.
FRENCH_START = "2004-07-01"

UNIVERSE = [
    "AAPL", "ABT", "ADBE", "AMGN", "AXP", "BA", "BAC", "BMY", "C", "CAT",
    "COST", "CSCO", "CVX", "DE", "DIS", "DUK", "GILD", "GS", "HD", "HON",
    "IBM", "INTC", "JNJ", "JPM", "KO", "LMT", "LOW", "MCD", "MDT", "MMM",
    "MRK", "MSFT", "NKE", "ORCL", "PEP", "PFE", "PG", "QCOM", "SBUX", "SO",
    "T", "TGT", "TXN", "UNH", "UPS", "VZ", "WFC", "WMT", "XOM",
]

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mirage-research/0.1"

FRENCH_FILES = {
    "ff_factors_daily.csv": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    ),
    "ff_momentum_daily.csv": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_daily_CSV.zip"
    ),
}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def fetch_prices(symbol: str, start: date, end: date) -> list[dict]:
    """Pull one symbol's daily bars. Returns rows sorted by date, gaps dropped.

    Yahoo returns null entries on days where the exchange was open but the
    symbol did not trade. Those rows are dropped rather than filled, so the
    calendar alignment in data.py sees a genuine hole instead of a fake price.
    """
    url = CHART_URL.format(symbol=symbol)
    url += f"?period1={_epoch(start)}&period2={_epoch(end)}&interval=1d&events=div%2Csplit"
    payload = json.loads(_get(url))

    result = payload["chart"]["result"]
    if not result:
        raise RuntimeError(f"{symbol}: empty chart result")
    block = result[0]

    stamps = block["timestamp"]
    quote = block["indicators"]["quote"][0]
    adj = block["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])

    rows: list[dict] = []
    for i, ts in enumerate(stamps):
        o, h, l, c = quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i]
        v, a = quote["volume"][i], adj[i]
        if None in (o, h, l, c, v, a):
            continue
        rows.append(
            {
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
                "open": round(float(o), 6),
                "high": round(float(h), 6),
                "low": round(float(l), 6),
                "close": round(float(c), 6),
                "adj_close": round(float(a), 6),
                "volume": int(v),
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def write_price_csv(path: Path, rows: list[dict]) -> None:
    fields = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fetch_french(name: str, url: str, out_dir: Path) -> tuple[int, str, str]:
    """Download and flatten one French daily factor file.

    The published CSVs carry a prose header and a trailing annual block. Only the
    daily section is kept, dates are normalized to ISO, and percent units are
    converted to decimal returns.
    """
    with zipfile.ZipFile(io.BytesIO(_get(url))) as zf:
        raw = zf.read(zf.namelist()[0]).decode("latin-1")

    lines = [ln.strip() for ln in raw.splitlines()]
    header_idx = next(
        i for i, ln in enumerate(lines) if ln.replace(" ", "").lower().startswith("," )
        and any(tok in ln for tok in ("Mkt-RF", "Mom", "MOM"))
    )
    cols = [c.strip() for c in lines[header_idx].split(",")]
    cols[0] = "date"

    out_rows: list[list[str]] = []
    for ln in lines[header_idx + 1 :]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(cols) or len(parts[0]) != 8 or not parts[0].isdigit():
            continue
        iso = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}"
        if iso < FRENCH_START:
            continue
        vals = [f"{float(p) / 100.0:.8f}" for p in parts[1:]]
        out_rows.append([iso] + vals)

    path = out_dir / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([c.lower().replace("-", "_") for c in cols])
        writer.writerows(out_rows)
    return len(out_rows), out_rows[0][0], out_rows[-1][0]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today()

    symbols = [BENCHMARK] + UNIVERSE
    failures: list[str] = []
    for i, sym in enumerate(symbols, 1):
        try:
            rows = fetch_prices(sym, START, end)
        except (urllib.error.URLError, KeyError, RuntimeError, TimeoutError) as exc:
            print(f"[{i:>3}/{len(symbols)}] {sym:<6} FAILED  {exc}")
            failures.append(sym)
            time.sleep(2.0)
            continue
        write_price_csv(PRICES_DIR / f"{sym}.csv", rows)
        print(f"[{i:>3}/{len(symbols)}] {sym:<6} {len(rows):>5} rows  {rows[0]['date']} to {rows[-1]['date']}")
        time.sleep(0.6)

    for name, url in FRENCH_FILES.items():
        n, first, last = fetch_french(name, url, DATA_DIR)
        print(f"french  {name:<24} {n:>6} rows  {first} to {last}")

    if failures:
        print(f"\n{len(failures)} symbols failed: {', '.join(failures)}")
        print("Rerun to retry. Do not proceed with a partial universe silently.")
    else:
        print(f"\nAll {len(symbols)} symbols written to {PRICES_DIR}")


if __name__ == "__main__":
    main()
