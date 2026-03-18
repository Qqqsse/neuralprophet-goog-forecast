"""
抓取 Google（GOOG）過去 5 年股價，並清理成 NeuralProphet 需要的格式。

輸出：
  - data/google_stock.csv

欄位格式（NeuralProphet 期待）：
  - ds: 日期（datetime，且不含時區資訊）
  - y : 目標值（此處使用收盤價 Close）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_google_stock_5y() -> pd.DataFrame:
    """
    使用 yfinance 抓取 GOOG 過去 5 年的日線資料，並整理為 ds/y 欄位。

    注意事項：
    - yfinance 回傳的日期通常在 index（DatetimeIndex）中，可能含時區。
    - NeuralProphet 需要欄位名稱 ds（datetime）與 y（數值）。
    """

    # 下載過去 5 年的日線股價資料
    # progress=False：避免在終端一直輸出進度列（較乾淨）
    raw = yf.download(
        tickers="GOOG",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError("yfinance 未抓到任何資料，請確認網路連線或代號是否正確（GOOG）。")

    # yfinance 常見欄位：Open, High, Low, Close, Adj Close, Volume
    #
    # 重要：不同 yfinance/參數組合下，欄位可能是單層欄位或 MultiIndex（例如 (Close, GOOG)）。
    # 若是 MultiIndex，我們會把它攤平成字串欄位，並優先找出 Close 對應的那一欄。
    df = raw.reset_index()

    # 嘗試找出日期欄位名稱（常見為 Date；某些情況可能是 Datetime）
    date_col = None
    for candidate in ("Date", "Datetime"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise RuntimeError(f"找不到日期欄位（預期 Date/Datetime），目前欄位：{list(df.columns)}")

    # 處理 Close 欄位：可能是單欄（Series），也可能是多欄（DataFrame；例如 MultiIndex 展開後）
    close_series = None

    if "Close" in df.columns:
        close_val = df["Close"]
        # 如果 Close 不是一維 Series（例如 DataFrame 多欄），就取第一欄作為目標值
        if isinstance(close_val, pd.DataFrame):
            if close_val.shape[1] < 1:
                raise RuntimeError("Close 欄位存在但沒有任何子欄位。")
            close_series = close_val.iloc[:, 0]
        else:
            close_series = close_val
    else:
        # 若欄位是 MultiIndex，reset_index() 後仍可能保留 MultiIndex columns
        # 這裡做更保守的處理：把欄位名稱攤平後再找含 Close 的欄位
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in tup if x not in (None, "")]).strip("_")
                for tup in df.columns.to_list()
            ]

        close_candidates = [c for c in df.columns if str(c).startswith("Close")]
        if not close_candidates:
            raise RuntimeError(f"找不到 Close 欄位，目前欄位：{list(df.columns)}")
        close_series = df[close_candidates[0]]

    # 轉換日期欄位為 ds，並移除時區資訊（tz_localize(None)）
    # - 若 ds 本身帶 tz，tz_localize(None) 會去除時區
    # - 若 ds 不帶 tz，tz_localize(None) 會拋錯，因此先做條件處理更安全
    ds = pd.to_datetime(df[date_col], errors="coerce")
    if getattr(ds.dt, "tz", None) is not None:
        ds = ds.dt.tz_localize(None)

    # 組成 NeuralProphet 需要的兩欄：ds, y
    out = pd.DataFrame(
        {
            "ds": ds,
            "y": pd.to_numeric(close_series, errors="coerce"),
        }
    )

    # 移除缺失值（例如最後幾天可能尚未完整、或因轉型失敗而出現 NaN）
    out = out.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    return out


def main() -> None:
    # 專案根目錄：此檔位於 src/，所以往上一層是專案根
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "google_stock.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_google_stock_5y()

    # 儲存成 CSV（不寫 index，避免多一欄無用欄位）
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✅ 資料抓取與清理完成：已輸出 {output_path}（筆數：{len(df)}）")


if __name__ == "__main__":
    main()

