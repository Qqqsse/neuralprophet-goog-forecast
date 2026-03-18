"""
抓取 Google（GOOG）過去 5 年股價，並清理成 NeuralProphet 需要的格式。
輸出：
  - data/google_stock.csv
欄位格式（NeuralProphet 期待）：
  - ds: 日期（datetime，且不含時區資訊）
  - y : 目標值（此處使用收盤價 Close）
"""
from __future__ import annotations  # 啟用 PEP 563，讓型別提示可以用字串形式延遲解析（相容舊版 Python）
from pathlib import Path  # 匯入 Path，用於跨平台的檔案路徑操作
import pandas as pd  # 匯入 pandas，用於資料處理與 DataFrame 操作
import yfinance as yf  # 匯入 yfinance，用於從 Yahoo Finance 抓取股價資料

def fetch_google_stock_5y() -> pd.DataFrame:  # 定義函式，回傳型別為 pd.DataFrame
    """
    使用 yfinance 抓取 GOOG 過去 5 年的日線資料，並整理為 ds/y 欄位。
    注意事項：
    - yfinance 回傳的日期通常在 index（DatetimeIndex）中，可能含時區。
    - NeuralProphet 需要欄位名稱 ds（datetime）與 y（數值）。
    """
    # 下載過去 5 年的日線股價資料
    # progress=False：避免在終端一直輸出進度列（較乾淨）
    raw = yf.download(  # 呼叫 yfinance 的 download 函式，將結果存入 raw
        tickers="GOOG",       # 指定要下載的股票代號為 GOOG（Google）
        period="5y",          # 抓取範圍為過去 5 年
        interval="1d",        # 資料頻率為每日（日線）
        auto_adjust=False,    # 不自動調整股價（保留原始 Close 欄位）
        progress=False,       # 關閉下載進度條，讓終端輸出更乾淨
    )
    if raw is None or raw.empty:  # 若 raw 為 None 或空 DataFrame，代表下載失敗
        raise RuntimeError("yfinance 未抓到任何資料，請確認網路連線或代號是否正確（GOOG）。")  # 拋出執行期錯誤並說明原因

    # yfinance 常見欄位：Open, High, Low, Close, Adj Close, Volume
    #
    # 重要：不同 yfinance/參數組合下，欄位可能是單層欄位或 MultiIndex（例如 (Close, GOOG)）。
    # 若是 MultiIndex，我們會把它攤平成字串欄位，並優先找出 Close 對應的那一欄。
    df = raw.reset_index()  # 將 DatetimeIndex 從索引轉為一般欄位（通常變成 "Date" 欄）

    # 嘗試找出日期欄位名稱（常見為 Date；某些情況可能是 Datetime）
    date_col = None  # 初始化日期欄位名稱變數為 None
    for candidate in ("Date", "Datetime"):  # 逐一嘗試常見的日期欄位名稱
        if candidate in df.columns:  # 若該候選名稱存在於 DataFrame 的欄位中
            date_col = candidate  # 將找到的欄位名稱記錄下來
            break  # 找到後立即停止迴圈

    if date_col is None:  # 若兩個候選名稱都找不到，代表欄位結構非預期
        raise RuntimeError(f"找不到日期欄位（預期 Date/Datetime），目前欄位：{list(df.columns)}")  # 拋出錯誤並列出現有欄位以供除錯

    # 處理 Close 欄位：可能是單欄（Series），也可能是多欄（DataFrame；例如 MultiIndex 展開後）
    close_series = None  # 初始化 close_series 變數為 None，稍後存放收盤價的一維序列
    if "Close" in df.columns:  # 若 "Close" 欄位直接存在（單層欄位的情況）
        close_val = df["Close"]  # 取出 Close 欄位，可能是 Series 或 DataFrame
        # 如果 Close 不是一維 Series（例如 DataFrame 多欄），就取第一欄作為目標值
        if isinstance(close_val, pd.DataFrame):  # 若 Close 取出來是 DataFrame（MultiIndex 展開後有多個子欄）
            if close_val.shape[1] < 1:  # 若子欄數量為 0，代表結構異常
                raise RuntimeError("Close 欄位存在但沒有任何子欄位。")  # 拋出錯誤提示
            close_series = close_val.iloc[:, 0]  # 取第一個子欄作為收盤價序列
        else:  # Close 取出來就是一般的一維 Series
            close_series = close_val  # 直接使用該 Series 作為收盤價
    else:  # "Close" 欄位不存在，可能仍是 MultiIndex 結構
        # 若欄位是 MultiIndex，reset_index() 後仍可能保留 MultiIndex columns
        # 這裡做更保守的處理：把欄位名稱攤平後再找含 Close 的欄位
        if isinstance(df.columns, pd.MultiIndex):  # 若欄位確實是 MultiIndex 結構
            df.columns = [  # 將 MultiIndex 欄位攤平為單層字串欄位名稱
                "_".join([str(x) for x in tup if x not in (None, "")]).strip("_")  # 將 tuple 元素串接成字串，去除空值與底線
                for tup in df.columns.to_list()  # 遍歷每個 MultiIndex tuple
            ]
        close_candidates = [c for c in df.columns if str(c).startswith("Close")]  # 找出所有以 "Close" 開頭的欄位名稱
        if not close_candidates:  # 若找不到任何以 "Close" 開頭的欄位
            raise RuntimeError(f"找不到 Close 欄位，目前欄位：{list(df.columns)}")  # 拋出錯誤並列出欄位供除錯
        close_series = df[close_candidates[0]]  # 取第一個符合的 Close 欄位作為收盤價序列

    # 轉換日期欄位為 ds，並移除時區資訊（tz_localize(None)）
    # - 若 ds 本身帶 tz，tz_localize(None) 會去除時區
    # - 若 ds 不帶 tz，tz_localize(None) 會拋錯，因此先做條件處理更安全
    ds = pd.to_datetime(df[date_col], errors="coerce")  # 將日期欄位轉換為 datetime 型別，無法解析的值變為 NaT
    if getattr(ds.dt, "tz", None) is not None:  # 若 datetime 序列帶有時區資訊
        ds = ds.dt.tz_localize(None)  # 移除時區資訊，使其成為 naive datetime（NeuralProphet 的要求）

    # 組成 NeuralProphet 需要的兩欄：ds, y
    out = pd.DataFrame(  # 建立新的 DataFrame，只保留 ds 與 y 兩欄
        {
            "ds": ds,  # 日期欄位（已去除時區的 datetime）
            "y": pd.to_numeric(close_series, errors="coerce"),  # 收盤價轉為數值型別，無法轉換的值變為 NaN
        }
    )

    # 移除缺失值（例如最後幾天可能尚未完整、或因轉型失敗而出現 NaN）
    out = out.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)  # 刪除 ds 或 y 為 NaN 的列，按日期排序，並重設索引
    return out  # 回傳整理完成的 DataFrame


def main() -> None:  # 定義主程式入口函式，無回傳值
    # 專案根目錄：此檔位於 src/，所以往上一層是專案根
    project_root = Path(__file__).resolve().parents[1]  # 取得此腳本的絕對路徑，並往上兩層取得專案根目錄
    output_path = project_root / "data" / "google_stock.csv"  # 組合出輸出 CSV 檔案的完整路徑
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 若 data/ 目錄不存在則自動建立（包含所有中間層目錄）
    df = fetch_google_stock_5y()  # 呼叫函式抓取並整理 Google 股價資料
    # 儲存成 CSV（不寫 index，避免多一欄無用欄位）
    df.to_csv(output_path, index=False, encoding="utf-8")  # 將 DataFrame 儲存為 UTF-8 編碼的 CSV 檔案，不輸出列索引
    print(f"✅ 資料抓取與清理完成：已輸出 {output_path}（筆數：{len(df)}）")  # 印出成功訊息，包含輸出路徑與資料筆數

if __name__ == "__main__":  # 確認此檔案是被直接執行（而非被 import）
    main()  # 呼叫主函式，啟動整個資料抓取流程