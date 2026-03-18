"""
使用 NeuralProphet 預測 GOOG 股價（優化版本）
改進重點：
- 加入 autoregression (n_lags)
- 移除不合理 seasonality（yearly/daily）
- 加入 regularization 防止 overfitting
- 加入 train/test split 做真實評估
"""
from __future__ import annotations  # 啟用 PEP 563，讓型別提示可以用字串形式延遲解析（相容舊版 Python）
from pathlib import Path  # 匯入 Path，用於跨平台的檔案路徑操作
import pandas as pd  # 匯入 pandas，用於資料讀取、清理與 DataFrame 操作
import warnings  # 匯入 warnings 模組，用於控制警告訊息的顯示
import logging  # 匯入 logging 模組，用於控制各套件的日誌輸出等級

# 🤫 關閉警告
warnings.filterwarnings("ignore")  # 全域忽略所有 Python 警告，避免訓練時干擾輸出
logging.getLogger("NP").setLevel(logging.ERROR)  # 將 NeuralProphet 的日誌等級設為 ERROR，只顯示錯誤
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)  # 將 PyTorch Lightning 的日誌等級設為 ERROR，抑制訓練過程輸出

# Matplotlib backend
import matplotlib  # 匯入 matplotlib 主模組，用於設定繪圖後端
matplotlib.use("Agg")  # 切換為非互動式後端 Agg，讓圖表可在無視窗環境（如伺服器）中存檔
import matplotlib.pyplot as plt  # noqa: E402  # 匯入 pyplot，用於存檔與關閉圖表（noqa 抑制 import 順序警告）
from neuralprophet import NeuralProphet  # noqa: E402  # 匯入 NeuralProphet 模型（noqa 抑制 import 順序警告）

def main() -> None:  # 定義主程式入口函式，無回傳值
    # 專案根目錄
    project_root = Path(__file__).resolve().parents[1]  # 取得此腳本的絕對路徑，並往上兩層取得專案根目錄
    data_path = project_root / "data" / "google_stock.csv"  # 組合出輸入 CSV 檔案的完整路徑

    if not data_path.exists():  # 若資料檔案不存在於指定路徑
        raise FileNotFoundError(f"找不到資料檔：{data_path}")  # 拋出錯誤並顯示找不到的路徑

    # =========================
    # 📥 讀取與清理資料
    # =========================
    df = pd.read_csv(data_path)  # 從 CSV 檔案讀取股價資料到 DataFrame
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")  # 將 ds 欄位轉為 datetime 型別，無法解析的值變為 NaT
    df["y"] = pd.to_numeric(df["y"], errors="coerce")  # 將 y 欄位轉為數值型別，無法轉換的值變為 NaN
    df = df.dropna(subset=["ds", "y"])  # 移除 ds 或 y 欄位為缺失值的列
    df = df.sort_values("ds").reset_index(drop=True)  # 依日期由舊到新排序，並重設列索引

    # =========================
    # ✂️ Train / Test Split
    # =========================
    split = int(len(df) * 0.8)  # 計算訓練集的截止索引（取前 80% 的資料）
    train_df = df[:split]  # 切出訓練集（前 80%）
    test_df = df[split:]  # 切出測試集（後 20%）
    print(f"📊 Train size: {len(train_df)} | Test size: {len(test_df)}")  # 印出訓練與測試集的資料筆數

    # =========================
    # 🤖 建立模型（優化版）
    # =========================
    model = NeuralProphet(  # 初始化 NeuralProphet 模型，並設定以下超參數
        # ⭐ 核心：自回歸
        n_lags=20,  # 使用前 20 個時間步的歷史值作為自回歸輸入特徵
        # 預測步長
        n_forecasts=1,  # 每次只預測未來 1 步（下一個交易日）
        # ❌ 關閉不合理 seasonality
        yearly_seasonality=False,  # 關閉年週期性（5 年資料量不足以學到穩定的年季節性）
        daily_seasonality=False,  # 關閉日週期性（日線資料每天只有一筆，無法學習日內規律）
        weekly_seasonality=True,  # 保留週週期性（股市有明顯的週一到週五交易規律）
        # ⭐ 防止 trend / season 爆炸
        trend_reg=1.0,  # 對趨勢項加入 L1 正規化，防止趨勢過度擬合
        seasonality_reg=1.0,  # 對季節性項加入正規化，避免季節波動過大
        # ⭐ 更穩定
        learning_rate=0.003,  # 設定較小的學習率，使訓練過程更穩定、不易震盪
        epochs=50,  # 訓練 50 個 epoch（完整遍歷訓練資料 50 次）
        # ⭐ 正規化（避免 scale 問題）
        normalize="standardize",  # 對輸入資料進行標準化（減均值除標準差），改善數值穩定性
        # ⭐ 金融資料常用
        seasonality_mode="multiplicative",  # 使用乘法季節性，適合股價等隨趨勢等比變動的資料
    )

    # =========================
    # 🏋️ 訓練模型
    # =========================
    metrics = model.fit(  # 用訓練集資料訓練模型，並將訓練過程的指標存入 metrics
        train_df,  # 傳入訓練集 DataFrame（含 ds 與 y 欄位）
        freq="B",  # 指定資料頻率為工作日（Business day），讓模型跳過週末
        progress="none"  # 關閉訓練進度條，保持終端輸出乾淨
    )

    # =========================
    # 🔮 預測（包含 test）
    # =========================
    future = model.make_future_dataframe(  # 建立未來的日期 DataFrame，供預測使用
        train_df,  # 以訓練集為基準延伸時間軸
        periods=len(test_df),  # 向未來延伸的步數等於測試集長度，對齊測試區間
        n_historic_predictions=True  # 同時保留歷史預測值，方便與訓練集真實值比對
    )
    forecast = model.predict(future)  # 對 future DataFrame 進行預測，結果存入 forecast

    # =========================
    # 📊 匯出預測資料
    # =========================
    csv_export_path = project_root / "data" / "forecast_results_v2.csv"  # 組合出預測結果 CSV 的輸出路徑
    forecast.to_csv(csv_export_path, index=False, encoding="utf-8-sig")  # 將預測結果儲存為 CSV（utf-8-sig 確保 Excel 開啟時中文不亂碼）
    print(f"📁 預測資料已輸出：{csv_export_path}")  # 印出輸出成功的路徑提示

    # =========================
    # 📈 畫圖（Forecast）
    # =========================
    model.plot(forecast, plotting_backend="matplotlib")  # 使用 matplotlib 繪製整體預測走勢圖
    forecast_path = project_root / "forecast_v2.png"  # 組合出預測圖的輸出路徑
    plt.savefig(forecast_path, dpi=150, bbox_inches="tight")  # 以 150 DPI 儲存圖片，bbox_inches="tight" 避免標籤被裁切
    plt.close()  # 關閉目前圖表，釋放記憶體，避免下一張圖疊加

    # =========================
    # 📊 畫圖（Components）
    # =========================
    # 只取最近 200 天
    recent_forecast = forecast.tail(100)  # 取預測結果的最後 100 列，聚焦於近期趨勢與季節性分解
    model.plot_components(recent_forecast, plotting_backend="matplotlib")  # 繪製各分解元件圖（趨勢、季節性、自回歸等）
    components_path = project_root / "components_v2.png"  # 組合出元件分解圖的輸出路徑
    plt.savefig(components_path, dpi=150, bbox_inches="tight")  # 以 150 DPI 儲存元件圖，確保邊界完整
    plt.close()  # 關閉圖表，釋放記憶體
    print(f"✅ 圖表輸出完成")  # 印出圖表輸出完成的提示訊息

    # =========================
    # 🧪 評估（修正版）
    # =========================
    # 只取預測結果中「有真實 y 的部分」
    forecast_valid = forecast.dropna(subset=["y"])  # 過濾掉 y 為 NaN 的列（純未來預測列沒有真實值）
    # 只保留 test 區間
    forecast_test = forecast_valid.iloc[-len(test_df):]  # 從有效預測中取最後 N 列，對應測試集區間

    mae = (forecast_test["y"] - forecast_test["yhat1"]).abs().mean()  # 計算平均絕對誤差（MAE）：預測值與真實值差的絕對值平均
    rmse = ((forecast_test["y"] - forecast_test["yhat1"])**2).mean()**0.5  # 計算均方根誤差（RMSE）：誤差平方的均值再開根號
    mape = ((forecast_test["y"] - forecast_test["yhat1"]).abs() / forecast_test["y"]).mean() * 100  # 計算平均絕對百分比誤差（MAPE）：誤差占真實值比例的百分比平均

    print("\n📊 Test Evaluation:")  # 印出評估結果的標題
    print(f"MAE  : {mae:.2f}")  # 印出 MAE，保留兩位小數
    print(f"RMSE : {rmse:.2f}")  # 印出 RMSE，保留兩位小數
    print(f"MAPE : {mape:.2f}%")  # 印出 MAPE，保留兩位小數並附上百分比符號

if __name__ == "__main__":  # 確認此檔案是被直接執行（而非被 import）
    main()  # 呼叫主函式，啟動整個預測流程