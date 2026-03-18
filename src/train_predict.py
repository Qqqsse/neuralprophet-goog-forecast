"""
使用 NeuralProphet 預測 GOOG 股價（優化版本）

改進重點：
- 加入 autoregression (n_lags)
- 移除不合理 seasonality（yearly/daily）
- 加入 regularization 防止 overfitting
- 加入 train/test split 做真實評估
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import warnings
import logging

# 🤫 關閉警告
warnings.filterwarnings("ignore")

logging.getLogger("NP").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuralprophet import NeuralProphet  # noqa: E402


def main() -> None:
    # 專案根目錄
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "google_stock.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"找不到資料檔：{data_path}")

    # =========================
    # 📥 讀取與清理資料
    # =========================
    df = pd.read_csv(data_path)

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds").reset_index(drop=True)

    # =========================
    # ✂️ Train / Test Split
    # =========================
    split = int(len(df) * 0.8)
    train_df = df[:split]
    test_df = df[split:]

    print(f"📊 Train size: {len(train_df)} | Test size: {len(test_df)}")

    # =========================
    # 🤖 建立模型（優化版）
    # =========================
    model = NeuralProphet(
        # ⭐ 核心：自回歸
        n_lags=20,

        # 預測步長
        n_forecasts=1,

        # ❌ 關閉不合理 seasonality
        yearly_seasonality=False,
        daily_seasonality=False,
        weekly_seasonality=True,

        # ⭐ 防止 trend / season 爆炸
        trend_reg=1.0,
        seasonality_reg=1.0,

        # ⭐ 更穩定
        learning_rate=0.003,
        epochs=50,

        # ⭐ 正規化（避免 scale 問題）
        normalize="standardize",

        # ⭐ 金融資料常用
        seasonality_mode="multiplicative",
    )

    # =========================
    # 🏋️ 訓練模型
    # =========================
    metrics = model.fit(
        train_df,
        freq="B",
        progress="none"
    )

    # =========================
    # 🔮 預測（包含 test）
    # =========================
    future = model.make_future_dataframe(
        train_df,
        periods=len(test_df),
        n_historic_predictions=True
    )

    forecast = model.predict(future)

    # =========================
    # 📊 匯出預測資料
    # =========================
    csv_export_path = project_root / "data" / "forecast_results_v2.csv"
    forecast.to_csv(csv_export_path, index=False, encoding="utf-8-sig")

    print(f"📁 預測資料已輸出：{csv_export_path}")

    # =========================
    # 📈 畫圖（Forecast）
    # =========================
    model.plot(forecast, plotting_backend="matplotlib")
    forecast_path = project_root / "forecast_v2.png"
    plt.savefig(forecast_path, dpi=150, bbox_inches="tight")
    plt.close()

    # =========================
    # 📊 畫圖（Components）
    # =========================
    model.plot_components(forecast, plotting_backend="matplotlib")
    components_path = project_root / "components_v2.png"
    plt.savefig(components_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 圖表輸出完成")

    # =========================
    # 🧪 評估（修正版）
    # =========================

    # 只取預測結果中「有真實 y 的部分」
    forecast_valid = forecast.dropna(subset=["y"])

    # 只保留 test 區間
    forecast_test = forecast_valid.iloc[-len(test_df):]

    mae = (forecast_test["y"] - forecast_test["yhat1"]).abs().mean()
    rmse = ((forecast_test["y"] - forecast_test["yhat1"])**2).mean()**0.5
    mape = ((forecast_test["y"] - forecast_test["yhat1"]).abs() / forecast_test["y"]).mean() * 100

    print("\n📊 Test Evaluation:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")


if __name__ == "__main__":
    main()