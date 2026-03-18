"""
使用 NeuralProphet 讀取已清理的 GOOG 股價資料，進行訓練並預測未來 30 天。

輸入：
  - data/google_stock.csv（欄位：ds, y）

輸出：
  - forecast.png     ：預測曲線圖
  - components.png   ：成分分解圖（趨勢/季節性等）
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import warnings
import logging

# 🤫 把煩人的警告訊息全部靜音
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 將 NeuralProphet 與 PyTorch Lightning 的日誌等級調高，只顯示錯誤
logging.getLogger("NP").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# 在無 GUI 的環境（例如遠端/CI）儲存圖片時，建議使用 Agg backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuralprophet import NeuralProphet  # noqa: E402

# PyTorch 2.6 起 torch.load 預設 weights_only=True，某些框架（如 Lightning 的 lr_finder）在
# 還原暫存 checkpoint 時，可能因「安全反序列化」限制而失敗，拋出 UnpicklingError。
# 這裡用 allowlist 方式將 NeuralProphet 的設定類別加入安全名單（你信任本機套件來源時才適用）。
try:  # pragma: no cover
    import torch  # type: ignore

    from neuralprophet.configure import ConfigSeasonality  # type: ignore

    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([ConfigSeasonality])
except Exception:
    # 若 torch 不存在或版本不支援 add_safe_globals，就略過；後續仍會用「關閉 lr_finder」避免踩雷
    pass


def main() -> None:
    # 專案根目錄：此檔位於 src/，所以往上一層是專案根
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "google_stock.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{data_path}。請先執行 `python src/fetch_data.py` 產生資料。"
        )

    # 讀取資料
    df = pd.read_csv(data_path)

    # 確保 ds 欄位為 datetime，並按照日期排序
    if "ds" not in df.columns or "y" not in df.columns:
        raise RuntimeError(f"資料欄位不如預期：{list(df.columns)}（需要 ds 與 y）")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    # 初始化 NeuralProphet 模型
    #
    # 你遇到的錯誤是 Lightning 的 learning rate finder 在還原暫存 checkpoint 時，
    # 觸發 PyTorch 2.6 的安全反序列化限制而失敗。
    # 最直接且穩定的做法：明確指定 learning_rate，讓 NeuralProphet 不需要啟動 lr_finder。
    model = NeuralProphet(learning_rate=0.01)

    # 訓練
    # NeuralProphet 需要指定資料頻率（此處為日頻率 D）
    _metrics = model.fit(df, freq="D")

    # 產生未來 30 天的資料框
    # n_historic_predictions=True：會在 forecast 中包含歷史區間的預測，方便畫圖對照
    future = model.make_future_dataframe(df, periods=30, n_historic_predictions=True)
    forecast = model.predict(future)

    # 1) 預測圖（Forecast plot）
    model.plot(forecast, plotting_backend="matplotlib")
    forecast_path = project_root / "forecast.png"
    # 直接儲存當前的畫布
    plt.savefig(forecast_path, dpi=150, bbox_inches="tight") 
    plt.close()
    
    # 2) 成分圖（Components plot）
    model.plot_components(forecast, plotting_backend="matplotlib")
    components_path = project_root / "components.png"
    # 直接儲存當前的畫布
    plt.savefig(components_path, dpi=150, bbox_inches="tight") 
    plt.close()

    print(f"✅ 訓練與預測完成：已輸出 {forecast_path} 與 {components_path}")


if __name__ == "__main__":
    main()

