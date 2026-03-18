# Google 股價預測（NeuralProphet）

本專案示範如何使用 `yfinance` 抓取 Google（`GOOG`）過去 5 年的歷史股價，並以 `NeuralProphet` 建立時間序列模型，預測未來 30 天的股價走勢，同時輸出預測圖與成分分解圖。

## 安裝

在專案根目錄（有 `requirements.txt` 的資料夾）執行：

> 注意：若你在 Windows 使用 **Python 3.13**，`neuralprophet` 的相依套件可能沒有對應的預編譯版本，導致 pip 嘗試編譯而失敗。
> 建議改用 **Python 3.10 / 3.11 / 3.12** 建立虛擬環境後再安裝。

```bash
pip install -r requirements.txt
```

## 下載與整理資料

執行後會產生 `data/google_stock.csv`（欄位 `ds`, `y`）：

```bash
python src/fetch_data.py
```

## 訓練與預測

執行後會在專案根目錄輸出兩張圖片：
- `forecast.png`（預測圖）
- `components.png`（成分分解圖）

```bash
python src/train_predict.py
```

