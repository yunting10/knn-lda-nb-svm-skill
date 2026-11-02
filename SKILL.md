---
name: knn-lda-nb-svm-skill
description: 以繁體中文分析使用者上傳的 CSV／Excel 資料，依樣本數、維度、類別比例與資料特性建議 KNN、LDA、QDA、單純貝氏、支援向量機（SVM）並執行區別與分類分析，Y 為連續時以 KNN 迴歸與 SVR 預測；完成經確認的前處理、EDA、交叉驗證調參與測試評估，產出 HTML、Markdown 與 Quarto 成果報告（需要時另產生 PDF）並附完整 Python 程式碼。適用於分類問題，以及連續型 Y 的預測。
---

# KNN、區別分析、單純貝氏與支援向量機分析

## 任務與交付

依使用者的研究問題與同意的方案，完成資料檢查、前處理、EDA、方法建議與可行性檢查、調參、評估與結果解讀。

最終交付：
- `report.html`（**主要報告**；圖片內嵌、完整程式碼預設收合，單一檔案即可開啟）
- `report.md`（圖片內嵌）
- `report.qmd`（完整程式碼放在不執行、預設折疊的區塊）
- `report.pdf`：**選用**，使用者需要且環境可行時才有（config 的 `pdf` 設為 `"on"`）

各份報告內容一致，最後都附完整 Python 程式碼（由 `finalize` 自動附上）。

**所有數值與圖表都必須由 `scripts/run_analysis.py` 實際執行產生。** 不得自行撰寫臨時程式取代它來產生報告中的數字，也不得直接繪製、描述或修改圖表內容。若使用者的需求超出程式能力，先說明限制；經同意擴充程式後，擴充的程式同樣附在報告中。

## 參考資料

依當下階段讀取，不需一次全部載入：

| 檔案 | 何時讀 |
|:--|:--|
| [references/decision-tree.md](references/decision-tree.md) | 說明方法建議與可行性時：流程圖、數值門檻、操作原則 |
| [references/methods.md](references/methods.md) | 撰寫方法說明、公式附錄、解讀模型結果時（只讀本次使用的方法） |
| [references/report-spec.md](references/report-spec.md) | 撰寫報告前：章節、寫作與重現規定 |

`docs/` 是給人閱讀的教學文件，分析時不需要讀。

## 適用範圍

- **適用：**Y 為類別（二元或多類別）的分類，以及 Y 為連續的預測（KNN 迴歸、SVR）；目的是預測。
- **不適用：**解釋變數效果或因果推論、重複測量或群聚資料、時間序列預測、存活資料。遇到時說明原因並停止，不自行改用其他方法。

## 互動原則

- 使用繁體中文；統計術語不另外解釋，但每個結果都要給諮詢者看得懂的結論。
- **每次只詢問並確認一個問題**，收到回答後才進入下一個；已提供或已同意的資訊不重複詢問。
- 需要使用者選擇時（前處理、候選方法、評估設定、門檻），若環境有選項式提問工具就以選單呈現，選項附「建議」標記與一句理由；沒有就用編號清單請使用者回覆編號。仍維持每次只問一個問題。
- 可直接執行檢查與提出建議，但**未經同意不得刪除資料、插補、排除變數、更換方法或擴大參數搜尋**。
- 只回報影響決策的重要資訊，不重複教學。

## 工作流程

程式位於 `scripts/run_analysis.py`。以下指令在工作資料夾執行；`config.json` 由你依使用者同意的內容建立與更新。

### 一、確認研究問題與資料

確認資料檔（與工作表）、背景、觀測單位、Y、X、分析目的，以及是否有時間順序或重複測量。未知資訊標示「未提供」，不自行補寫。不得把 ID、日期或會洩漏答案的欄位放進 X。目的或資料結構屬於「不適用」範圍時，說明原因並停止。

### 二、資料檢查與前處理

```bash
python scripts/run_analysis.py inspect --data 資料.csv --y Y欄位 [--x X1 X2 ...] --out output
```

產生各變數的型態、缺失筆數與比例、建議處理方式、Y 的分佈，以及 `output/config_template.json`。

- 向使用者報告摘要，並**逐項確認**前處理：
  - 數值型 X 缺失比例 ≤ 20%：建議中位數插補。
  - 數值型 X 缺失比例 > 20%：列出刪除後的影響，經同意才刪除，並記錄確認內容。
  - 類別型 X：確認以眾數（`most_frequent`）或「缺失」類別（`missing_category`）處理。
  - Y 缺失：確認是否排除這些資料列，不插補 Y。
  - 「數值但不同值 ≤ 5」的欄位：確認是否為類別編碼。
  - 二元 Y：確認正類。
- 依確認結果把 `config_template.json` 另存為 `config.json` 並填寫（欄位說明見文末）。

### 三、資料概況與方法建議

```bash
python scripts/run_analysis.py profile --config config.json
```

以訓練資料計算統計診斷：n、原始變數數、編碼後進入模型的欄位數 p、n/p、各類樣本數與比例、每折最小類別筆數、特徵間最大相關、各類散佈（只當參考）；再依 `references/decision-tree.md` 輸出三張表：**建議方法**（附理由）、**其他可行方法**、**不可行方法**（附原因）。建議方法同時預設填入 `output/config_template.json` 的 `candidates`。此步驟不配適任何模型。

- 以「判斷依據 → 數值 → 規則 → 影響」向使用者說明建議方法與理由、其他可行方法、不可行方法與原因。
- 再以選單（可複選）請使用者確認候選方法：**建議方法預設勾選**，**其他可行方法不勾選但可加選**，**不可行方法不列為選項**。比較基準（分類：多數類別、logistic 迴歸；迴歸：線性迴歸）由程式自動加入，不出現在選單中。
- 使用者確認後填入 `config.json` 的 `candidates`；填入不可行方法時，`run` 會停止並印出原因。可用方法代碼：

| 任務 | 代碼 |
|:--|:--|
| 分類 | `knn`、`lda`、`lda_shrink`、`qda`、`nb_gaussian`、`nb_mixed`（連續＋類別）、`nb_multinomial`（文字）、`svm_linear`、`svm_rbf`；`logistic` 為比較基準，自動加入 |
| 迴歸 | `knn_reg`、`svr`；`linear_reg` 為比較基準，自動加入 |

### 四、確認評估設定

逐項確認（使用者接受預設值時直接沿用）：測試集比例（預設 0.25）、交叉驗證折數（預設 5）、選模指標（預設自動：平衡資料用準確率，不平衡二元用 AUC，不平衡多類別用平衡準確率）、二元分類門檻（預設 0.5）、隨機種子（預設 2026）、稀有類別時是否使用 `class_weight: "balanced"`。

### 五、執行分析

```bash
python scripts/run_analysis.py run --config config.json
```

產生 `output/figures/`、`output/tables/`、`output/results.json` 與 `output/report_draft.md`。

- 檢查終端機印出的模型比較表。
- 某方法所有參數組合都無法估計時，程式會略過它並記錄原因；向使用者說明，不更改資料來繞過。
- 調參表顯示「最佳值在網格邊緣」時，向使用者提議擴大網格（`grids`），同意後重跑。
- 稀有類別時，依「不同門檻下的表現」表（訓練資料交叉驗證預測）向使用者提議門檻；同意後更新 `threshold` 並重跑。
- 比較用的 logistic／線性迴歸若不低於最終模型，須告知使用者，並在結果與限制中說明線性模型的預測表現與本 skill 的方法相當。
- 程式報錯時如實告知，不更改資料或方法來繞過錯誤。

### 六、撰寫報告

1. 先讀 `references/report-spec.md`。
2. 把 `report_draft.md` 另存為 `output/report.md`，把每一處 `【待填：…】` 換成內容：
   - 數字一律取自 `results.json` 或草稿中的表格，不重新計算、不四捨五入成不同位數。
   - 每張圖、每個表之後寫一到三句給諮詢者的解讀，說明回答了研究問題的哪一部分。
   - p 值寫出虛無假設、顯著水準與結論；「不顯著」寫成「沒有足夠證據顯示有差異」。
   - 方法公式只列本次使用的方法，取自 `references/methods.md`。
   - 不作因果宣稱；未提供誤判成本時，不宣稱模型可直接上線。
3. 不刪除程式產生的表格與圖片。

### 七、完成交付

```bash
python scripts/run_analysis.py finalize --config config.json --report output/report.md
```

檢查是否還有未填寫處、附上完整程式碼與 config，輸出 `report.md`（圖片內嵌）、`report.qmd` 與 `report.html`。config 的 `pdf` 預設為 `"off"`，完全不檢查 PDF 環境；**使用者表示需要 PDF 時**，才把 `pdf` 設為 `"on"` 再執行，此時會嘗試產生 `report.pdf`（需要 playwright＋Chromium，或含 xeCJK 的 TeX）。最後印出摘要：已產生的檔案，以及略過的檔案與原因。

- **成功條件是 `report.html` 產生成功**（結束代碼 0）；只有這樣才可宣稱已交付。
- PDF 被略過不算失敗：告知使用者摘要中的原因，並說明可用瀏覽器開啟 `report.html` → 列印 → 另存為 PDF。
- 在 Claude 雲端環境執行時，提醒使用者下載 `report.html`（對話結束後環境會被回收）。

交付前確認：`report.html` 的中文、圖、表、數學式與折疊的程式碼都正常顯示；有 PDF 時確認內容未截斷；各份報告內容一致。

## config.json 欄位

| 欄位 | 說明 |
|:--|:--|
| `data_path`、`sheet`、`encoding` | 資料路徑、Excel 工作表、CSV 編碼（中文 Windows 常為 `cp950` 或 `utf-8-sig`） |
| `na_values` | 讀取時視為缺失的字串；預設 `["", "NA", "N/A", "NaN", "?"]` |
| `y`、`x` | 反應變數與解釋變數 |
| `task` | `classification` 或 `regression` |
| `positive_class` | 二元分類的正類 |
| `categorical` | 強制視為類別的欄位（例如數字編碼的類別） |
| `text_column` | 文字分類時的文字欄位（此時只用這一欄） |
| `drop_columns` | 經同意刪除的變數：`[{"name": …, "reason": …, "confirmed": "使用者同意的時間與內容"}]` |
| `impute_categorical` | `most_frequent` 或 `missing_category`（數值型固定為中位數插補） |
| `drop_missing_y` | 是否排除 Y 缺失的資料列（須經同意） |
| `test_size`、`cv_folds`、`seed`、`time_order` | 切分比例、折數、隨機種子、是否依時間順序切分 |
| `metric` | 選模指標；`null` 表示自動 |
| `candidates` | 經同意的候選方法代碼（`profile` 預設填入建議方法；比較基準不必填） |
| `threshold`、`class_weight` | 二元分類門檻；`"balanced"` 或 `null` |
| `grids` | 覆寫參數網格，例如 `{"svm_rbf": {"C": [1, 4, 16], "gamma": [0.01, 0.1]}}` |
| `max_n_kernel` | 訓練資料筆數超過此值時核 SVM（`svm_rbf`）與 SVR 不可行，`svm_linear` 改用 `LinearSVC`；預設 20000 |
| `pdf` | `"off"`（預設：只產生 HTML、MD、QMD，完全不檢查 PDF 環境）或 `"on"`（使用者需要 PDF 時，嘗試產生 `report.pdf`） |
| `out_dir` | 輸出資料夾 |
| `report` | `title`、`background`、`unit`、`goal`（報告第 1 章） |

## Jupyter 使用

`run_analysis.py` 可直接貼進 Jupyter 單一儲存格：修改檔案開頭「設定」區的 `MODE`、`CONFIG_PATH` 等常數後執行；圖片會在存檔後顯示。
