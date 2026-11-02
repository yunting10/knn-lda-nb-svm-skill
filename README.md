# KNN、區別分析、單純貝氏與支援向量機分析 Skill

讓 AI 助理依照固定流程分析一份上傳的資料：檢查資料、與使用者確認前處理、依資料特性建議優先比較的方法（KNN、LDA、QDA、單純貝氏、SVM／SVR）並剔除不可行的方法，以交叉驗證調參、在測試集上評估，最後產出 HTML（主要）、Markdown 與 Quarto 成果報告（需要時另產生 PDF），並附上可重現結果的完整 Python 程式碼。

所有數值與圖表都由同一支程式 `scripts/run_analysis.py` 產生，AI 只負責和使用者溝通、填寫設定與撰寫解讀；同一份資料、同樣的設定與套件版本，會得到相同的數值與圖表。

## 資料夾

```
knn-lda-nb-svm-skill/
├── SKILL.md                    給 AI 的流程與規則
├── README.md                   本說明
├── requirements.txt            Python 套件
├── references/                 給 AI 查閱的參考資料
│   ├── decision-tree.md        方法建議的流程圖（mermaid）、建議依據與可行性條件、操作原則
│   ├── methods.md              各方法的原理、公式、調參與適用情況
│   └── report-spec.md          成果報告規格
├── scripts/
│   └── run_analysis.py         分析主程式
├── docs/                       教學文件（Quarto 原始檔，中英兩版）
│   ├── knn-lda-nb-svm_zh.qmd   中文版
│   ├── knn-lda-nb-svm_en.qmd   英文版
│   └── report.css              兩份 qmd 共用的 HTML 樣式
└── examples/                   示範資料（不放進上傳 Claude 的 zip）
    ├── README.md               資料說明、來源與對話範例
    ├── breast_cancer.csv       分類示範
    └── auto-mpg.csv            迴歸示範
```

## 安裝

```bash
pip install -r requirements.txt
```

**必要：**Python 與 `requirements.txt` 的套件。只要這些就能產生主要報告 `report.html`。

**選用：**

- [pandoc](https://pandoc.org)：`report.html` 的數學式較佳、可離線顯示（沒有時改用 `markdown` 套件，數學式需要網路）。
- [Quarto](https://quarto.org)：想自行 render `report.qmd` 時才需要。
- PDF 工具（擇一）：`pip install playwright && playwright install chromium`，或含 `xeCJK` 套件的 TeX（TeX Live、MiKTeX）。只有在 config 設定 `"pdf": "on"` 時才會用到；預設 `"off"` 完全不檢查。

沒有 PDF 工具時，用瀏覽器開啟 `report.html` → 列印 → 另存為 PDF 即可。

報告以字型清單指定標楷體與 Times New Roman，電腦沒有時自動使用相近字型。

## 安裝到 Claude

### Claude.ai（網頁或桌面版）

1. 設定 → Capabilities：開啟程式碼執行（code execution）。
2. 下載 repo 中已打包好的 [`knn-lda-nb-svm-skill.zip`](knn-lda-nb-svm-skill.zip)，下載後**不要解壓縮**。
3. 自訂 → Skills → 「+」→ 上傳 `knn-lda-nb-svm-skill.zip`。

> **自行修改後重新打包：** zip 最上層必須是 `knn-lda-nb-svm-skill/` 資料夾，資料夾內直接就是 `SKILL.md`；資料夾名稱必須與 `SKILL.md` 的 `name` 相同。不要多包一層，也不要只壓縮資料夾裡的檔案而沒有資料夾本身。建議只放 `SKILL.md`、`references/`、`scripts/`、`README.md`、`requirements.txt`，不放 `docs/`、`examples/`。

### Claude Code

- 個人使用：把資料夾放到 `~/.claude/skills/`，路徑應為 `~/.claude/skills/knn-lda-nb-svm-skill/SKILL.md`。
- 單一專案使用：放到專案的 `.claude/skills/knn-lda-nb-svm-skill/SKILL.md`。
- 放的是 `knn-lda-nb-svm-skill` 這個資料夾，不是整個 GitHub 專案資料夾，否則多包一層會找不到。
- 本機需要 Python 與 `requirements.txt` 的套件；PDF 工具為選用（見「安裝」）。

## 如何對話

用自然語言描述需求並附上資料（CSV、Excel），不需要說出方法名稱。例如：

- 「附檔是病人的檢驗數值，想預測病人屬於哪一型，幫我分析並產出報告。」
- 「這份資料的 Y 是是否違約（0／1），違約的人很少，幫我建分類模型，要能抓到違約的人。」
- 「我想用這些變數預測房價，關係可能不是直線，幫我比較幾種方法的預測表現。」

AI 會依 `SKILL.md` 的流程逐步詢問；需要選擇時會以選單或編號清單呈現。

## 示範資料

`examples/` 內附兩份公開資料，可直接上傳到 Claude 試用：

- `breast_cancer.csv`：乳房腫瘤良性／惡性分類（569 筆，二元）
- `auto-mpg.csv`：汽車油耗預測（398 筆，迴歸）

資料說明、來源與對話範例見 [examples/README.md](examples/README.md)。

## 使用方式（終端機）

輸出（`output/`）：

| 檔案 | 內容 |
|:--|:--|
| `figures/`、`tables/` | EDA 與診斷圖表、描述統計表 |
| `profile.json` | 統計診斷、建議方法、其他可行方法與不可行方法 |
| `results.json` | 報告用到的全部數值 |
| `report_draft.md` | 數值與圖表已填好、解讀待填的報告草稿 |
| `report.html` | **主要報告**（`finalize` 產生）：單一檔案、圖片內嵌、完整程式碼預設收合 |
| `report.md` | Markdown 版報告，圖片內嵌 |
| `report.qmd` | Quarto 版報告，程式碼預設折疊（自行 render 時需要 Quarto） |
| `report.pdf` | 選用：config 的 `pdf` 設為 `"on"` 且環境有 PDF 工具時才產生（預設 `"off"`，不檢查 PDF 環境） |

也可在 Jupyter 中把 `run_analysis.py` 貼進一個儲存格，修改開頭的 `MODE`、`CONFIG_PATH` 後執行。

## 限制與提醒

- 本 skill 輔助分析流程，不能取代統計專業判斷；重要結論請自行確認。
- 前處理、方法選擇與門檻會影響結論，AI 會說明理由，最終決定在使用者。
- 結果是預測表現與變數間的關聯，不代表因果。
- 資料若含個人資料，上傳前請先去識別化（移除姓名、身分證字號、聯絡方式等，或改為代碼）。
- 在 Claude 雲端環境執行時，對話結束後環境會被回收，報告與程式碼請在對話中下載保存。

## 參考文獻

- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning: with Applications in Python*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. *ACM TIST, 2*(3), 27.
- Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2003). *A practical guide to support vector classification*. Technical report, National Taiwan University.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
