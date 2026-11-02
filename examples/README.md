# 示範資料

兩份公開資料，分別示範分類與迴歸。兩者皆可直接上傳到 Claude，搭配本 skill 分析。

| 檔案 | 任務 | 規模 | Y |
|:--|:--|:--|:--|
| `breast_cancer.csv` | 分類（二元） | 569 筆 × 30 個特徵 | `diagnosis` |
| `auto-mpg.csv` | 迴歸 | 398 筆 × 8 個變數（另含車名） | `mpg` |

---

## breast_cancer.csv：乳房腫瘤良性／惡性分類

**內容。** 以細針抽吸（fine needle aspirate）取得乳房腫塊的數位影像，從影像中的細胞核計算 10 種特徵；每種特徵各取平均值、標準誤與最大值（最差的 3 個細胞核的平均），共 30 個特徵。每一列是一位病人的一個腫塊。

| 欄位 | 說明 |
|:--|:--|
| `mean radius`、`radius error`、`worst radius` | 半徑（中心到周長各點距離的平均）的平均值、標準誤、最大值 |
| `… texture` | 紋理（灰階值的標準差） |
| `… perimeter` | 周長 |
| `… area` | 面積 |
| `… smoothness` | 平滑度（半徑長度的局部變化） |
| `… compactness` | 緊密度（周長² ÷ 面積 − 1） |
| `… concavity` | 凹陷程度 |
| `… concave points` | 凹陷點數 |
| `… symmetry` | 對稱性 |
| `… fractal dimension` | 碎形維度（海岸線近似 − 1） |
| `diagnosis` | **Y**：malignant（惡性）= 212、benign（良性）= 357 |

每種特徵都有三個欄位：`mean …`（平均值）、`… error`（標準誤）、`worst …`（最大值）。

**特點。** 全部為連續型特徵、無缺失值；二元分類，惡性約占 37%；特徵數較多（p = 30），且部分特徵幾乎完全相關（例如半徑、周長與面積，相關係數最高約 0.998）。適合示範：正類的指定、高度相關時不建議單純貝氏、漏判惡性的代價與門檻調整。

**取得方式。** 由 scikit-learn 的 `sklearn.datasets.load_breast_cancer()` 匯出；scikit-learn 原本以 0 表示惡性、1 表示良性，此處改為文字標籤 `malignant`、`benign`。

**來源與授權。**
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
授權：CC BY 4.0。

**對話範例。**

> 請使用 knn-lda-nb-svm-skill 分析附檔 breast_cancer.csv。這是乳房腫塊細胞核影像特徵的資料，想用這些特徵預測腫瘤是良性或惡性，Y 是 diagnosis，正類是 malignant，其他欄位都當 X。漏判惡性的代價比誤判高。報告輸出 HTML。

---

## auto-mpg.csv：汽車油耗預測

**內容。** 1970–1982 年車款的油耗與車輛規格。每一列是一款車。

| 欄位 | 說明 |
|:--|:--|
| `mpg` | **Y**：每加侖可行駛英里數（城市行駛油耗，越高越省油） |
| `cylinders` | 汽缸數（3、4、5、6、8） |
| `displacement` | 排氣量（立方英寸） |
| `horsepower` | 馬力；**6 筆缺失，以 `?` 表示** |
| `weight` | 車重（磅） |
| `acceleration` | 0–60 英里／小時加速時間（秒） |
| `model year` | 車款年份（70–82，即 1970–1982） |
| `origin` | 產地代碼：1 = 美國、2 = 歐洲、3 = 日本 |
| `car name` | 車名（每列幾乎都不同，屬於識別欄位，不放進 X） |

**特點。** Y 為連續；有缺失值；`origin` 是以數字編碼的類別變數，`cylinders` 只有 5 種值；`car name` 為識別欄位。適合示範缺失值處理、類別編碼確認，以及 KNN 迴歸、SVR 與線性迴歸基準的比較。

**取得方式。** UCI Machine Learning Repository 的 Auto MPG 資料，原樣保留（含 `?`）。原始資料共 406 筆，UCI 已移除 8 筆 `mpg` 未知的資料。

**來源與授權。**
Quinlan, R. (1993). *Auto MPG* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5859H
授權：CC BY 4.0。資料最早來自卡內基美隆大學的 StatLib 資料庫。

**對話範例。**

> 請使用 knn-lda-nb-svm-skill 分析附檔 auto-mpg.csv。想用車輛規格預測油耗，Y 是 mpg；car name 是車名，不要放進 X；origin 是產地代碼（1 美國、2 歐洲、3 日本），請當類別變數。horsepower 的 ? 是缺失值。報告輸出 HTML。
