# 示範資料

兩份公開資料，分別示範分類與迴歸。兩者皆可直接上傳到 Claude，搭配本 skill 分析。

| 檔案 | 任務 | 規模 | Y |
|:--|:--|:--|:--|
| `wine.csv` | 分類（三類） | 178 筆 × 13 個特徵 | `cultivar` |
| `auto-mpg.csv` | 迴歸 | 398 筆 × 8 個變數（另含車名） | `mpg` |

---

## wine.csv：葡萄酒品種分類

**內容。** 義大利同一產區、三個不同葡萄品種（cultivar）釀成的葡萄酒，以化學分析測得 13 項成分。每一列是一個葡萄酒樣本。

| 欄位 | 說明 |
|:--|:--|
| `alcohol` | 酒精濃度 |
| `malic_acid` | 蘋果酸 |
| `ash` | 灰分 |
| `alcalinity_of_ash` | 灰分鹼度 |
| `magnesium` | 鎂 |
| `total_phenols` | 總酚 |
| `flavanoids` | 類黃酮 |
| `nonflavanoid_phenols` | 非類黃酮酚 |
| `proanthocyanins` | 原花青素 |
| `color_intensity` | 色澤強度 |
| `hue` | 色調 |
| `od280/od315_of_diluted_wines` | 稀釋酒液在 280／315 nm 的吸光度比 |
| `proline` | 脯胺酸 |
| `cultivar` | **Y**：葡萄品種（class_0 = 59、class_1 = 71、class_2 = 48） |

**特點。** 全部為連續型特徵、無缺失值、三類大致平衡；各特徵尺度差異大（例如 `proline` 約數百至一千多，`hue` 約為 1），適合示範標準化、LDA 與其他分類方法的比較。

**取得方式。** 由 scikit-learn 的 `sklearn.datasets.load_wine()` 匯出，類別名稱沿用 scikit-learn 的 `class_0`、`class_1`、`class_2`。

**來源與授權。**
Aeberhard, S., & Forina, M. (1992). *Wine* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J
授權：CC BY 4.0。

**對話範例。**

> 請使用 knn-lda-nb-svm-skill 分析附檔 wine.csv。這是三個葡萄品種釀成的葡萄酒化學成分資料，想用化學成分預測葡萄品種，Y 是 cultivar，其他欄位都當 X。報告輸出 HTML。

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
