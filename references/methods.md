# 方法說明：KNN、LDA／QDA、單純貝氏、支援向量機

撰寫報告的「資料與方法」、方法公式（附錄）與解讀時查閱。只讀取本次實際使用的方法。選擇方法的規則見 `decision-tree.md`。

## 共同概念

**貝氏分類器。** 把 $x_0$ 分到條件機率 $\Pr(Y = j \mid X = x_0)$ 最大的類別，可使錯誤率最小；兩類時決策邊界為 $\Pr(Y = 1 \mid X = x) = 0.5$。真實分佈未知，各方法以不同方式近似它：KNN 用鄰域內的比例，LDA、QDA、單純貝氏以貝氏定理

$$
\Pr(Y = k \mid X = x) = \frac{\pi_k f_k(x)}{\sum_{l=1}^{K} \pi_l f_l(x)}
$$

估計（$\pi_k$ 為先驗機率，$f_k$ 為類別條件密度），SVM 直接找分隔邊界。

**錯誤率與 MSE。** 分類錯誤率 $\frac{1}{n}\sum I(y_i \ne \hat{y}_i)$；迴歸用均方誤差 $\frac{1}{n}\sum (y_i - \hat{f}(x_i))^2$。只有未參與配適的資料上計算的測試誤差代表預測能力。

**彈性與偏差–變異。** 彈性指模型能配適的邊界或函數形狀有多廣。彈性越高，偏差越低、變異越高；訓練誤差一定下降，測試誤差先降後升。資料少、雜訊大時選彈性低的方法。

| 方法 | log-odds 形式 | 彈性 |
|:--|:--|:--|
| LDA、logistic 迴歸、線性 SVM | 線性 | 低 |
| 單純貝氏 | 加法，無交互作用 | 中 |
| QDA | 二次，含 $x_j x_l$ 項 | 中到高 |
| RBF 核 SVM | 由 $C$、$\gamma$ 決定 | 高 |
| KNN | 無參數形式，由 $k$ 決定 | 最高 |

| 方法 | 分類 | 迴歸 | 輸出機率 |
|:--|:--:|:--:|:--|
| KNN | ✓ | ✓ | ✓（鄰居中各類比例） |
| LDA、QDA | ✓ | — | ✓ |
| 單純貝氏 | ✓ | — | ✓（常過於極端） |
| SVM | ✓（SVC） | ✓（SVR） | 只輸出類別；機率需另外校準 |

## KNN

**核心想法。** 以離 $x_0$ 最近的 $k$ 個訓練點 $\mathcal{N}_k(x_0)$ 投票或取平均：

$$
\widehat{\Pr}(Y = j \mid X = x_0) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x_0)} I(y_i = j),
\qquad
\hat{f}(x_0) = \frac{1}{k}\sum_{x_i \in \mathcal{N}_k(x_0)} y_i .
$$

**偏差–變異。** KNN 迴歸在 $x_0$ 的期望預測誤差

$$
\operatorname{Err}(x_0) = \sigma^2_\varepsilon + \left[f(x_0) - \frac{1}{k}\sum_{\ell=1}^{k} f(x_{(\ell)})\right]^2 + \frac{\sigma^2_\varepsilon}{k}.
$$

$k$ 小：低偏差、高變異（$k = 1$ 時訓練誤差為 0）；$k$ 大：高偏差、低變異。有效參數個數約為 $N/k$。

**前處理與調參。** 必須標準化（以歐氏距離運作）。以交叉驗證選 $k$（1 到約 $\sqrt{N}$ 的奇數，兩類時用奇數避免平手）；不可用訓練誤差選 $k$。

**適用。** 預測為目的、邊界高度不規則、$p$ 小且 $N \gg p$、連續特徵。**不適用：** 需要解釋效果、$p$ 大或含許多無關特徵（維度詛咒：鄰域需延伸到各變數大部分範圍，「最近」的鄰居已不近）、$N$ 極大且需要即時預測（須儲存並搜尋全部訓練資料）。

**診斷。** CV 誤差對 $k$ 的曲線。KNN 沒有係數可解讀。

## LDA 與 QDA

**模型。** LDA 假設 $X \mid Y = k \sim N(\mu_k, \Sigma)$（共同共變異），區別函數為線性：

$$
\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \tfrac{1}{2}\, \mu_k^\top \Sigma^{-1} \mu_k + \log \pi_k ,
$$

$$
\hat{\pi}_k = \frac{N_k}{N}, \quad
\hat{\mu}_k = \frac{1}{N_k}\sum_{y_i = k} x_i, \quad
\hat{\Sigma} = \frac{1}{N - K}\sum_{k}\sum_{y_i = k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^\top .
$$

QDA 假設 $X \mid Y = k \sim N(\mu_k, \Sigma_k)$，區別函數為二次：

$$
\delta_k(x) = -\tfrac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) - \tfrac{1}{2}\log\lvert\Sigma_k\rvert + \log \pi_k .
$$

| 參數個數 | LDA | QDA |
|:--|:--|:--|
| 共變異參數 | $p(p+1)/2$ | $K\,p(p+1)/2$ |
| 決策規則參數 | $(K-1)(p+1)$ | $(K-1)\{p(p+3)/2 + 1\}$ |

**偏差–變異。** LDA 變異低，共同共變異假設錯誤時偏差高；QDA 需要相對 $Kp^2/2$ 很大的 $N$，任一類 $N_k \le p$ 時 $\hat{\Sigma}_k$ 奇異、無法估計。$p$ 接近 $N$（本 skill 的門檻為 $N/p < 10$，含 $p \ge N$）時用收縮 LDA（`solver="lsqr", shrinkage="auto"`）。

**Fisher 區別。** 不需高斯假設，找使組間變異相對組內變異最大的方向：$J(w) = \dfrac{w^\top S_B w}{w^\top S_W w}$，兩類時 $w \propto S_W^{-1}(m_2 - m_1)$。$K$ 類最多有 $K-1$ 個區別方向，可用來把資料投影到低維空間觀察類別分離。

**門檻。** 輸出後驗機率，稀有類別時可降低門檻（例如 0.5 → 0.2），以整體錯誤率換取較高的敏感度；門檻依誤判成本決定。

**適用。** LDA：各類散布相近、$N$ 小到中等、類別多、類別分得很開、需要機率。QDA：各類共變異明顯不同且每類樣本充足。兩者都不必先通過常態或共變異同質性檢定，以交叉驗證比較即可；對離群值不穩健。

**LDA 與 logistic 迴歸。** 兩者 log-odds 皆為線性；LDA 最大化聯合概似、假設 $X$ 為高斯，假設成立時較有效率，類別完全分開時仍有定義；logistic 只用條件概似、較穩健。結果通常相近。

**診斷。** 區別座標散佈圖與各方向承載的組間變異比例；QDA 檢查各類共變異結構與樣本數。

## 單純貝氏

**模型。** 假設在第 $k$ 類之內各特徵獨立：

$$
f_k(x) = \prod_{j=1}^{p} f_{kj}(x_j),
\qquad
\log\frac{\Pr(Y = k \mid x)}{\Pr(Y = K \mid x)} = \log\frac{\pi_k}{\pi_K} + \sum_{j=1}^{p} \log\frac{f_{kj}(x_j)}{f_{Kj}(x_j)} ,
$$

為加法模型、無交互作用。每個一維密度依變數型態選擇：連續 → 高斯 $N(\mu_{jk}, \sigma^2_{jk})$（`GaussianNB`）；類別 → 各水準比例（`CategoricalNB`）；計數 → 多項分佈（`MultinomialNB`）。某水準在某類從未出現時機率為 0，以拉普拉斯平滑（`alpha = 1`）處理。

**多項單純貝氏（文字分類）。** 詞袋表示 $x = (x_1, \ldots, x_V)$（各字詞次數），$\Pr(x \mid Y = k) \propto \prod_w \theta_{kw}^{x_w}$，
$\hat{\theta}_{kw} = (N_{kw} + \alpha)/(N_k + \alpha V)$，分類取 $\log \hat{\pi}_k + \sum_w x_w \log \hat{\theta}_{kw}$ 最大者。實作：`CountVectorizer` + `MultinomialNB`；只記錄是否出現時用 `BernoulliNB`。

**適用。** $p$ 很大（文字、基因）、特徵型態混合、$N$ 小（甚至 $p > N$）；類別內特徵大致獨立時與 LDA 一樣好。**不適用：** 特徵高度相關（假設錯誤造成的偏差不隨 $n$ 消失）、交互作用重要、$N$ 大而 $p$ 小時好處有限。後驗機率常過於極端，直接使用機率值前須檢查校準。

**診斷。** 使用機率時：預測機率分組與實際比例的校準表或圖。

## 支援向量機（SVM）

**軟邊界分類器。**

$$
\min_{\beta_0, \beta, \xi}\; \tfrac{1}{2}\lVert\beta\rVert^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad y_i(\beta_0 + \beta^\top x_i) \ge 1 - \xi_i,\; \xi_i \ge 0 .
$$

$\xi_i = 0$ 在邊界正確側，$0 < \xi_i \le 1$ 在邊界內，$\xi_i > 1$ 誤分類；邊界寬 $2/\lVert\beta\rVert$。落在邊界上、邊界內或誤分類的點為支援向量，只有它們決定解。$C$ 為違反邊界的成本（scikit-learn、LIBSVM 的定義）：$C$ 小 → 邊界寬、支援向量多、變異低；$C$ 大 → 邊界窄、變異高。

**核函數。** 決策函數 $f(x) = \operatorname{sign}\bigl(\sum_{i \in \mathcal{S}} y_i \alpha_i K(x_i, x) + b\bigr)$。

| 核函數 | $K(x, x')$ | 適用 |
|:--|:--|:--|
| 線性 | $\langle x, x' \rangle$ | $p \gg n$、文字、基因 |
| 多項式 | $(1 + \langle x, x' \rangle)^d$ | 已知邊界為低次多項式 |
| RBF | $\exp(-\gamma \lVert x - x' \rVert^2)$ | 一般非線性邊界的預設 |

**多類別。** 一對一：$K(K-1)/2$ 個分類器、多數決（`SVC`）；一對多：$K$ 個「此類對其他類」分類器、取 $f_k(x)$ 最大（`LinearSVC` 預設），訓練資料兩類數量懸殊。

**支援向量迴歸（ε-SVR）。** 目的：預測連續反應變數，以核函數處理非線性，對離群值較最小平方不敏感。ε-不敏感損失 $V_\varepsilon(r) = \max(0, \lvert r \rvert - \varepsilon)$：殘差小於 $\varepsilon$ 不計損失（擬合曲線上下各 $\varepsilon$ 的 ε 不敏感帶），超出部分線性計價。帶內點的權重為 0；帶邊界上的點 $0 < \alpha_i < C$；帶外的點 $\alpha_i = C$，後兩者皆為支援向量。$\varepsilon$ 越大，支援向量越少。

**前處理與調參。** 必須標準化。$(C, \gamma)$ 以 log₂ 網格搜尋（$C = 2^{-5}, \ldots, 2^{15}$；$\gamma = 2^{-15}, \ldots, 2^{3}$），先粗後細，兩者一起調；最佳點在網格邊緣時擴大網格。SVR 另調 $\varepsilon$。不平衡時用 `class_weight="balanced"`。排序或時間資料的交叉驗證要打亂或依時間切分。

**適用。** 追求分類準確率、非線性邊界（RBF）、$p \gg n$（線性核）、中小樣本、連續且已標準化的特徵、類別分得開。**不適用：** 需要機率或可解釋效果（`SVC(probability=True)` 的 Platt 機率只是近似）、$n$ 極大（改用 `LinearSVC` 或子抽樣）、非線性核搭配許多無關特徵（先挑選特徵）。線性 SVM 與 logistic 迴歸結果常相近；類別分得很開時 SVM 往往較好，重疊多時 logistic 較好。

**診斷。** $(C, \gamma)$ 的 CV 曲面與最佳點、支援向量個數；SVR 加上預測值對實際值圖與殘差圖。

## 評估與推論

| 指標 | 定義 |
|:--|:--|
| 準確率 | $(TP + TN)/n$ |
| 敏感度（召回率） | $TP/(TP + FN)$ |
| 特異度 | $TN/(TN + FP)$ |
| 精確率 | $TP/(TP + FP)$ |
| F1 | 精確率與敏感度的調和平均 |
| 平衡準確率 | 各類召回率的平均 |
| AUC | ROC 曲線下面積；隨機一個正類得分高於隨機一個負類的機率 |
| RMSE、MAE、$R^2$ | 迴歸：均方根誤差、平均絕對誤差、決定係數 |

- 混淆矩陣：列為實際類別、欄為預測類別。
- 基準：分類為永遠預測最多數類別；迴歸為以訓練資料平均數預測。
- 準確率的 95% Wilson 信賴區間：$\dfrac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$，$z = 1.96$。
- 兩模型在同一測試集上的比較：McNemar 檢定，$H_0$：兩模型錯誤率相同。令 $b$、$c$ 為只有其中一個模型分錯的筆數，在 $H_0$ 下 $b \sim \text{Binomial}(b + c, 0.5)$，以精確二項檢定計算雙尾 p 值。
- 交叉驗證：分類用分層 $K$ 折、先打亂；不平衡資料以平衡準確率選模；用於選模的 CV 分數不能當成獨立測試表現。
- 排列重要性（permutation importance）：在測試集上隨機打亂某個變數後指標下降的幅度，代表對預測的貢獻，不代表因果效果。
