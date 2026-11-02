#!/usr/bin/env python3
"""KNN、LDA／QDA、單純貝氏與支援向量機分析程式（同一份程式負責全部數值、圖表與報告草稿）。

用法（終端機）：
  python run_analysis.py inspect  --data 資料.csv --y 欄位 [--x 欄位1 欄位2 ...]
  python run_analysis.py profile  --config config.json
  python run_analysis.py run      --config config.json
  python run_analysis.py finalize --config config.json --report output/report.md

  inspect   資料摘要：筆數、欄位型態、各欄缺失、Y 的分佈與前處理建議；同時寫出 config 範本。
  profile   套用已同意的前處理後，以訓練資料計算統計診斷，列出建議方法（附理由）、其他可行方法與不可行方法（附原因）。
  run       完整分析：EDA 圖表、交叉驗證調參、測試集評估、診斷圖、results.json 與 report_draft.md。
  finalize  檢查解讀是否填寫完畢、附上完整程式碼，輸出 report.html（主要報告）、report.md（內嵌圖片）
            與 report.qmd；config 的 pdf 為 "on" 時另嘗試產生 report.pdf（失敗不算錯誤）。

用法（Jupyter）：在下方「設定」修改 MODE 與路徑後，把整個檔案貼進一個儲存格執行。
"""
import argparse
import base64
import hashlib
import html as html_lib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────────────────────── 設定（Jupyter 使用時在此修改）─────────────────────────────
# 相對路徑以「執行程式時所在的資料夾」為基準；可用 Path.cwd() 確認。
# Windows 路徑請寫成 r"C:\Users\name\data.csv"。
MODE = "run"                     # inspect / profile / run / finalize
DATA_PATH = "data.csv"           # 只有 inspect 使用；其他模式從 config 讀取
Y_COLUMN = None                  # 只有 inspect 使用
CONFIG_PATH = "config.json"
REPORT_PATH = "output/report.md" # 只有 finalize 使用
# ────────────────────────────────────────────────────────────────────────────────────

IN_JUPYTER = "ipykernel" in sys.modules
if not IN_JUPYTER:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scipy.stats import binomtest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=FitFailedWarning)          # 失敗的參數組合另記錄在 results.json
warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite")
warnings.filterwarnings("ignore", message="Variables are collinear")

CLASSIFIERS = ["knn", "lda", "lda_shrink", "qda", "nb_gaussian", "nb_mixed", "nb_multinomial",
               "svm_linear", "svm_rbf", "logistic"]
REGRESSORS = ["knn_reg", "svr", "linear_reg"]
METHOD_NAMES = {
    "knn": "KNN", "lda": "LDA", "lda_shrink": "收縮 LDA", "qda": "QDA",
    "nb_gaussian": "高斯單純貝氏", "nb_mixed": "單純貝氏（連續＋類別）", "nb_multinomial": "多項單純貝氏",
    "svm_linear": "線性核 SVM", "svm_rbf": "RBF 核 SVM", "logistic": "Logistic 迴歸（比較用）",
    "knn_reg": "KNN 迴歸", "svr": "SVR", "linear_reg": "線性迴歸（比較用）",
    "baseline": "基準模型",
}
MAX_N_KERNEL = 20000             # 訓練資料超過此筆數時核 SVM／SVR 不可行（config 的 max_n_kernel 可覆寫）
NA_VALUES = ["", "NA", "N/A", "NaN", "?"]   # 讀取時視為缺失的字串（config 的 na_values 可覆寫）
DROP_FIRST = ("lda", "lda_shrink", "qda", "nb_gaussian")   # 這些方法的 one-hot 刪除基準水準，避免完全共線


# ═════════════════════════════════ 共用工具 ═════════════════════════════════
def fail(msg):
    """遇到錯誤立即停止，不產出部分結果。"""
    raise SystemExit(f"[錯誤] {msg}")


def to_builtin(o):
    if isinstance(o, dict):
        return {str(k): to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_builtin(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return to_builtin(o.tolist())
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫出：{Path(path).resolve()}")


def show_table(df, title):
    """數值結果一律整理成表格印出。"""
    print(f"\n── {title} ──")
    with pd.option_context("display.max_rows", 60, "display.max_columns", 20, "display.width", 160):
        print(df)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(path, sheet=None, encoding="utf-8", na_values=None):
    """讀取 CSV 或 Excel。CSV 預設 UTF-8；中文 Windows 匯出的檔案常為 cp950 或 utf-8-sig。
    na_values 中的字串（預設含 "?"）視為缺失。"""
    p = Path(path)
    if not p.exists():
        fail(f"找不到資料檔，程式實際尋找的路徑為：{p.resolve()}")
    na = NA_VALUES if na_values is None else na_values
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=sheet or 0, na_values=na)
    try:
        return pd.read_csv(p, encoding=encoding, na_values=na)
    except UnicodeDecodeError:
        fail(f"以編碼 {encoding} 讀取失敗，請在 config 的 encoding 改用 cp950 或 utf-8-sig。")


# ─────────────── 圖表字型：英數優先 Times New Roman，中文依清單選用系統有的字型 ───────────────
CHART_FONTS = ["Times New Roman", "DFKai-SB", "BiauKai", "Microsoft JhengHei", "PingFang TC",
               "Noto Sans CJK TC", "Noto Sans TC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
CHART_CJK = CHART_FONTS[1:]
NO_CJK_WARNING = "系統沒有任何中文字型，圖中的中文可能顯示為方框"


def setup_fonts():
    """font.family 只保留系統實際有的字型；一個中文字型都沒有時只警告，不中止。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in CHART_FONTS if f in available]
    plt.rcParams["font.family"] = family or ["DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100
    cjk = next((f for f in family if f in CHART_CJK), None)
    if not cjk:
        print(f"[警告] {NO_CJK_WARNING}。")
    return {"family": family or ["DejaVu Serif"], "cjk": cjk}


PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]


class FigureSaver:
    def __init__(self, out_dir):
        self.dir = Path(out_dir) / "figures"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = {}

    def save(self, fig, name, caption, source):
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.index[name] = {"file": f"figures/{name}.png", "caption": caption, "source": source}
        print(f"已存圖：{path.resolve()}")
        if IN_JUPYTER:
            from IPython.display import Image, display
            display(Image(filename=str(path)))
        return self.index[name]


# ═════════════════════════════════ 資料準備 ═════════════════════════════════
def column_types(df, cols, forced_cat=()):
    num, cat = [], []
    for c in cols:
        if c in forced_cat or not pd.api.types.is_numeric_dtype(df[c]):
            cat.append(c)
        else:
            num.append(c)
    return num, cat


def guess_task(y):
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10:
        return "classification"
    return "regression"


def inspect(data, y_col, x_cols=None, sheet=None, encoding="utf-8", out_dir="output"):
    df = load_data(data, sheet, encoding)
    if y_col not in df.columns:
        fail(f"找不到 Y 欄位 {y_col!r}；可用欄位：{list(df.columns)}")
    x_cols = x_cols or [c for c in df.columns if c != y_col]
    miss = [c for c in x_cols if c not in df.columns]
    if miss:
        fail(f"找不到 X 欄位：{miss}")
    num, cat = column_types(df, x_cols)
    n = len(df)
    rows = []
    for c in [y_col] + x_cols:
        m = int(df[c].isna().sum())
        share = m / n
        role = "Y" if c == y_col else "X"
        typ = "數值" if c in num or (c == y_col and pd.api.types.is_numeric_dtype(df[c])) else "類別"
        if m == 0:
            prop = "無缺失，不需處理"
        elif role == "Y":
            prop = "Y 缺失：須另行確認是否排除這些資料列，不插補"
        elif typ == "數值" and share <= 0.20:
            prop = "建議以中位數插補（只用訓練資料估計中位數）"
        elif typ == "數值":
            prop = "缺失比例 > 20%：列出刪除後影響，須經使用者同意才刪除變數"
        else:
            prop = "類別變數：建議以眾數或「缺失」類別處理，須經使用者確認"
        note = ""
        if typ == "數值" and df[c].nunique() <= 5:
            note = "數值但不同值 ≤ 5，可能是類別編碼"
        if df[c].nunique() == n and role == "X":
            note = "每列都不同，可能是 ID 欄位，不應納入模型"
        if df[c].nunique() <= 1:
            note = "常數欄位"
        rows.append({"變數": c, "角色": role, "型態": typ, "不同值個數": int(df[c].nunique()),
                     "缺失筆數": m, "缺失比例": round(share, 4), "建議處理": prop, "備註": note})
    table = pd.DataFrame(rows)
    task = guess_task(df[y_col])
    yinfo = (df[y_col].value_counts(dropna=False).rename("筆數").to_frame()
             .assign(比例=lambda d: (d["筆數"] / n).round(4))) if task == "classification" else df[y_col].describe().to_frame()
    show_table(table, "各變數型態與缺失")
    show_table(yinfo, f"Y（{y_col}）的分佈；推測任務：{'分類' if task == 'classification' else '迴歸'}")
    dup = int(df.duplicated().sum())
    print(f"\n原始資料 {n} 列、{df.shape[1]} 欄；選定 X {len(x_cols)} 個（數值 {len(num)}、類別 {len(cat)}）；完全重複列 {dup} 列。")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "inspect_missing.csv", index=False, encoding="utf-8-sig")
    template = {
        "data_path": str(data), "sheet": sheet, "encoding": encoding, "na_values": NA_VALUES,
        "y": y_col, "x": x_cols, "task": task,
        "positive_class": None, "categorical": cat, "text_column": None,
        "drop_columns": [], "impute_numeric": "median", "impute_categorical": "most_frequent",
        "drop_missing_y": False, "test_size": 0.25, "cv_folds": 5, "seed": 2026, "time_order": False,
        "metric": None, "candidates": [], "threshold": 0.5, "class_weight": None, "grids": {},
        "max_n_kernel": MAX_N_KERNEL, "pdf": "off", "out_dir": out_dir,
        "report": {"title": "分析報告", "background": "", "unit": "", "goal": ""},
    }
    write_json({"n_rows": n, "n_cols": df.shape[1], "duplicates": dup, "task_guess": task,
                "variables": rows}, out / "inspect.json")
    write_json(template, out / "config_template.json")
    return table


class Prepared:
    """依 config 讀取資料、套用已同意的變數刪除與 Y 缺失處理，並切分訓練／測試資料。"""

    def __init__(self, cfg):
        self.cfg = cfg
        df = load_data(cfg["data_path"], cfg.get("sheet"), cfg.get("encoding", "utf-8"), cfg.get("na_values"))
        self.raw_n, self.raw_cols = len(df), df.shape[1]
        y = cfg["y"]
        if y not in df.columns:
            fail(f"找不到 Y 欄位 {y!r}")
        x = cfg.get("x") or [c for c in df.columns if c != y]
        miss = [c for c in x if c not in df.columns]
        if miss:
            fail(f"找不到 X 欄位：{miss}")
        dropped = [d["name"] for d in cfg.get("drop_columns", [])]
        self.x_all = x
        self.text = cfg.get("text_column")
        x = [c for c in x if c not in dropped]
        if self.text:
            x = [self.text]
        n0 = len(df)
        # 缺失概況（處理前，以全部資料計算）
        self.missing = []
        for c in [y] + self.x_all:
            m = int(df[c].isna().sum())
            self.missing.append({"變數": c, "缺失筆數": m, "缺失比例": m / n0})
        y_missing = int(df[y].isna().sum())
        if y_missing:
            if not cfg.get("drop_missing_y"):
                fail(f"Y 有 {y_missing} 筆缺失，請先與使用者確認處理方式（config 的 drop_missing_y）。")
            df = df[df[y].notna()].copy()
        self.y_dropped_rows = y_missing
        self.df = df.reset_index(drop=True)
        self.y_name, self.x = y, x
        self.task = cfg.get("task") or guess_task(df[y])
        self.num, self.cat = ([], []) if self.text else column_types(df, x, cfg.get("categorical", []))
        if self.text:
            self.df[self.text] = self.df[self.text].fillna("").astype(str)
        for c in self.cat:
            self.df[c] = self.df[c].astype("object").where(self.df[c].notna(), np.nan)
        X, Y = (self.df[self.text] if self.text else self.df[x]), self.df[y]   # 文字欄位以一維 Series 傳入向量化器
        self.classes, self.pos = None, None
        if self.task == "classification":
            Y = Y.astype(str)
            self.classes = sorted(Y.unique())
            if len(self.classes) < 2:
                fail("Y 只有一個類別，無法進行分類。")
            if len(self.classes) == 2:
                pos = cfg.get("positive_class")
                pos = str(pos) if pos is not None else Y.value_counts().idxmin()
                if pos not in self.classes:
                    fail(f"positive_class {pos!r} 不在 Y 的類別 {self.classes} 中")
                self.pos = pos
        seed, ts = cfg.get("seed", 2026), cfg.get("test_size", 0.25)
        if cfg.get("time_order"):
            cut = int(round(len(X) * (1 - ts)))
            self.Xtr, self.Xte, self.ytr, self.yte = X.iloc[:cut], X.iloc[cut:], Y.iloc[:cut], Y.iloc[cut:]
        else:
            strat = Y if self.task == "classification" and Y.value_counts().min() >= 2 else None
            self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
                X, Y, test_size=ts, random_state=seed, stratify=strat)
        # 實際處理筆數與插補值（插補值只以訓練資料估計）
        for row in self.missing:
            c = row["變數"]
            if c == y:
                row["處理方式"] = "排除缺失 Y 的資料列" if y_missing else "無缺失"
                row["處理筆數"] = y_missing
            elif c in dropped:
                d = next(d for d in cfg["drop_columns"] if d["name"] == c)
                row["處理方式"] = f"刪除變數（{d.get('reason', '')}；{d.get('confirmed', '使用者同意')}）"
                row["處理筆數"] = int(self.raw_n)
            elif c not in x:
                row["處理方式"] = "未納入模型"
                row["處理筆數"] = 0
            else:
                m = int(self.df[c].isna().sum())
                row["處理筆數"] = m
                if m == 0:
                    row["處理方式"] = "無缺失"
                elif c in self.num:
                    med = float(self.Xtr[c].median())
                    row["處理方式"] = f"中位數插補（訓練資料中位數 = {med:.4g}；交叉驗證各折重新估計）"
                    row["插補值"] = med
                else:
                    how = cfg.get("impute_categorical", "most_frequent")
                    row["處理方式"] = "眾數插補" if how == "most_frequent" else "以「（缺失）」為新類別"
        self.n_final = len(self.df)
        self.p_final = len(self.x)

    def cv(self, folds=None):
        k = folds or self.cfg.get("cv_folds", 5)
        seed = self.cfg.get("seed", 2026)
        if self.cfg.get("time_order"):
            return TimeSeriesSplit(n_splits=k)
        if self.task == "classification":
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return KFold(n_splits=k, shuffle=True, random_state=seed)

    def imputed_train(self):
        """EDA 用：訓練資料套用插補（不標準化），保留原始尺度。"""
        if self.text:
            return self.Xtr.to_frame()
        X = self.Xtr.copy()
        for c in self.num:
            X[c] = X[c].fillna(self.Xtr[c].median())
        for c in self.cat:
            if self.cfg.get("impute_categorical", "most_frequent") == "most_frequent":
                mode = self.Xtr[c].mode(dropna=True)
                X[c] = X[c].fillna(mode.iloc[0] if len(mode) else "（缺失）")
            else:
                X[c] = X[c].fillna("（缺失）")
        return X


# ═════════════════════════════════ 模型與參數網格 ═════════════════════════════════
class MixedNB(ClassifierMixin, BaseEstimator):
    """連續變數用高斯、類別變數用各水準比例（拉普拉斯平滑）的單純貝氏。"""

    def __init__(self, num_cols=(), cat_cols=(), alpha=1.0):
        self.num_cols, self.cat_cols, self.alpha = num_cols, cat_cols, alpha

    def _codes(self, X):
        out = np.zeros((len(X), len(self.cat_cols)), dtype=int)
        for j, c in enumerate(self.cat_cols):
            lv = self.levels_[c]
            out[:, j] = [lv.get(v, len(lv)) for v in X[c].astype(str)]
        return out

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.classes_ = np.unique(y)
        self.medians_ = {c: X[c].median() for c in self.num_cols}
        self.levels_ = {}
        for c in self.cat_cols:
            vals = X[c].astype(str).where(X[c].notna(), "（缺失）")
            self.levels_[c] = {v: i for i, v in enumerate(sorted(vals.unique()))}
        if self.num_cols:
            self.gnb_ = GaussianNB().fit(X[list(self.num_cols)].fillna(self.medians_), y)
        if self.cat_cols:
            Xc = X[list(self.cat_cols)].astype(str).where(X[list(self.cat_cols)].notna(), "（缺失）")
            self.cnb_ = CategoricalNB(alpha=self.alpha,
                                      min_categories=[len(self.levels_[c]) + 1 for c in self.cat_cols]
                                      ).fit(self._codes(Xc), y)
        self.log_prior_ = np.log(np.array([np.mean(np.asarray(y) == k) for k in self.classes_]))
        return self

    def _jll(self, X):
        X = pd.DataFrame(X)
        jll = np.tile(self.log_prior_, (len(X), 1))
        if self.num_cols:
            jll += self.gnb_.predict_joint_log_proba(X[list(self.num_cols)].fillna(self.medians_)) - self.log_prior_
        if self.cat_cols:
            Xc = X[list(self.cat_cols)].astype(str).where(X[list(self.cat_cols)].notna(), "（缺失）")
            jll += self.cnb_.predict_joint_log_proba(self._codes(Xc)) - self.cnb_.class_log_prior_
        return jll

    def predict_proba(self, X):
        jll = self._jll(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._jll(X), axis=1)]


def preprocessor(prep, scale=True, drop_first=False):
    """drop_first=True 時 one-hot 刪除基準水準（LDA、QDA、收縮 LDA、高斯單純貝氏）；KNN、SVM 維持完整編碼。"""
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    how = prep.cfg.get("impute_categorical", "most_frequent")
    cat_imp = (SimpleImputer(strategy="most_frequent") if how == "most_frequent"
               else SimpleImputer(strategy="constant", fill_value="（缺失）"))
    onehot = OneHotEncoder(drop="first" if drop_first else None, handle_unknown="ignore", sparse_output=False)
    trans = []
    if prep.num:
        trans.append(("num", Pipeline(num_steps), prep.num))
    if prep.cat:
        trans.append(("cat", Pipeline([("impute", cat_imp), ("onehot", onehot)]), prep.cat))
    return ColumnTransformer(trans)


def encoded_p(prep, drop_first):
    """進入模型的欄位數：數值欄位數 ＋ Σ（各類別變數的水準數 − 1；不刪基準水準時為水準數）。文字資料為詞彙數。"""
    if prep.text:
        return len(CountVectorizer().fit(prep.Xtr).vocabulary_)
    X = prep.imputed_train()
    return len(prep.num) + sum(int(X[c].astype(str).nunique()) - (1 if drop_first else 0) for c in prep.cat)


def odd_ks(n_train):
    kmax = int(min(51, max(3, np.sqrt(n_train) * 1.5)))
    return [k for k in range(1, kmax + 1, 2)]


def build(name, prep):
    """回傳 (pipeline, 參數網格)。網格可在 config 的 grids 中覆寫。"""
    n = len(prep.Xtr)
    big = n > prep.cfg.get("max_n_kernel", MAX_N_KERNEL)
    if prep.text:
        vec = TfidfVectorizer() if name in ("svm_linear", "logistic") else CountVectorizer()
        if name == "nb_multinomial":
            return (Pipeline([("vec", vec), ("model", MultinomialNB())]), {"model__alpha": [0.1, 0.5, 1.0, 2.0]})
        if name == "svm_linear":
            return (Pipeline([("vec", vec), ("model", LinearSVC(max_iter=10000) if big else SVC(kernel="linear"))]),
                    {"model__C": [2.0 ** e for e in range(-5, 8, 2)]})
        if name == "logistic":
            return (Pipeline([("vec", vec), ("model", LogisticRegression(max_iter=5000))]),
                    {"model__C": [10.0 ** e for e in range(-3, 4)]})
        fail(f"文字資料只支援 nb_multinomial、svm_linear 與 logistic，不支援 {name}")
    if name == "nb_multinomial":
        fail("nb_multinomial 需要在 config 指定 text_column（文字欄位）。")
    if name == "nb_mixed":
        return Pipeline([("model", MixedNB(tuple(prep.num), tuple(prep.cat)))]), {"model__alpha": [0.5, 1.0, 2.0]}
    sd_y = float(prep.ytr.std()) if prep.task == "regression" else 1.0
    scale = name not in ("lda", "lda_shrink", "qda", "nb_gaussian", "linear_reg", "baseline")
    pre = preprocessor(prep, scale=scale, drop_first=name in DROP_FIRST)
    table = {
        "knn": (KNeighborsClassifier(), {"model__n_neighbors": odd_ks(n)}),
        "lda": (LinearDiscriminantAnalysis(), {}),
        "lda_shrink": (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), {}),
        "qda": (QuadraticDiscriminantAnalysis(), {"model__reg_param": [0.0, 0.001, 0.01, 0.1, 0.5]}),
        "nb_gaussian": (GaussianNB(), {}),
        "svm_linear": (LinearSVC(max_iter=10000) if big else SVC(kernel="linear"),
                       {"model__C": [2.0 ** e for e in range(-5, 8, 2)]}),
        # 粗網格：C = 2^-5 … 2^15、γ = 2^-15 … 2^3，以 2^4 為間隔並包含端點；細網格見 tune()
        "svm_rbf": (SVC(kernel="rbf"), {"model__C": [2.0 ** e for e in range(-5, 16, 4)],
                                        "model__gamma": [2.0 ** e for e in sorted(set(range(-15, 4, 4)) | {3})]}),
        "logistic": (LogisticRegression(max_iter=5000), {"model__C": [10.0 ** e for e in range(-3, 4)]}),
        "knn_reg": (KNeighborsRegressor(), {"model__n_neighbors": list(range(1, max(4, int(min(50, np.sqrt(n) * 1.5))) + 1))}),
        "svr": (SVR(kernel="rbf"), {"model__C": [2.0 ** e for e in range(-3, 10, 2)],
                                    "model__gamma": [2.0 ** e for e in range(-7, 4, 2)],
                                    "model__epsilon": [f * sd_y for f in (0.01, 0.05, 0.1, 0.2, 0.5)]}),
        "linear_reg": (LinearRegression(), {}),
    }
    if name not in table:
        fail(f"不支援的方法 {name!r}；可用：{CLASSIFIERS + REGRESSORS}")
    model, grid = table[name]
    if name in ("svm_linear", "svm_rbf", "logistic") and prep.cfg.get("class_weight") == "balanced":
        model.set_params(class_weight="balanced")
    return Pipeline([("pre", pre), ("model", model)]), grid


def scoring_for(prep):
    if prep.task == "regression":
        return "neg_root_mean_squared_error"
    if prep.cfg.get("metric"):
        return prep.cfg["metric"]
    shares = prep.ytr.value_counts(normalize=True)
    if shares.min() >= 0.20:
        return "accuracy"
    # 不平衡：二元用 AUC（與門檻無關，門檻另外決定）；多類別用平衡準確率
    return "roc_auc" if len(shares) == 2 else "balanced_accuracy"


def tune(name, prep, scoring):
    pipe, grid = build(name, prep)
    over = prep.cfg.get("grids", {}).get(name)
    if over:
        grid = {f"model__{k}": v for k, v in over.items()}
    cv = prep.cv()
    gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1).fit(prep.Xtr, prep.ytr)
    coarse = gs
    if name == "svm_rbf" and not over:          # 先粗後細：在最佳點附近以 2 為底細化
        bc, bg = np.log2(gs.best_params_["model__C"]), np.log2(gs.best_params_["model__gamma"])
        fine = {"model__C": [2.0 ** (bc + d) for d in (-2, -1, 0, 1, 2)],
                "model__gamma": [2.0 ** (bg + d) for d in (-2, -1, 0, 1, 2)]}
        gs = GridSearchCV(pipe, fine, cv=cv, scoring=scoring, n_jobs=-1).fit(prep.Xtr, prep.ytr)
        strip = lambda g: {k.replace("model__", ""): v for k, v in g.items()}
        grid = {"粗網格": strip(coarse.param_grid), "細網格": strip(fine)}
    i = gs.best_index_
    edge = []
    for k, v in gs.best_params_.items():
        vals = sorted(set(gs.param_grid[k])) if isinstance(gs.param_grid, dict) else []
        if len(vals) > 1 and v in (vals[0], vals[-1]):
            edge.append(k.replace("model__", ""))
    return {
        "name": name, "estimator": gs.best_estimator_, "search": gs, "coarse": coarse,
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
        "cv_mean": float(gs.cv_results_["mean_test_score"][i]),
        "cv_std": float(gs.cv_results_["std_test_score"][i]),
        "grid": {k.replace("model__", ""): v for k, v in (grid.items() if isinstance(grid, dict) else [])},
        "best_on_edge": edge,
        "failed_settings": [{k.replace("model__", ""): v for k, v in prm.items()}
                            for prm, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"])
                            if not np.isfinite(s)],
    }


# ═════════════════════════════════ 資料概況與方法建議 ═════════════════════════════════
def profile_stats(prep):
    """以訓練資料（已插補）計算統計診斷；不配適任何模型。"""
    X = prep.imputed_train()
    n, p_raw = len(X), len(prep.x)
    p, pf = encoded_p(prep, True), encoded_p(prep, False)
    folds = prep.cfg.get("cv_folds", 5)
    st = {"n_train": n, "n_test": len(prep.Xte), "p_raw": p_raw,
          "p_encoded": p, "p_encoded_full": pf, "n_over_p": n / max(p, 1), "n_over_p_full": n / max(pf, 1),
          "n_numeric": len(prep.num), "n_categorical": len(prep.cat), "text": bool(prep.text),
          "cv_folds": folds, "max_n_kernel": prep.cfg.get("max_n_kernel", MAX_N_KERNEL)}
    if prep.text:
        st["mean_words"] = float(X[prep.text].str.split().str.len().mean())
    if len(prep.num) >= 2:
        r = X[prep.num].corr().abs().to_numpy().copy()
        np.fill_diagonal(r, 0)
        st["max_abs_corr"] = float(np.nanmax(r))
    else:
        st["max_abs_corr"] = None
    if prep.task == "classification":
        counts = prep.ytr.value_counts()
        st["class_counts"] = counts.to_dict()
        st["min_class_n"] = int(counts.min())
        st["min_class_share"] = float(counts.min() / n)
        st["min_class_per_fold"] = st["min_class_n"] * (folds - 1) / folds
        if len(prep.num) >= 1:
            g = [X.loc[prep.ytr == k, prep.num] for k in counts.index]
            sds = np.array([gi.std().to_numpy() for gi in g])
            with np.errstate(divide="ignore", invalid="ignore"):
                st["sd_ratio_max"] = float(np.nanmax(sds.max(0) / sds.min(0)))
            if len(prep.num) >= 2:
                cors = [gi.corr().to_numpy() for gi in g]
                st["corr_diff_max"] = float(max(np.nanmax(np.abs(a - b)) for a in cors for b in cors))
    return st


BENCHMARKS = {"classification": "logistic", "regression": "linear_reg"}   # 比較基準，一律自動加入，不出現在選單中
# 「其他可行方法」的說明：該方法在什麼資料情況下才會被建議
SUGGEST_WHEN = {
    "knn": "p ≤ 10、n/p ≥ 20 且每折最小類別筆數 ≥ 10p 時建議",
    "lda": "連續型為主且 n/p ≥ 10 時建議",
    "lda_shrink": "p 接近 n（n/p < 10）時建議",
    "qda": "每折最小類別筆數 ≥ 10p 時建議",
    "nb_gaussian": "連續型為主、特徵間相關不高，且 p 接近 n 或每類樣本數不足 10p 時建議",
    "nb_mixed": "類別型特徵占一半以上時建議",
    "svm_linear": "文字資料或 p 接近 n 時建議",
    "svm_rbf": "每折最小類別筆數 ≥ 10p 時建議",
    "knn_reg": "p ≤ 10 且 n/p ≥ 20 時建議",
    "svr": "Y 連續時建議",
}


def recommend(prep, st):
    """依 references/decision-tree.md 決定建議方法，並檢查每個方法的可行性。
    回傳（規則表, {"suggested": 建議方法, "other_feasible": 其他可行方法, "infeasible": 不可行方法}）。
    建議依據決定預設勾選哪些方法；可行性條件只剔除數學上無法估計或計算上跑不完的方法。"""
    rules = []

    def add(kind, basis, value, rule, effect):
        rules.append({"性質": kind, "判斷依據": basis, "本資料的數值": value, "依據的規則": rule, "對方法選擇的影響": effect})

    n, p_raw, p, pf = st["n_train"], st["p_raw"], st["p_encoded"], st["p_encoded_full"]
    nk = st["max_n_kernel"]
    big = n > nk
    corr = st.get("max_abs_corr")
    strong = corr is not None and corr >= 0.7
    ratio, ratio_f = n / max(p, 1), n / max(pf, 1)
    knn_ok = pf <= 10 and ratio_f >= 20
    kernel_why = f"訓練資料 n = {n} > max_n_kernel = {nk}：配適時間至少隨 n² 成長，訓練資料超過門檻"
    add("建議依據", "樣本數與變數數", f"n = {n}；原始變數 {p_raw} 個；進入模型的欄位 p = {p}（刪除基準水準）、{pf}（完整編碼）",
        "p 以編碼後進入模型的欄位數計算；n/p < 10 → p 接近 n",
        f"n/p = {ratio:.1f}（LDA、QDA、單純貝氏）；{ratio_f:.1f}（KNN、SVM）")
    add("建議依據", "KNN 適用", f"p = {pf}（完整編碼），n/p = {ratio_f:.1f}", "p ≤ 10 且 n/p ≥ 20 → 可建議 KNN",
        "符合" if knn_ok else "不符合")
    add("可行性條件", "樣本規模", f"訓練資料 n = {n}", f"n > max_n_kernel（{nk}）→ 核 SVM、SVR 不可行",
        "核方法不可行；線性 SVM 改用 LinearSVC" if big else "核方法可行")

    suggested, why = [], {}          # why：建議或移到其他可行方法的理由
    infeasible = {}                  # 代碼 → 不可行原因

    if prep.task == "regression":
        add("建議依據", "Y 型態", "連續", "Y 連續 → KNN 迴歸、SVR；線性迴歸只作比較基準（自動加入）", "")
        if knn_ok:
            suggested = ["knn_reg", "svr"]
            why = {"knn_reg": f"p = {pf} ≤ 10 且 n/p = {ratio_f:.1f} ≥ 20：鄰域內有足夠的點",
                   "svr": "以核函數處理非線性；對離群值較最小平方不敏感"}
        else:
            suggested = ["svr"]
            why = {"svr": "以核函數處理非線性；對離群值較最小平方不敏感",
                   "knn_reg": f"p = {pf}、n/p = {ratio_f:.1f}：特徵多或 n/p 小時 KNN 受維度影響大"}
        if big:
            infeasible["svr"] = kernel_why
            suggested = ["knn_reg"]
            why["knn_reg"] = (why.get("knn_reg", "") + "；" if knn_ok else "") + "SVR 不可行，改建議 KNN 迴歸"
        methods = ["knn_reg", "svr"]
    else:
        K = len(prep.classes)
        mcf = st["min_class_per_fold"]
        add("建議依據", "Y 型態", f"類別，{K} 類", "Y 為類別 → 分類方法；多數類別與 logistic 迴歸為比較基準（自動加入）", "")
        add("建議依據", "類別比例", f"最小類別 {st['min_class_n']} 筆（{st['min_class_share']:.1%}）",
            "最小類別比例 < 10% → 稀有類別", "需調整門檻或 class_weight，以敏感度、精確率、AUC 評估"
            if st["min_class_share"] < 0.10 else "類別大致平衡")
        add("建議依據", "每類樣本數遠大於 p", f"每折最小類別筆數 {mcf:.1f}（最小類別 {st['min_class_n']} 筆 × "
            f"({st['cv_folds']} − 1)/{st['cv_folds']}）；10p = {10 * p}",
            "≥ 10p → 可建議 QDA、RBF 核 SVM", "符合" if mcf >= 10 * p else "不符合")
        add("可行性條件", "QDA 可估計", f"每折最小類別筆數 {mcf:.1f}；p = {p}", "≤ p → QDA 不可行",
            "QDA 不可行" if mcf <= p else "QDA 可行")
        add("可行性條件", "LDA 可估計", f"p = {p}；n = {n}", "p ≥ n → LDA 不可行（改用收縮 LDA）",
            "LDA 不可行" if p >= n else "LDA 可行")
        methods = [c for c in CLASSIFIERS if c != BENCHMARKS["classification"]]
        cat_major = st["n_categorical"] >= 1 and st["n_categorical"] >= p_raw / 2
        if prep.text:
            add("建議依據", "特徵型態", f"文字（詞彙 {pf} 個）", "文字欄位 → 多項單純貝氏、線性核 SVM", "")
            suggested = ["nb_multinomial", "svm_linear"]
            why = {"nb_multinomial": "文字欄位：以字詞次數建模",
                   "svm_linear": "文字轉成高維詞頻向量，線性核適合 p 大的資料"}
            for c in methods:
                if c not in suggested:
                    infeasible[c] = "文字分類只使用文字欄位"
        else:
            add("建議依據", "特徵型態", f"數值 {st['n_numeric']} 個、類別 {st['n_categorical']} 個",
                "類別型特徵 ≥ 全部特徵的一半 → 建議單純貝氏（連續＋類別）", "類別型為主" if cat_major else "連續型為主")
            add("建議依據", "特徵間最大相關", "—" if corr is None else f"{corr:.2f}",
                "≥ 0.7 → 不建議單純貝氏（獨立假設不成立）", "不建議單純貝氏" if strong else "—")
            sdr, cd = st.get("sd_ratio_max"), st.get("corr_diff_max")
            add("建議依據（參考）", "各類散佈", "—" if sdr is None else f"標準差比最大 {sdr:.2f}"
                + ("" if cd is None else f"、相關差異最大 {cd:.2f}"),
                "只當參考，不影響建議", "LDA 與 QDA 的取捨以交叉驗證比較為準")
            if cat_major:
                suggested = ["nb_mixed"]
                why = {"nb_mixed": f"類別型特徵 {st['n_categorical']} 個，占全部 {p_raw} 個特徵的一半以上：各特徵用適合的一維分佈"}
            elif ratio < 10:
                suggested = ["lda_shrink", "svm_linear"] + ([] if strong else ["nb_gaussian"])
                base = f"n/p = {ratio:.1f} < 10：p 接近 n，"
                why = {"lda_shrink": base + "合併共變異矩陣不穩定，以收縮估計",
                       "svm_linear": base + "線性邊界、正規化控制變異",
                       "nb_gaussian": base + "各特徵分開估計一維密度，變異低"}
            elif mcf < 10 * p:
                suggested = ["lda", "nb_gaussian"]
                base = f"n/p = {ratio:.1f} ≥ 10，但每折最小類別筆數 {mcf:.1f} < 10p = {10 * p}：各類共變異矩陣不易穩定估計，"
                why = {"lda": base + "以共同共變異的線性邊界為主",
                       "nb_gaussian": base + "單純貝氏只估計一維密度，變異低"}
            else:
                base = f"n/p = {ratio:.1f} ≥ 10 且每折最小類別筆數 {mcf:.1f} ≥ 10p = {10 * p}："
                suggested = ["lda", "qda", "knn", "svm_rbf"] if knn_ok else ["lda", "qda", "svm_rbf"]
                why = {"lda": base + "線性邊界的基本選擇",
                       "qda": base + "每類樣本足以估計各自的共變異矩陣",
                       "svm_rbf": base + "樣本足以支撐非線性邊界",
                       "knn": base + f"且 p = {pf} ≤ 10、n/p = {ratio_f:.1f} ≥ 20，鄰域內有足夠的點"}
            if strong:
                for c in ("nb_gaussian", "nb_mixed"):
                    why[c] = f"特徵間最大相關 {corr:.2f} ≥ 0.7：特徵間高度相關，獨立假設不成立"
                    if c in suggested:
                        suggested.remove(c)
            # 不可行（硬性條件）
            infeasible["nb_multinomial"] = "沒有文字欄位：只用於文字或計數資料"
            if st["n_categorical"] == 0:
                infeasible["nb_mixed"] = "沒有類別型特徵時與高斯單純貝氏相同"
            if mcf <= p:
                infeasible["qda"] = f"每折最小類別筆數 {mcf:.1f} ≤ p = {p}：某類每折樣本數不超過進入模型的欄位數，共變異矩陣無法估計"
            if p >= n:
                infeasible["lda"] = f"p = {p} ≥ n = {n}：合併共變異矩陣奇異，改用收縮 LDA"
            if big:
                infeasible["svm_rbf"] = kernel_why
    removed = [c for c in suggested if c in infeasible]
    suggested = [c for c in suggested if c not in infeasible]
    if not suggested:
        fallback = "knn_reg" if prep.task == "regression" else "svm_linear"
        suggested = [fallback]
        why[fallback] = ("原本建議的方法" + "、".join(METHOD_NAMES[c] for c in removed) + "不可行" if removed
                         else "依資料特性沒有其他適合建議的方法") + f"，改建議{METHOD_NAMES[fallback]}"
    elif removed:
        for c in suggested:
            why[c] = why.get(c, "") + "（" + "、".join(METHOD_NAMES[r] for r in removed) + " 不可行，已從建議移除）"
    if big and "svm_linear" in suggested:
        why["svm_linear"] += f"；n > {nk}，改用 LinearSVC"
    sel = {
        "suggested": [{"方法": METHOD_NAMES[c], "代碼": c, "理由": why.get(c, "")} for c in suggested],
        "other_feasible": [{"方法": METHOD_NAMES[c], "代碼": c,
                            "理由": why.get(c) or f"可行；{SUGGEST_WHEN[c]}，本資料未達建議條件"}
                           for c in methods if c not in suggested and c not in infeasible],
        "infeasible": [{"方法": METHOD_NAMES[c], "代碼": c, "原因": infeasible[c]} for c in methods if c in infeasible],
    }
    return rules, sel


def run_profile(cfg):
    prep = Prepared(cfg)
    st = profile_stats(prep)
    rules, sel = recommend(prep, st)
    show_table(pd.DataFrame(prep.missing), "缺失與處理（處理前以全部資料計算）")
    show_table(pd.Series({k: v for k, v in st.items() if not isinstance(v, dict)}).to_frame("數值"), "資料概況（訓練資料）")
    show_table(pd.DataFrame(rules), "統計診斷與規則")
    show_table(pd.DataFrame(sel["suggested"]), "建議方法（預設勾選）")
    show_table(pd.DataFrame(sel["other_feasible"]), "其他可行方法（可加選）")
    show_table(pd.DataFrame(sel["infeasible"]), "不可行方法與原因")
    cands = [r["代碼"] for r in sel["suggested"]]
    bench = "多數類別、Logistic 迴歸" if prep.task == "classification" else "線性迴歸"
    out = Path(cfg.get("out_dir", "output"))
    write_json({"profile": st, "rules": rules, "selection": sel, "suggested": cands, "missing": prep.missing},
               out / "profile.json")
    write_json({**cfg, "candidates": cands}, out / "config_template.json")
    print("\n建議方法：", [METHOD_NAMES[c] for c in cands],
          f"已預設填入 config_template.json 的 candidates；比較基準（{bench}）會自動加入。"
          "請使用者確認或加選其他可行方法後，更新 config 的 candidates。")
    return st, rules, sel


# ═════════════════════════════════ EDA ═════════════════════════════════
def pick_numeric(X, y, cols, task, limit):
    """變數太多時，依明確規則挑選：分類取各類平均差（以標準差標準化）最大者；迴歸取與 Y 相關絕對值最大者。"""
    if len(cols) <= limit:
        return list(cols), "全部數值變數"
    if task == "classification":
        sd = X[cols].std().replace(0, np.nan)
        means = X[cols].groupby(y.values).mean()
        score = ((means.max() - means.min()) / sd).fillna(0)
        rule = f"各類平均差÷標準差最大的前 {limit} 個"
    else:
        score = X[cols].corrwith(y.astype(float)).abs().fillna(0)
        rule = f"與 Y 相關係數絕對值最大的前 {limit} 個"
    return list(score.sort_values(ascending=False).index[:limit]), rule


def eda(prep, figs, out):
    X, y = prep.imputed_train(), prep.ytr
    res = {"sample": f"訓練資料 n = {len(X)}；數值變數已依訓練資料中位數插補"}
    tables = Path(out) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    imputed = any(r.get("處理筆數", 0) and "插補" in r.get("處理方式", "") for r in prep.missing)
    tag = "（已插補）" if imputed else ""
    if prep.text:
        lens = X[prep.text].str.split().str.len()
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, k in enumerate(prep.classes):
            ax.hist(lens[y == k], bins=30, alpha=0.6, color=PALETTE[i % 8], label=str(k))
        ax.set_xlabel("字詞數"); ax.set_ylabel("文件數"); ax.legend(frameon=False)
        figs.save(fig, "eda_text_length", f"各類別文件的字詞數分佈（訓練資料 n = {len(X)}）", "eda()")
        vec = CountVectorizer().fit(X[prep.text])
        M = vec.transform(X[prep.text])
        vocab = np.array(vec.get_feature_names_out())
        top = {str(k): ", ".join(vocab[np.asarray(M[(y == k).values].sum(0)).ravel().argsort()[::-1][:15]])
               for k in prep.classes}
        res["top_words"] = top
        pd.Series(top).to_csv(tables / "eda_top_words.csv", encoding="utf-8-sig")
    if prep.num:
        desc = X[prep.num].describe().T
        desc.to_csv(tables / "eda_describe_numeric.csv", encoding="utf-8-sig")
        res["describe_numeric"] = (desc.round(4).astype({"count": int}).astype(object)
                                   .reset_index().rename(columns={"index": "變數"}).to_dict("records"))
        if prep.task == "classification":
            by = X[prep.num].groupby(y.values).agg(["mean", "std"]).round(4)
            by.to_csv(tables / "eda_by_class.csv", encoding="utf-8-sig")
            res["by_class"] = {f"{c[0]}｜{'平均' if c[1] == 'mean' else '標準差'}": by[c].to_dict() for c in by.columns}
        show_table(desc.round(3), "數值變數描述統計（訓練資料）")
        cols, rule = pick_numeric(X, y, prep.num, prep.task, 12)
        res["plotted_numeric"], res["plotted_rule"] = cols, rule
        nc = min(4, len(cols)); nr = int(np.ceil(len(cols) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), squeeze=False)
        for a, c in zip(axes.ravel(), cols):
            a.hist(X[c], bins=30, color=PALETTE[0], alpha=0.85)
            a.set_title(c, fontsize=10)
        for a in axes.ravel()[len(cols):]:
            a.axis("off")
        fig.tight_layout()
        figs.save(fig, "eda_histograms", f"數值變數直方圖{tag}（訓練資料 n = {len(X)}；{rule}）", "eda()")
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.8 * nr), squeeze=False)
        for a, c in zip(axes.ravel(), cols):
            if prep.task == "classification":
                a.boxplot([X.loc[y == k, c] for k in prep.classes], tick_labels=[str(k) for k in prep.classes],
                          flierprops={"markersize": 2})
            else:
                a.boxplot(X[c], flierprops={"markersize": 2})
            a.set_title(c, fontsize=10)
        for a in axes.ravel()[len(cols):]:
            a.axis("off")
        fig.tight_layout()
        by_txt = "依類別分組的" if prep.task == "classification" else ""
        figs.save(fig, "eda_boxplots", f"{by_txt}數值變數箱形圖{tag}（訓練資料 n = {len(X)}；圓點為超過 1.5 倍四分位距的可能離群點）", "eda()")
        if len(prep.num) >= 2:
            cc = prep.num if len(prep.num) <= 30 else pick_numeric(X, y, prep.num, prep.task, 30)[0]
            r = X[cc].corr()
            fig, ax = plt.subplots(figsize=(0.45 * len(cc) + 3, 0.45 * len(cc) + 2.5))
            im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cc)), cc, rotation=90, fontsize=8); ax.set_yticks(range(len(cc)), cc, fontsize=8)
            if len(cc) <= 12:
                for i in range(len(cc)):
                    for j in range(len(cc)):
                        ax.text(j, i, f"{r.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(r.iloc[i, j]) > 0.6 else "black")
            fig.colorbar(im, ax=ax, shrink=0.8, label="相關係數")
            figs.save(fig, "eda_correlation", f"數值變數相關係數熱圖（訓練資料 n = {len(X)}）", "eda()")
            res["corr_high_pairs"] = [
                {"變數1": a, "變數2": b, "相關係數": float(r.loc[a, b])}
                for i, a in enumerate(cc) for b in cc[i + 1:] if abs(r.loc[a, b]) >= 0.7]
        sc = cols[:4]
        if prep.task == "classification" and len(sc) >= 2:
            k = len(sc)
            fig, axes = plt.subplots(k, k, figsize=(2.4 * k, 2.4 * k), squeeze=False)
            for i, a_ in enumerate(sc):
                for j, b_ in enumerate(sc):
                    ax = axes[i, j]
                    if i == j:
                        for t, cl in enumerate(prep.classes):
                            ax.hist(X.loc[y == cl, a_], bins=20, alpha=0.5, color=PALETTE[t % 8])
                    else:
                        for t, cl in enumerate(prep.classes):
                            ax.scatter(X.loc[y == cl, b_], X.loc[y == cl, a_], s=5, alpha=0.5, color=PALETTE[t % 8],
                                       label=str(cl))
                    if i == k - 1: ax.set_xlabel(b_, fontsize=8)
                    if j == 0: ax.set_ylabel(a_, fontsize=8)
            axes[0, 1].legend(frameon=False, fontsize=7, markerscale=2)
            fig.tight_layout()
            figs.save(fig, "eda_scatter", f"依類別上色的散佈圖矩陣{tag}（訓練資料 n = {len(X)}；變數：{', '.join(sc)}）", "eda()")
        elif prep.task == "regression":
            k = min(6, len(cols))
            fig, axes = plt.subplots(1, k, figsize=(3 * k, 3), squeeze=False)
            for a, c in zip(axes.ravel(), cols[:k]):
                a.scatter(X[c], y, s=5, alpha=0.5, color=PALETTE[0]); a.set_xlabel(c)
            axes[0, 0].set_ylabel(prep.y_name)
            fig.tight_layout()
            figs.save(fig, "eda_scatter", f"各數值變數對 {prep.y_name} 的散佈圖{tag}（訓練資料 n = {len(X)}）", "eda()")
    if prep.cat:
        cc = prep.cat[:6]
        fig, axes = plt.subplots(1, len(cc), figsize=(3.6 * len(cc), 3.4), squeeze=False)
        res["categorical_counts"] = {}
        for a, c in zip(axes.ravel(), cc):
            counts = X[c].astype(str).value_counts()
            res["categorical_counts"][c] = counts.to_dict()
            if prep.task == "classification":
                tab = pd.crosstab(X[c].astype(str), y, normalize="index")
                bottom = np.zeros(len(tab))
                for t, cl in enumerate(tab.columns):
                    a.bar(tab.index, tab[cl], bottom=bottom, color=PALETTE[t % 8], label=str(cl))
                    bottom += tab[cl].to_numpy()
                a.set_ylabel("各類別比例")
            else:
                a.bar(counts.index, counts.values, color=PALETTE[0]); a.set_ylabel("筆數")
            a.set_title(c, fontsize=10); a.tick_params(axis="x", rotation=45)
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        figs.save(fig, "eda_categorical", f"類別變數{'在各水準下的 Y 類別比例' if prep.task == 'classification' else '次數'}"
                  f"（訓練資料 n = {len(X)}）", "eda()")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if prep.task == "classification":
        vc = y.value_counts().reindex(prep.classes)
        bars = ax.bar([str(k) for k in vc.index], vc.values, color=PALETTE[:len(vc)])
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}（{v / len(y):.1%}）", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("筆數"); ax.set_xlabel(prep.y_name)
        figs.save(fig, "eda_y", f"{prep.y_name} 各類別的筆數與比例（訓練資料 n = {len(y)}）", "eda()")
    else:
        ax.hist(y, bins=30, color=PALETTE[0]); ax.set_xlabel(prep.y_name); ax.set_ylabel("筆數")
        figs.save(fig, "eda_y", f"{prep.y_name} 的分佈（訓練資料 n = {len(y)}）", "eda()")
        res["describe_y"] = y.describe().to_dict()
    miss = pd.DataFrame(prep.missing)
    if miss["缺失筆數"].sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(miss) + 1.2))
        ax.barh(miss["變數"], miss["缺失比例"], color=PALETTE[1])
        ax.set_xlabel("缺失比例（處理前，全部資料）"); ax.invert_yaxis()
        ax.axvline(0.2, color="grey", lw=0.8, ls="--")
        figs.save(fig, "eda_missing", "處理前各變數缺失比例（虛線為 20%）；處理後進入模型的變數皆無缺失", "eda()")
    return res


# ═════════════════════════════════ 評估 ═════════════════════════════════
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def scores_of(est, X, prep):
    """正類分數：有機率用機率，否則用決策函數（只用於 ROC／AUC）。"""
    if hasattr(est, "predict_proba"):
        try:
            P = est.predict_proba(X)
            classes = list(est.classes_)
            return P, "proba", classes
        except AttributeError:
            pass
    if hasattr(est, "decision_function"):
        return est.decision_function(X), "decision", list(est.classes_)
    return None, None, None


def classify_metrics(prep, est, threshold):
    Xte, yte = prep.Xte, prep.yte
    S, kind, classes = scores_of(est, Xte, prep)
    if prep.pos and kind == "proba":
        pos_col = classes.index(prep.pos)
        neg = [c for c in classes if c != prep.pos][0]
        pred = np.where(S[:, pos_col] >= threshold, prep.pos, neg)
    else:
        pred = est.predict(Xte)
    pred = np.asarray(pred).astype(str)
    yv = yte.to_numpy().astype(str)
    correct = int((pred == yv).sum())
    lo, hi = wilson(correct, len(yv))
    m = {"accuracy": correct / len(yv), "accuracy_ci95": [lo, hi], "error_rate": 1 - correct / len(yv),
         "balanced_accuracy": balanced_accuracy_score(yv, pred), "n_test": len(yv)}
    labels = prep.classes
    cm = confusion_matrix(yv, pred, labels=labels)
    m["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}
    if prep.pos:
        m["sensitivity"] = recall_score(yv, pred, pos_label=prep.pos, zero_division=0)
        neg = [c for c in labels if c != prep.pos][0]
        m["specificity"] = recall_score(yv, pred, pos_label=neg, zero_division=0)
        m["precision"] = precision_score(yv, pred, pos_label=prep.pos, zero_division=0)
        m["f1"] = f1_score(yv, pred, pos_label=prep.pos, zero_division=0)
        if kind is not None:
            s = S[:, classes.index(prep.pos)] if kind == "proba" else (S if classes[1] == prep.pos else -S)
            m["auc"] = roc_auc_score((yv == prep.pos).astype(int), s)
            m["_roc"] = roc_curve((yv == prep.pos).astype(int), s)[:2]
            m["_score"] = s
    else:
        m["recall_by_class"] = dict(zip(labels, recall_score(yv, pred, labels=labels, average=None, zero_division=0)))
        m["precision_by_class"] = dict(zip(labels, precision_score(yv, pred, labels=labels, average=None, zero_division=0)))
        m["f1_macro"] = f1_score(yv, pred, average="macro")
        if kind == "proba":
            try:
                m["auc_ovr"] = roc_auc_score(yv, S, multi_class="ovr", labels=classes)
            except ValueError:
                pass
    m["_pred"] = pred
    m["score_kind"] = kind
    return m


def regress_metrics(prep, est):
    pred = est.predict(prep.Xte)
    yv = prep.yte.to_numpy().astype(float)
    return {"rmse": float(np.sqrt(mean_squared_error(yv, pred))), "mae": float(mean_absolute_error(yv, pred)),
            "r2": float(r2_score(yv, pred)), "n_test": len(yv), "_pred": pred}


def mcnemar(pred_a, pred_b, y):
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"only_first_correct": b, "only_second_correct": c, "p_value": float(p),
            "H0": "兩模型在測試集上的錯誤率相同", "alpha": 0.05}


def threshold_table(prep, est):
    """以訓練資料的交叉驗證（out-of-fold）預測機率計算不同門檻下的表現，不使用測試集。"""
    if not prep.pos or not hasattr(est, "predict_proba"):
        return None
    try:
        P = cross_val_predict(clone(est), prep.Xtr, prep.ytr, cv=prep.cv(), method="predict_proba")
    except (AttributeError, ValueError):
        return None
    classes = sorted(prep.ytr.unique())
    s = P[:, classes.index(prep.pos)]
    yv = (prep.ytr.to_numpy() == prep.pos)
    rows = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        pr = s >= t
        tp, fp = int((pr & yv).sum()), int((pr & ~yv).sum())
        fn = int((~pr & yv).sum())
        rows.append({"門檻": t, "敏感度": tp / max(tp + fn, 1), "精確率": tp / max(tp + fp, 1) if tp + fp else None,
                     "整體錯誤率": float((pr != yv).mean()), "預測為正類筆數": int(pr.sum())})
    return rows


# ═════════════════════════════════ 診斷圖 ═════════════════════════════════
def diagnostics(prep, fits, final, figs, metrics):
    out = {}
    for name, f in fits.items():
        gs = f["coarse"]
        if name in ("knn", "knn_reg"):
            ks = gs.cv_results_["param_model__n_neighbors"].data.astype(int)
            mu, sd = gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"]
            fig, ax = plt.subplots(figsize=(6.5, 3.4))
            ax.errorbar(ks, mu, yerr=sd / np.sqrt(prep.cv().get_n_splits()), fmt="o-", ms=3, color=PALETTE[0], capsize=2)
            ax.axvline(f["best_params"]["n_neighbors"], color=PALETTE[1], ls="--", lw=1)
            ax.set_xlabel("k（鄰居數）"); ax.set_ylabel(f"CV {gs.scoring}")
            figs.save(fig, f"diag_{name}_k", f"{METHOD_NAMES[name]}：交叉驗證表現對 k 的曲線（±1 標準誤；虛線為選出的 k）",
                      "diagnostics()")
        if name == "svm_rbf":
            res = pd.DataFrame(gs.cv_results_)
            piv = res.pivot_table(index="param_model__C", columns="param_model__gamma", values="mean_test_score")
            fig, ax = plt.subplots(figsize=(6, 4.2))
            im = ax.imshow(piv.values, origin="lower", cmap="Blues", aspect="auto")
            ax.set_xticks(range(piv.shape[1]), [f"{np.log2(v):.0f}" for v in piv.columns])
            ax.set_yticks(range(piv.shape[0]), [f"{np.log2(v):.0f}" for v in piv.index])
            ax.set_xlabel("log2 γ"); ax.set_ylabel("log2 C")
            fig.colorbar(im, ax=ax, label=f"CV {gs.scoring}")
            figs.save(fig, "diag_svm_rbf_grid", "RBF 核 SVM：(C, γ) 粗網格的交叉驗證表現（顏色越深越好）", "diagnostics()")
            out["svm_rbf_n_support"] = int(f["estimator"]["model"].n_support_.sum())
        if name == "svm_linear" and not prep.text and hasattr(f["estimator"]["model"], "n_support_"):
            out["svm_linear_n_support"] = int(f["estimator"]["model"].n_support_.sum())   # LinearSVC 沒有支援向量個數
        if name == "svr":
            out["svr_n_support"] = int(len(f["estimator"]["model"].support_))
        if name in ("lda", "lda_shrink") and name == final and not prep.text:
            est = f["estimator"]
            if hasattr(est["model"], "explained_variance_ratio_") and name == "lda":
                out["lda_explained_variance_ratio"] = est["model"].explained_variance_ratio_.tolist()
            if name == "lda":
                Z = est.transform(prep.Xtr)
                fig, ax = plt.subplots(figsize=(6.5, 3.8))
                for t, cl in enumerate(prep.classes):
                    zz = Z[prep.ytr.to_numpy() == cl]
                    if Z.shape[1] >= 2:
                        ax.scatter(zz[:, 0], zz[:, 1], s=8, alpha=0.6, color=PALETTE[t % 8], label=str(cl))
                    else:
                        ax.hist(zz[:, 0], bins=30, alpha=0.5, color=PALETTE[t % 8], label=str(cl))
                ax.set_xlabel("區別座標 1"); ax.set_ylabel("區別座標 2" if Z.shape[1] >= 2 else "筆數")
                ax.legend(frameon=False)
                figs.save(fig, "diag_lda_coordinates", f"LDA 區別座標（訓練資料 n = {len(Z)}）", "diagnostics()")
    if prep.task == "classification":
        m = metrics[final]
        cm = np.array(m["confusion_matrix"]["matrix"])
        fig, ax = plt.subplots(figsize=(0.8 * len(prep.classes) + 2, 0.8 * len(prep.classes) + 1.6))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(prep.classes)), prep.classes); ax.set_yticks(range(len(prep.classes)), prep.classes)
        ax.set_xlabel("預測類別"); ax.set_ylabel("實際類別")
        pos = f"；正類 = {prep.pos}" if prep.pos else ""
        figs.save(fig, "diag_confusion", f"{METHOD_NAMES[final]} 的混淆矩陣（測試資料 n = {m['n_test']}{pos}）", "diagnostics()")
        if prep.pos:
            fig, ax = plt.subplots(figsize=(5, 4.4))
            for t, (name, mm) in enumerate(metrics.items()):
                if "_roc" in mm:
                    fpr, tpr = mm["_roc"]
                    ax.plot(fpr, tpr, color=PALETTE[t % 8], lw=1.4, label=f"{METHOD_NAMES[name]}（AUC {mm['auc']:.3f}）")
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
            ax.set_xlabel("1 − 特異度"); ax.set_ylabel("敏感度"); ax.legend(frameon=False, fontsize=8)
            figs.save(fig, "diag_roc", f"測試資料的 ROC 曲線（n = {m['n_test']}；正類 = {prep.pos}）", "diagnostics()")
            fm = metrics[final]
            if fm.get("score_kind") == "proba":
                s = fm["_score"]; yv = prep.yte.to_numpy() == prep.pos
                bins = np.linspace(0, 1, 6); idx = np.clip(np.digitize(s, bins) - 1, 0, 4)
                cal = [{"機率區間": f"{bins[b]:.1f}–{bins[b + 1]:.1f}", "平均預測機率": float(s[idx == b].mean()),
                        "實際正類比例": float(yv[idx == b].mean()), "筆數": int((idx == b).sum())}
                       for b in range(5) if (idx == b).sum() > 0]
                out["calibration"] = cal
                fig, ax = plt.subplots(figsize=(4.6, 4.2))
                ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
                ax.plot([c["平均預測機率"] for c in cal], [c["實際正類比例"] for c in cal], "o-", color=PALETTE[0])
                ax.set_xlabel("平均預測機率"); ax.set_ylabel("實際正類比例")
                figs.save(fig, "diag_calibration", f"{METHOD_NAMES[final]} 的機率校準（測試資料；虛線為完美校準）",
                          "diagnostics()")
    else:
        pred = metrics[final]["_pred"]; yv = prep.yte.to_numpy().astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].scatter(yv, pred, s=8, alpha=0.6, color=PALETTE[0])
        lim = [min(yv.min(), pred.min()), max(yv.max(), pred.max())]
        axes[0].plot(lim, lim, color="grey", ls="--", lw=0.8)
        axes[0].set_xlabel("實際值"); axes[0].set_ylabel("預測值")
        axes[1].scatter(pred, yv - pred, s=8, alpha=0.6, color=PALETTE[0]); axes[1].axhline(0, color="grey", lw=0.8)
        axes[1].set_xlabel("預測值"); axes[1].set_ylabel("殘差")
        fig.tight_layout()
        figs.save(fig, "diag_regression", f"{METHOD_NAMES[final]}：預測值對實際值與殘差圖（測試資料 n = {len(yv)}）",
                  "diagnostics()")
    return out


# ═════════════════════════════════ 主分析 ═════════════════════════════════
def run(cfg, config_path):
    out = Path(cfg.get("out_dir", "output"))
    out.mkdir(parents=True, exist_ok=True)
    fonts = setup_fonts()
    figs = FigureSaver(out)
    prep = Prepared(cfg)
    cands = cfg.get("candidates") or []
    if not cands:
        fail("config 的 candidates 為空：請先執行 profile，並經使用者確認候選方法後填入。")
    valid = CLASSIFIERS if prep.task == "classification" else REGRESSORS
    bad = [c for c in cands if c not in valid]
    if bad:
        fail(f"{bad} 不適用於 {prep.task}；可用：{valid}")
    st = profile_stats(prep)
    rules, sel = recommend(prep, st)
    suggested = [r["代碼"] for r in sel["suggested"]]
    blocked = [r for r in sel["infeasible"] if r["代碼"] in cands]
    if blocked:
        fail("candidates 中有不可行的方法：\n" + "\n".join(f"  {r['方法']}（{r['代碼']}）：{r['原因']}" for r in blocked))
    bench = BENCHMARKS[prep.task]
    eda_res = eda(prep, figs, out)
    scoring = scoring_for(prep)
    fits, unestimable = {}, []
    for c in [c for c in cands if c != bench] + [bench]:      # 比較基準一律自動加入
        print(f"\n調參中：{METHOD_NAMES[c]} …")
        try:
            f = tune(c, prep, scoring)
            reason = None if np.isfinite(f["cv_mean"]) else "所有參數組合的交叉驗證分數皆無法計算"
        except Exception as e:                  # 例如共變異矩陣奇異、所有參數組合都配適失敗
            f, reason = None, f"{type(e).__name__}：{str(e).strip().splitlines()[0][:200] if str(e).strip() else '配適失敗'}"
        if reason:
            unestimable.append({"方法": METHOD_NAMES[c], "代碼": c, "原因": reason})
            print(f"[注意] {METHOD_NAMES[c]} 無法估計，已從比較中略過：{reason}")
            continue
        fits[c] = f
    if not fits:
        fail("所有候選方法都無法估計，請檢查資料或與使用者確認其他候選方法。")
    # 最終模型從本 skill 的方法中選 CV 最佳者；logistic、線性迴歸只作比較基準
    pool = [k for k in fits if k not in ("logistic", "linear_reg")] or list(fits)
    final = max(pool, key=lambda k: fits[k]["cv_mean"])
    base = (DummyClassifier(strategy="most_frequent") if prep.task == "classification" else DummyRegressor()).fit(prep.Xtr, prep.ytr)
    base_cv = float(cross_val_score(base, prep.Xtr, prep.ytr, cv=prep.cv(), scoring=scoring).mean())
    metrics = {}
    thr = cfg.get("threshold", 0.5)
    for c, f in fits.items():
        metrics[c] = classify_metrics(prep, f["estimator"], thr) if prep.task == "classification" else regress_metrics(prep, f["estimator"])
    metrics["baseline"] = classify_metrics(prep, base, 0.5) if prep.task == "classification" else regress_metrics(prep, base)
    diag = diagnostics(prep, fits, final, figs, metrics)
    tests = {}
    if prep.task == "classification":
        yv = prep.yte.to_numpy().astype(str)
        order = [final] + sorted([k for k in fits if k != final], key=lambda k: -fits[k]["cv_mean"])
        if len(order) >= 2:
            tests["mcnemar_final_vs_second"] = {"first": order[0], "second": order[1],
                                                **mcnemar(metrics[order[0]]["_pred"], metrics[order[1]]["_pred"], yv)}
        tests["mcnemar_final_vs_baseline"] = {"first": final, "second": "baseline",
                                              **mcnemar(metrics[final]["_pred"], metrics["baseline"]["_pred"], yv)}
    thr_tab = threshold_table(prep, fits[final]["estimator"]) if prep.task == "classification" else None
    table = []
    for c in list(fits) + ["baseline"]:
        m = metrics[c]
        row = {"方法": METHOD_NAMES[c],
               "CV（平均）": fits[c]["cv_mean"] if c in fits else base_cv,
               "CV（標準差）": fits[c]["cv_std"] if c in fits else None}
        if prep.task == "classification":
            row.update({"測試準確率": m["accuracy"], "準確率 95% CI": m["accuracy_ci95"],
                        "平衡準確率": m["balanced_accuracy"]})
            zh = {"sensitivity": "敏感度", "specificity": "特異度", "precision": "精確率", "f1": "F1",
                  "auc": "AUC", "f1_macro": "F1（各類平均）", "auc_ovr": "AUC（一對多）"}
            for k, lab in zh.items():
                if k in m:
                    row[lab] = m[k]
        else:
            row.update({"RMSE": m["rmse"], "MAE": m["mae"], "R²": m["r2"]})
        table.append(row)
    show_table(pd.DataFrame(table).round(4), f"模型比較（CV 指標：{scoring}；最終模型：{METHOD_NAMES[final]}）")
    import sklearn, matplotlib, scipy
    results = {
        "config": cfg,
        "data": {"path": str(cfg["data_path"]), "sha256": sha256(cfg["data_path"]), "raw_rows": prep.raw_n,
                 "raw_cols": prep.raw_cols, "rows_after": prep.n_final, "x_after": prep.p_final,
                 "x_before": len(prep.x_all), "y_rows_dropped": prep.y_dropped_rows,
                 "n_train": len(prep.Xtr), "n_test": len(prep.Xte), "task": prep.task,
                 "classes": prep.classes, "positive_class": prep.pos,
                 "numeric": prep.num, "categorical": prep.cat, "text_column": prep.text},
        "missing": prep.missing, "profile": st, "rules": rules, "selection": sel,
        "suggested": suggested, "candidates": cands, "benchmark": bench, "unestimable": unestimable,
        "scoring": scoring, "eda": eda_res,
        "tuning": {c: {k: f[k] for k in ("best_params", "cv_mean", "cv_std", "grid", "best_on_edge", "failed_settings")} for c, f in fits.items()},
        "baseline_cv": base_cv, "final_model": final,
        "test": {c: {k: v for k, v in m.items() if not k.startswith("_")} for c, m in metrics.items()},
        "comparison_table": table, "tests": tests, "threshold_table_cv": thr_tab, "diagnostics": diag,
        "figures": figs.index,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "fonts": fonts},
        "commands": [f"python run_analysis.py run --config {config_path}"],
    }
    write_json(results, out / "results.json")
    draft = write_draft(results, prep)
    (out / "report_draft.md").write_text(draft, encoding="utf-8")
    print(f"已寫出報告草稿：{(out / 'report_draft.md').resolve()}")
    return results


# ═════════════════════════════════ 報告草稿 ═════════════════════════════════
TODO = "【待填："


def fmt(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{v[0]:.{d}f}–{v[1]:.{d}f}"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def md_table(rows, cols=None):
    if not rows:
        return "（無）\n"
    cols = cols or list(rows[0].keys())
    out = "| " + " | ".join(cols) + " |\n|" + "|".join([":--"] * len(cols)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(fmt(r.get(c)) if not isinstance(r.get(c), str) else r.get(c) for c in cols) + " |\n"
    return out


def fig_md(results, name):
    f = results["figures"].get(name)
    if not f:
        return ""
    return (f"![{f['caption']}]({f['file']}){{width=88%}}\n\n"
            f"*（圖源：run_analysis.py 的 {f['source']}，檔案 {f['file']}）*\n\n"
            f"> **解讀：**{TODO}依圖說明分佈、偏態、可能離群點、變數關係或模型表現，以及對研究問題與後續選模的意義】\n\n")


def write_draft(r, prep):
    d, cfg = r["data"], r["config"]
    rep = cfg.get("report", {})
    task_zh = "分類" if d["task"] == "classification" else "迴歸"
    L = [f"# {rep.get('title', '分析報告')}\n"]
    L.append("## 1. 研究問題\n")
    L.append(f"- 背景：{rep.get('background') or TODO + '背景；未知時寫「未提供」】'}\n"
             f"- 觀測單位：{rep.get('unit') or TODO + '一列代表什麼；未知時寫「未提供」】'}\n"
             f"- 反應變數 Y：`{cfg['y']}`（{task_zh}"
             + (f"，類別：{', '.join(d['classes'])}" if d["classes"] else "")
             + (f"；正類 = {d['positive_class']}" if d["positive_class"] else "") + "）\n"
             f"- 解釋變數 X（{d['x_after']} 個）：{', '.join('`' + c + '`' for c in (prep.x))}\n"
             f"- 分析目的：{rep.get('goal') or TODO + '分析目的】'}\n\n")
    L.append("## 2. 重點結果與建議\n")
    L.append(f"{TODO}3–5 點：先回答研究問題；最終方法與理由；測試集主要指標與基準模型比較（含信賴區間或檢定結論）；"
             "最常見的錯誤型態；建議及其適用前提。數字一律取自 results.json】\n\n")
    L.append("## 3. 資料與方法\n")
    L.append("### 3.1 樣本與變數\n")
    L.append(md_table([{"項目": "原始資料", "筆數": str(d["raw_rows"]), "變數數（選定 X）": str(d["x_before"])},
                       {"項目": "處理後（進入模型）", "筆數": str(d["rows_after"]), "變數數（選定 X）": str(d["x_after"])},
                       {"項目": "訓練資料", "筆數": str(d["n_train"]), "變數數（選定 X）": str(d["x_after"])},
                       {"項目": "測試資料", "筆數": str(d["n_test"]), "變數數（選定 X）": str(d["x_after"])}]))
    if d["task"] == "classification":
        cc = r["profile"]["class_counts"]
        L.append("\n訓練資料各類別筆數：" + "、".join(f"{k} {v} 筆（{v / d['n_train']:.1%}）" for k, v in cc.items()) + "。\n")
    L.append(f"\n> **解讀：**{TODO}樣本與變數規模對分析的意義】\n\n")
    L.append("### 3.2 缺失與處理\n")
    rows = [{"變數": m["變數"], "缺失筆數": str(m["缺失筆數"]), "缺失比例": f"{m['缺失比例']:.1%}",
             "處理方式": m.get("處理方式", ""), "處理筆數": str(m.get("處理筆數", 0))} for m in r["missing"]]
    L.append(md_table(rows))
    L.append(f"\n處理前 {d['raw_rows']} 筆、{d['x_before']} 個 X；處理後 {d['rows_after']} 筆、{d['x_after']} 個 X。"
             "插補值只以訓練資料估計，交叉驗證每一折重新估計。\n\n")
    L.append(fig_md(r, "eda_missing"))
    L.append("### 3.3 前處理\n")
    L.append(f"{TODO}列出實際採用的前處理（標準化、類別編碼、文字轉詞頻）與理由；說明皆在 Pipeline 中只以訓練資料估計】\n\n")
    L.append("### 3.4 模型選用理由\n")
    L.append("**統計診斷與規則**\n\n" + md_table(r["rules"]))
    sel = r["selection"]
    L.append("\n**建議方法與理由**\n\n" + md_table(sel["suggested"], ["方法", "代碼", "理由"]))
    L.append("\n**其他可行方法**\n\n" + md_table(sel["other_feasible"], ["方法", "代碼", "理由"]))
    L.append("\n**不可行方法與原因**\n\n" + md_table(sel["infeasible"], ["方法", "代碼", "原因"]))
    bench = "多數類別基準模型與 Logistic 迴歸" if d["task"] == "classification" else "基準模型（以平均數預測）與線性迴歸"
    added = [c for c in r["candidates"] if c not in r["suggested"] and c != r["benchmark"]]
    L.append(f"\n使用者最後確認的候選方法：{'、'.join(METHOD_NAMES[c] for c in r['candidates'] if c != r['benchmark'])}"
             + (f"（其中 {'、'.join(METHOD_NAMES[c] for c in added)} 為加選的其他可行方法）" if added else "")
             + f"；比較用的{bench}自動加入。邊界形狀（線性是否足夠）在第 5 章的模型比較中解讀。\n\n")
    if r["unestimable"]:
        L.append("**無法估計的方法（已從比較中略過）**\n\n" + md_table(r["unestimable"], ["方法", "代碼", "原因"]) + "\n")
    L.append(f"> **解讀：**{TODO}逐列說明這些資料特性如何決定建議方法、其他可行方法與不可行方法】\n\n")
    L.append("### 3.5 資料切分\n")
    L.append(f"訓練／測試 = {1 - cfg.get('test_size', 0.25):.0%}／{cfg.get('test_size', 0.25):.0%}，"
             + ("依時間順序切分（不打亂）" if cfg.get("time_order") else
                ("分層隨機抽樣" if d["task"] == "classification" else "隨機抽樣"))
             + f"，隨機種子 {cfg.get('seed', 2026)}。測試資料只在最後評估時使用一次。\n\n")
    L.append("### 3.6 交叉驗證與參數搜尋\n")
    L.append(f"{cfg.get('cv_folds', 5)} 折" + ("時間序列交叉驗證" if cfg.get("time_order") else
                                             ("分層交叉驗證（先打亂）" if d["task"] == "classification" else "交叉驗證（先打亂）"))
             + f"，選模指標：`{r['scoring']}`。\n\n")
    trows = []
    for c, t in r["tuning"].items():
        trows.append({"方法": METHOD_NAMES[c],
                      "搜尋範圍": "；".join(f"{k}: {fmt_grid(v)}" for k, v in t["grid"].items()) or "無需調參",
                      "選出的值": "；".join(f"{k} = {fmt(v, 4)}" for k, v in t["best_params"].items()) or "—",
                      "CV（平均 ± 標準差）": f"{t['cv_mean']:.3f} ± {t['cv_std']:.3f}",
                      "最佳值在網格邊緣": "、".join(t["best_on_edge"]) or "否"})
    L.append(md_table(trows))
    L.append(f"\n> **解讀：**{TODO}參數選擇結果的意義；若最佳值在網格邊緣須說明】\n\n")
    L.append("## 4. EDA 探索性資料分析\n")
    L.append(f"以下皆為訓練資料（{r['eda']['sample']}）。\n\n")
    if r["eda"].get("describe_numeric"):
        cols = ["變數", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        L.append("**數值變數描述統計**\n\n" + md_table(r["eda"]["describe_numeric"][:30], cols) + "\n")
        L.append(f"> **解讀：**{TODO}尺度差異、偏態與可能離群點】\n\n")
    if r["eda"].get("top_words"):
        L.append("**各類別最常見的字詞**\n\n" + md_table([{"類別": k, "字詞": v} for k, v in r["eda"]["top_words"].items()]) + "\n")
    for name in ("eda_y", "eda_histograms", "eda_boxplots", "eda_categorical", "eda_scatter", "eda_correlation", "eda_text_length"):
        L.append(fig_md(r, name))
    L.append(f"**EDA 對模型推薦的意義：**{TODO}綜合上述觀察，說明對 3.4 方法選擇的支持或疑慮】\n\n")
    L.append("## 5. 模型結果與診斷\n")
    L.append("### 5.1 模型比較\n")
    L.append(md_table(r["comparison_table"]))
    bench = [c for c in ("logistic", "linear_reg") if c in r["tuning"]
             and r["tuning"][c]["cv_mean"] >= r["tuning"][r["final_model"]]["cv_mean"]]
    if bench:
        L.append(f"\n比較用的 {METHOD_NAMES[bench[0]]} 在 CV 上不低於最終模型：須在結果與限制中說明，"
                 "線性模型的預測表現與本 skill 的方法相當。\n")
    if r["unestimable"]:
        L.append("\n無法估計而略過的方法：" + "、".join(f"{u['方法']}（{u['原因']}）" for u in r["unestimable"]) + "。\n")
    L.append(f"\n最終模型（本流程方法中 CV 表現最佳）：**{METHOD_NAMES[r['final_model']]}**；基準模型 CV = {r['baseline_cv']:.3f}。"
             "CV 欄為訓練資料的交叉驗證結果，其餘為測試資料結果。\n\n"
             f"> **解讀：**{TODO}比較各方法與基準；說明準確率信賴區間的意義】\n\n")
    if r["tests"]:
        L.append("### 5.2 模型比較的檢定（McNemar）\n")
        trs = [{"比較": f"{METHOD_NAMES[t['first']]} vs {METHOD_NAMES[t['second']]}",
                "只有前者分對": str(t["only_first_correct"]), "只有後者分對": str(t["only_second_correct"]),
                "p 值": f"{t['p_value']:.4f}", "虛無假設": t["H0"]} for t in r["tests"].values()]
        L.append(md_table(trs))
        L.append(f"\n> **解讀：**{TODO}以 α = 0.05 寫出結論：p < 0.05 時表示錯誤率差異達統計顯著並說明代表什麼；"
                 "p ≥ 0.05 時寫「沒有足夠證據顯示有差異」】\n\n")
    if r["data"]["task"] == "classification":
        L.append("### 5.3 混淆矩陣與錯誤型態\n")
        L.append(fig_md(r, "diag_confusion"))
        L.append(fig_md(r, "diag_roc"))
        if r["threshold_table_cv"]:
            L.append("**不同門檻下的表現（訓練資料交叉驗證預測，不使用測試集）**\n\n" + md_table(r["threshold_table_cv"]) +
                     f"\n本次採用門檻 {r['config'].get('threshold', 0.5)}。\n\n> **解讀：**{TODO}門檻選擇與誤判成本】\n\n")
        L.append(fig_md(r, "diag_calibration"))
    else:
        L.append(fig_md(r, "diag_regression"))
    L.append("### 5.4 方法特定診斷\n")
    for name in r["figures"]:
        if name.startswith("diag_") and name not in ("diag_confusion", "diag_roc", "diag_calibration", "diag_regression"):
            L.append(fig_md(r, name))
    labels = {"svm_rbf_n_support": "RBF 核 SVM 的支援向量個數", "svm_linear_n_support": "線性核 SVM 的支援向量個數",
              "svr_n_support": "SVR 的支援向量個數", "lda_explained_variance_ratio": "LDA 各區別方向承載的組間變異比例"}
    extra = {k: v for k, v in r["diagnostics"].items() if k != "calibration"}
    if extra:
        L.append("".join(f"- {labels.get(k, k)}：{fmt(v) if not isinstance(v, list) else ', '.join(fmt(x) for x in v)}"
                         f"（訓練資料 n = {d['n_train']}）\n" for k, v in extra.items()) + "\n")
    if r["diagnostics"].get("calibration"):
        L.append("**機率校準（測試資料）**\n\n" + md_table(r["diagnostics"]["calibration"]) + "\n")
    L.append("## 6. 限制\n")
    L.append(f"{TODO}遺漏變數、代表性、模型假設、測試集大小造成的波動、類別不平衡與門檻、缺失與插補的影響、適用邊界；不作因果宣稱】\n\n")
    L.append("## 7. 附錄\n")
    L.append(f"### 7.1 方法公式\n\n{TODO}依 references/methods.md 列出本次實際使用方法的公式】\n\n")
    env = r["environment"]
    L.append("### 7.2 重現資訊\n\n"
             f"- Python {env['python']}；numpy {env['numpy']}、pandas {env['pandas']}、scikit-learn {env['scikit-learn']}、"
             f"scipy {env['scipy']}、matplotlib {env['matplotlib']}；平台：{env['platform']}\n"
             f"- 圖表字型：{'、'.join(env['fonts']['family'])}"
             + ("" if env["fonts"]["cjk"] else f"（警告：{NO_CJK_WARNING}）") + "\n"
             "- 報告字型：HTML 依字型清單由瀏覽器選用（優先 Times New Roman 與標楷體）\n"
             f"- 資料檔：`{d['path']}`，SHA-256 = `{d['sha256']}`\n"
             f"- 隨機種子：{cfg.get('seed', 2026)}；執行指令：`{r['commands'][0]}`\n"
             "- 浮點運算、套件版本與字型可能造成些微差異。\n\n")
    text = "".join(L)
    text = re.sub(r"^(#{1,4} [^\n]*)\n(?!\n)", r"\1\n\n", text, flags=re.M)   # 標題後空一行（pandoc 需要）
    return re.sub(r"([^\n|])\n(\|)", r"\1\n\n\2", text)                       # 表格前空一行


def fmt_grid(v):
    if isinstance(v, dict):
        return "；".join(f"{k}: {fmt_grid(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        vals = list(v)
        if all(isinstance(x, (int, float)) and x > 0 for x in vals) and len(vals) > 2:
            lg = np.log2(vals)
            if np.allclose(lg, np.round(lg)):
                return f"2^{int(lg[0])} … 2^{int(lg[-1])}（{len(vals)} 個值）"
            return f"{fmt(min(vals), 4)} … {fmt(max(vals), 4)}（{len(vals)} 個值）"
        return ", ".join(fmt(x, 4) for x in vals)
    return str(v)


# ═════════════════════════════════ 完成報告 ═════════════════════════════════
CJK_PDF_FONTS = ["DFKai-SB", "BiauKai", "TW-Kai", "AR PL UKai TW", "Noto Serif CJK TC", "Noto Sans CJK TC"]
LATIN_PDF_FONTS = ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"]
DETECT_TIMEOUT, CONVERT_TIMEOUT, XELATEX_TIMEOUT = 10, 300, 300     # 秒；逾時視為不可用
REPORT_FONTS = '"Times New Roman", Times, "DFKai-SB", "BiauKai", "標楷體", "Kaiti TC", "TW-Kai", serif'
QMD_FONTS = '"Times New Roman", Times, "DFKai-SB", "BiauKai", "標楷體", "Kaiti TC", serif'
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"
PDF_HINT = "可用瀏覽器開啟 report.html → 列印 → 另存為 PDF"
# 英數字型在前、中文楷體在後：中文字元由瀏覽器改用清單中第一個有的楷體，都沒有時退回系統襯線字型
HTML_STYLE = """<style>
body { font-family: FONTS;
       max-width: 900px; margin: 2em auto; padding: 0 16px; line-height: 1.7; color: #222; background: #fff; }
code, pre { font-family: Consolas, "Courier New", monospace; }
pre { background: #f6f8fa; padding: 12px; overflow-x: auto; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; }
img { max-width: 100%; }
details > summary { cursor: pointer; font-weight: bold; }
@media print { body { max-width: none; margin: 0; } pre { white-space: pre-wrap; } }
</style>
<script>window.addEventListener("beforeprint",()=>document.querySelectorAll("details").forEach(d=>d.open=true));</script>
""".replace("FONTS", REPORT_FONTS)


def call(cmd, timeout, **kw):
    """執行外部指令；找不到指令或逾時時回傳 (None, 原因)。"""
    try:
        return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, **kw), ""
    except FileNotFoundError:
        return None, f"找不到 {Path(str(cmd[0])).name}"
    except subprocess.TimeoutExpired:
        return None, f"{Path(str(cmd[0])).name} 超過 {timeout} 秒沒有回應，視為不可用"


def find_pandoc():
    """系統的 pandoc；沒有時改用 Quarto 內附的 pandoc。"""
    found = shutil.which("pandoc")
    if found:
        return found
    quarto = shutil.which("quarto")
    if quarto:
        tools = Path(quarto).resolve().parent / "tools"
        for c in (tools / "pandoc.exe", tools / "pandoc", tools / "x86_64" / "pandoc", tools / "aarch64" / "pandoc"):
            if c.exists():
                return str(c)
    return None


def finalize(cfg, report_path, config_path):
    out = Path(cfg.get("out_dir", "output"))
    pdf_mode = str(cfg.get("pdf", "off")).lower()
    if pdf_mode == "auto":                        # 舊版設定值，視同 "on"
        pdf_mode = "on"
    if pdf_mode not in ("on", "off"):
        fail(f'config 的 pdf 只能是 "off"（預設）或 "on"，目前為 {cfg.get("pdf")!r}。')
    rp = Path(report_path)
    if not rp.exists():
        fail(f"找不到報告檔：{rp.resolve()}（請把 report_draft.md 的解讀填完後另存為 report.md）")
    # 程式碼一章一律由本程式重新附上；先切掉，避免附上的程式碼本身被當成未填寫處（重跑 finalize 時）
    text = rp.read_text(encoding="utf-8").split("\n## 8. 完整 Python 程式碼")[0].rstrip() + "\n"
    left = re.findall(r"【待填：[^】]*】", text)
    if left:
        fail(f"報告中還有 {len(left)} 處未填寫的解讀，例如：{left[0][:60]}")
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    final = results["final_model"]
    if results["data"]["task"] == "classification":
        acc = f"{results['test'][final]['accuracy']:.3f}"
        if acc not in text:
            print(f"[注意] 報告中找不到最終模型的測試準確率 {acc}，請確認數字取自 results.json。")
    src = Path(globals().get("__file__", "run_analysis.py"))   # Jupyter 沒有 __file__，改找目前資料夾的 run_analysis.py
    if not src.exists():
        fail(f"找不到程式原始檔 {src.resolve()}；報告必須附上完整程式碼。")
    code = src.read_text(encoding="utf-8")
    cfg_text = Path(config_path).read_text(encoding="utf-8")

    def code_chapter(fence_open, fence_close, fold=False):
        """fold=True 時（HTML）程式碼與設定檔放進預設收合的 <details>。"""
        def block(summary, inner):
            return f'<details markdown="1"><summary>{summary}</summary>\n\n{inner}\n\n</details>\n' if fold else inner + "\n"
        return ("\n## 8. 完整 Python 程式碼\n\n"
                "重現方式：安裝 requirements.txt 的套件後，依序執行\n\n"
                f"```bash\npython run_analysis.py run --config {Path(config_path).name}\n"
                f"python run_analysis.py finalize --config {Path(config_path).name} --report report.md\n```\n\n"
                "### config.json\n\n" + block("顯示設定檔", f"```json\n{cfg_text}\n```") + "\n"
                "### run_analysis.py\n\n" + block("顯示完整程式碼", f"{fence_open}{code}\n{fence_close}"))

    def embed(s):
        return re.sub(r"!\[([^\]]*)\]\((figures/[^)]+\.png)\)(\{[^}]*\})?",
                      lambda m: f"![{m.group(1)}](data:image/png;base64,"
                                f"{base64.b64encode((out / m.group(2)).read_bytes()).decode()})", s)

    title = cfg.get("report", {}).get("title", "分析報告")
    plain = ("```python\n", "```")
    done, skipped = [], []

    def emit(body):
        """寫出 report.md、report.qmd 與 report.html；report.html 失敗時 finalize 失敗。"""
        (out / "report.md").write_text(embed(body + code_chapter(*plain)), encoding="utf-8")
        write_qmd(body, code_chapter, out, cfg)
        return build_html(embed(body + code_chapter(*plain, fold=True)), out / "report.html", title)

    engine = emit(text)
    done += [out / "report.html", out / "report.md", out / "report.qmd"]
    pdf = out / "report.pdf"
    if pdf_mode == "off":                         # 預設：完全不檢查 PDF 環境
        skipped.append(("report.pdf", 'config 的 pdf 為 "off"（預設），未產生；需要 PDF 時改為 "on" 後重跑 finalize'))
    else:
        res = build_pdf(out, text + code_chapter(*plain))
        if res["ok"] and res["fonts"]:            # 能讀出 PDF 實際字型時寫進附錄，再產生一次讓各檔一致
            text = text.replace("- 浮點運算", f"- PDF 實際使用的字型（pdffonts）：{'、'.join(res['fonts'])}\n- 浮點運算", 1)
            emit(text)
            res = build_pdf(out, text + code_chapter(*plain), route=res["route"], combo=res.get("combo"))
        if res["ok"]:
            done.append(pdf)
        else:
            print(f"[提示] 未產生 report.pdf：{res['reason']}。{PDF_HINT}。")
            skipped.append(("report.pdf", f"{res['reason']}；{PDF_HINT}"))
    print("\n── 完成摘要 ──")
    print(f"report.html 轉換方式：{engine}")
    for p in done:
        print(f"已產生：{p.resolve()}")
    for name, why in skipped:
        print(f"已略過：{name}（{why}）")


def build_html(md_text, html_path, title):
    """產生自足的 report.html（圖片已內嵌）。依序：pandoc → Python markdown 套件；
    環境變數 RUN_ANALYSIS_HTML_ENGINE=pandoc|markdown 可強制指定。回傳實際使用的方式。"""
    force = os.environ.get("RUN_ANALYSIS_HTML_ENGINE", "").strip().lower()
    body = re.sub(r"\A# [^\n]*\n+", "", md_text)                  # 標題改由 metadata／<h1> 產生，避免重複
    html_path.unlink(missing_ok=True)
    pandoc = find_pandoc() if force != "markdown" else None
    if force == "pandoc" and not pandoc:
        fail("RUN_ANALYSIS_HTML_ENGINE=pandoc，但找不到 pandoc。")
    if pandoc:
        r, why = call([pandoc, "--version"], DETECT_TIMEOUT)
        m = re.search(r"(\d+)\.(\d+)", r.stdout) if r else None
        ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        src, style = html_path.with_name("_report_for_html.md"), html_path.with_name("_report_style.html")
        src.write_text(body, encoding="utf-8")
        style.write_text(HTML_STYLE, encoding="utf-8")
        r, why = call([pandoc, src, "-f", "markdown", "-t", "html5", "-s", "--toc", "--mathml",
                       "--metadata", f"title={title}", "--metadata", "lang=zh-TW", "-H", style,
                       "--embed-resources" if ver >= (2, 19) else "--self-contained", "-o", html_path], CONVERT_TIMEOUT)
        src.unlink(missing_ok=True)
        style.unlink(missing_ok=True)
        if r and r.returncode == 0 and html_path.exists():
            print(f"已完成：{html_path.resolve()}")
            return f"pandoc {ver[0]}.{ver[1]}（數學式為 MathML，可離線顯示）"
        reason = why or r.stderr.strip()[-300:]
        if force == "pandoc":
            fail(f"pandoc 產生 report.html 失敗：{reason}")
        print(f"[注意] pandoc 產生 report.html 失敗，改用 markdown 套件：{reason}")
    try:
        import markdown
    except ImportError:
        fail("無法產生 report.html：找不到 pandoc，也沒有 Python markdown 套件。請安裝 markdown 套件：pip install markdown"
             "（report.md、report.qmd 已保留）")
    # 數學式先換成佔位符，避免被 Markdown 處理；程式碼區塊與行內程式碼不動
    maths = []

    def hold(m):
        maths.append(m.group(0))
        return f"MATHPH{len(maths) - 1}END"

    def protect(seg):
        parts = re.split(r"(`[^`\n]+`)", seg)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r"\$\$.+?\$\$", hold, parts[i], flags=re.S)
            parts[i] = re.sub(r"(?<![\\$])\$(?=\S)[^$\n]+?(?<=\S)\$(?!\d)", hold, parts[i])
        return "".join(parts)

    pieces, pos = [], 0
    for m in re.finditer(r"(?ms)^(`{3,}).*?^\1[ \t]*$", body):
        pieces += [protect(body[pos:m.start()]), m.group(0)]
        pos = m.end()
    pieces.append(protect(body[pos:]))
    inner = markdown.markdown("[TOC]\n\n" + "".join(pieces),
                              extensions=["tables", "fenced_code", "toc", "md_in_html", "sane_lists"])
    inner = re.sub(r"MATHPH(\d+)END", lambda m: html_lib.escape(maths[int(m.group(1))]), inner)
    page = ("<!DOCTYPE html>\n<html lang=\"zh-TW\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{html_lib.escape(title)}</title>\n{HTML_STYLE}"
            "<script>window.MathJax = {tex: {inlineMath: [['$','$']], displayMath: [['$$','$$']]}};</script>\n"
            f"<script async src=\"{MATHJAX_URL}\"></script>\n</head>\n<body>\n"
            f"<h1>{html_lib.escape(title)}</h1>\n{inner}\n</body>\n</html>\n")
    html_path.write_text(page, encoding="utf-8")
    print(f"已完成：{html_path.resolve()}")
    return "Python markdown 套件（數學式以 MathJax 顯示，需要網路連線）"


def write_qmd(body, code_chapter, out, cfg):
    """report.qmd：內容同 report.md，完整程式碼放在不執行的 {python} 區塊（可折疊）；不在此 render。"""
    title = cfg.get("report", {}).get("title", "分析報告")
    body = re.sub(r"\A# [^\n]*\n+", "", body)                     # 標題改放在 YAML
    yaml = ("---\n"
            f"title: {json.dumps(title, ensure_ascii=False)}\n"
            "lang: zh-TW\n"
            "format:\n"
            "  html:\n"
            "    toc: true\n"
            "    embed-resources: true\n"
            "    code-fold: true\n"
            '    code-summary: "顯示程式碼"\n'
            f"    mainfont: '{QMD_FONTS}'\n"
            "execute:\n"
            "  eval: false\n"
            "---\n\n")
    qmd = out / "report.qmd"
    qmd.write_text(yaml + body + code_chapter("````{python}\n#| eval: false\n", "````"), encoding="utf-8")
    print(f"已完成：{qmd.resolve()}")


def available_fonts():
    r, _ = call(["fc-list", ":", "family"], DETECT_TIMEOUT)
    if r and r.returncode == 0:
        return {n.strip().replace("\\-", "-") for line in r.stdout.splitlines() for n in line.split(",")}  # fc-list 以 \- 跳脫連字號
    return {f.name for f in font_manager.fontManager.ttflist}


def pdf_fonts(pdf):
    """以 pdffonts 讀出 PDF 實際嵌入的字型；讀不到時回傳 None。"""
    if not shutil.which("pdffonts"):
        return None
    r, _ = call(["pdffonts", pdf], DETECT_TIMEOUT)
    if not r or r.returncode != 0:
        return None
    names = [re.sub(r"^[A-Z]{6}\+", "", ln.split()[0]) for ln in r.stdout.splitlines()[2:] if ln.strip()]
    return sorted(set(names)) or None


def build_pdf(out, md_text, route=None, combo=None):
    """選用的 PDF。依序：playwright（以 Chromium 列印 report.html）→ xelatex＋xeCJK（pandoc）。
    回傳 {"ok", "reason", "route", "combo", "fonts"}；失敗時刪除不完整的 report.pdf，不中止程式。"""
    pdf, html = out / "report.pdf", out / "report.html"
    reasons = []
    if route in (None, "playwright"):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            sync_playwright = None
            reasons.append("未安裝 playwright")
        if sync_playwright:
            try:
                with sync_playwright() as pw:
                    b = pw.chromium.launch()
                    try:
                        pg = b.new_page()
                        pg.goto(html.resolve().as_uri())
                        pg.wait_for_load_state("networkidle", timeout=60000)
                        pg.evaluate("() => (window.MathJax && MathJax.startup) ? MathJax.startup.promise.then(() => true) : true")
                        pg.evaluate("() => document.querySelectorAll('details').forEach(d => d.open = true)")
                        pg.pdf(path=str(pdf), format="A4", print_background=True,
                               margin={"top": "2cm", "bottom": "2cm", "left": "2cm", "right": "2cm"})
                    finally:
                        b.close()
                if pdf.exists():
                    print(f"已完成：{pdf.resolve()}")
                    return {"ok": True, "reason": "", "route": "playwright", "fonts": pdf_fonts(pdf)}
            except Exception as e:
                reasons.append(f"playwright 無法啟動 Chromium 或列印失敗（{type(e).__name__}：{str(e).strip()[:120]}）")
            pdf.unlink(missing_ok=True)
    if route in (None, "xelatex"):
        pandoc = find_pandoc()
        has_xecjk = False
        if not shutil.which("xelatex"):
            reasons.append("找不到 xelatex")
        elif not pandoc:
            reasons.append("有 xelatex 但找不到 pandoc")
        elif not shutil.which("kpsewhich"):
            reasons.append("找不到 kpsewhich，無法確認 xeCJK")
        else:
            r, why = call(["kpsewhich", "xeCJK.sty"], DETECT_TIMEOUT)
            has_xecjk = bool(r and r.returncode == 0 and r.stdout.strip())
            if not has_xecjk:
                reasons.append(why or "TeX 缺少 xeCJK 套件")
        if has_xecjk:
            have = available_fonts()
            latin_ok = [f for f in LATIN_PDF_FONTS if f in have] or ["serif"]
            cjk_ok = [f for f in CJK_PDF_FONTS if f in have] or ["serif"]
            md, header = out / "_report_for_pdf.md", out / "_header.tex"
            md.write_text(md_text, encoding="utf-8")
            last = ""
            for latin, cjk in ([combo] if combo else [(l, c) for c in cjk_ok for l in latin_ok]):
                header.write_text(
                    "\\usepackage{xeCJK}\n"
                    f"\\setmainfont{{{latin}}}\n\\setCJKmainfont{{{cjk}}}\n"
                    "\\usepackage{fvextra}\n"
                    "\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\\\\{\\}}\n"
                    "\\renewenvironment{Shaded}{}{}\n",          # framed 底色框遇到很長的程式碼會 Dimension too large
                    encoding="utf-8")
                r, why = call([pandoc, md, "-o", pdf, "--pdf-engine=xelatex", f"--resource-path={out}",
                               "-V", "fontfamily=fontspec", "-V", "geometry:margin=2.2cm", "-V", "fontsize=11pt",
                               "-H", header, "--highlight-style=tango"], XELATEX_TIMEOUT)
                if r and r.returncode == 0 and pdf.exists():
                    md.unlink(missing_ok=True)
                    header.unlink(missing_ok=True)
                    print(f"已完成：{pdf.resolve()}")
                    return {"ok": True, "reason": "", "route": "xelatex", "combo": (latin, cjk), "fonts": pdf_fonts(pdf)}
                err = "" if why else r.stderr
                key = re.findall(r"^! .+$", err, re.M) or [ln for ln in err.splitlines() if ln.strip()][-1:]
                last = why or re.sub(r"\s+", " ", key[0] if key else "未知錯誤").strip()[:160]
                pdf.unlink(missing_ok=True)
                if why or not re.search(r"fontspec|font .*not (be )?found", err, re.I):   # 只有字型問題才換字型重試
                    break
            md.unlink(missing_ok=True)
            header.unlink(missing_ok=True)
            reasons.append(f"xelatex 產生 PDF 失敗（{last}）")
    pdf.unlink(missing_ok=True)
    return {"ok": False, "reason": "；".join(reasons) or "沒有可用的 PDF 工具", "route": None, "fonts": None}


# ═════════════════════════════════ 入口 ═════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default=MODE, choices=["inspect", "profile", "run", "finalize"])
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--y", default=Y_COLUMN)
    ap.add_argument("--x", nargs="*", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--out", default="output")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--report", default=REPORT_PATH)
    # Jupyter 會自動帶入 -f kernel.json 等參數：在 Jupyter 中一律使用檔案開頭的設定
    args, _ = ap.parse_known_args([] if IN_JUPYTER else None)
    if args.mode == "inspect":
        if not args.y:
            fail("inspect 需要 --y 指定反應變數欄位。")
        inspect(args.data, args.y, args.x, args.sheet, args.encoding, args.out)
        return
    cp = Path(args.config)
    if not cp.exists():
        fail(f"找不到 config 檔：{cp.resolve()}")
    cfg = json.loads(cp.read_text(encoding="utf-8"))
    np.random.seed(cfg.get("seed", 2026))
    if args.mode == "profile":
        setup_fonts()
        run_profile(cfg)
    elif args.mode == "run":
        run(cfg, args.config)
    else:
        finalize(cfg, args.report, args.config)


if __name__ == "__main__":
    main()
