#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analysis.py — Quant Data Exercise
Python ≥ 3.8
"""

# ----------------- 全局依赖与配置 -----------------
import warnings, math
from typing import Tuple, Dict, List

import matplotlib
matplotlib.use("Agg")               # 非交互后端：不会弹窗阻塞
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

try:
    from IPython.display import display   # 若无 IPython 会 fallback
except ImportError:
    def display(x, **k):
        print(x)

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (8, 6)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ----------------- 工具函数 -----------------
def rmse(y_true, y_pred) -> float:
    """统一计算 RMSE，兼容旧版 sklearn (无 squared 参数)"""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:  # <=0.22
        return math.sqrt(mean_squared_error(y_true, y_pred))


def winsorize_df(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """对数值列做 winsorize（复制一份）"""
    df_ = df.copy()
    num_cols = df_.select_dtypes(include="number").columns
    for col in num_cols:
        lo, hi = df_[col].quantile([lower, upper])
        df_[col] = df_[col].clip(lo, hi)
    return df_


def load_data(
    x_path: str = "X.csv",
    y_path: str = "Y.csv",
    z_path: str = "Z.csv",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """读取并清洗数据；Z 列名统一加前缀 Z_"""
    X = pd.read_csv(x_path)
    Y_df = pd.read_csv(y_path)
    Z = pd.read_csv(z_path)

    for df in (X, Y_df, Z):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", inplace=True)

    if Y_df.shape[1] != 1:
        raise ValueError("Y.csv 应仅 1 列")
    Y = Y_df.iloc[:, 0]

    # 给 Z 列加前缀防重名
    Z.columns = [f"Z_{c}" for c in Z.columns]

    print("\n>>> 缺失值统计 (NaN 数量)")
    print("X:", X.isna().sum().sum(), "| Z:", Z.isna().sum().sum(), "| Y:", int(Y.isna().sum()))

    # Winsorize 后返回
    return winsorize_df(X), Y, winsorize_df(Z)


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")),
         ("sc", RobustScaler())]
    )
    cat_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="most_frequent")),
         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)]
    )


def train_eval(
    X_df: pd.DataFrame,
    y,
    model_choice: str,
    name: str,
) -> Tuple[Pipeline, Dict[str, float]]:
    """训练 + 评估；返回 pipeline 与测试集指标"""
    num_cols = X_df.select_dtypes(include="number").columns.tolist()
    cat_cols = X_df.select_dtypes(exclude="number").columns.tolist()
    preproc = build_preprocessor(num_cols, cat_cols)

    if model_choice == "ols":
        reg = LinearRegression()
    elif model_choice == "ridge":
        reg = RidgeCV(alphas=np.logspace(-2, 2, 15), cv=4)
    elif model_choice == "lasso":
        reg = LassoCV(alphas=np.logspace(-2, 1, 15), cv=4, max_iter=50000)
    else:
        raise ValueError("model_choice 必须是 ols / ridge / lasso")

    pipe = Pipeline([("prep", preproc), ("reg", reg)])

    Xtr, Xte, ytr, yte = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = RepeatedKFold(n_splits=4, n_repeats=2, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipe, Xtr, ytr,
        cv=cv, scoring="neg_mean_squared_error",
        error_score=np.nan
    )
    failed = np.isnan(cv_scores).sum()
    ok = cv_scores[~np.isnan(cv_scores)]
    cv_rmse = (-ok.mean())**0.5 if ok.size else np.nan
    print(f"\n{name} | CV 折失败 {failed}/{len(cv_scores)} | CV RMSE: {cv_rmse:.4f}")

    if ok.size == 0:  # 全失败，直接返回 nan 指标
        return pipe, {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    metrics = {
        "rmse": rmse(yte, y_pred),
        "mae": mean_absolute_error(yte, y_pred),
        "r2": r2_score(yte, y_pred),
    }
    print(f"{name} | Test metrics: {metrics}")

    # 保存可视化
    plt.scatter(yte, y_pred, alpha=0.6)
    plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "k--")
    plt.xlabel("True Y"); plt.ylabel("Pred Y"); plt.title(f"{name} True vs Pred")
    plt.tight_layout(); plt.savefig(f"{name}_scatter.png", dpi=120); plt.close()

    sns.histplot(yte - y_pred, kde=True)
    plt.title(f"{name} Residuals"); plt.tight_layout()
    plt.savefig(f"{name}_residual.png", dpi=120); plt.close()

    return pipe, metrics


def summarize(res: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        [[k, v["rmse"], v["mae"], v["r2"]] for k, v in res.items()],
        columns=["Model", "RMSE", "MAE", "R²"]
    ).set_index("Model")
    print("\n===== 汇总比较 ====="); display(df)
    return df


def bootstrap_rmse(model, X_df, y, n=500):
    rng = np.random.RandomState(RANDOM_STATE)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        vals.append(rmse(y.iloc[idx], model.predict(X_df.iloc[idx])))
    vals = np.array(vals)
    return vals.mean(), (np.percentile(vals, 2.5), np.percentile(vals, 97.5))


# ----------------- 主流程 -----------------
def main():
    X, Y, Z = load_data()

    results: Dict[str, Dict[str, float]] = {}
    p1, results["X-OLS"]   = train_eval(X, Y, "ols",   "X-OLS")
    p2, results["X-Ridge"] = train_eval(X, Y, "ridge", "X-Ridge")
    p3, results["X-Lasso"] = train_eval(X, Y, "lasso", "X-Lasso")

    XZ = pd.concat([X, Z], axis=1)
    p4, results["XZ-OLS"]   = train_eval(XZ, Y, "ols",   "XZ-OLS")
    p5, results["XZ-Ridge"] = train_eval(XZ, Y, "ridge", "XZ-Ridge")
    p6, results["XZ-Lasso"] = train_eval(XZ, Y, "lasso", "XZ-Lasso")

    df = summarize(results)
    best = df["RMSE"].idxmin()
    best_pipe = {"X-OLS": p1, "X-Ridge": p2, "X-Lasso": p3,
                 "XZ-OLS": p4, "XZ-Ridge": p5, "XZ-Lasso": p6}[best]
    best_X = XZ if best.startswith("XZ") else X
    mean_, (lo, hi) = bootstrap_rmse(best_pipe, best_X, Y)
    print(f"\n{best} | Bootstrap RMSE ≈ {mean_:.4f} (95% CI [{lo:.4f}, {hi:.4f}])")

    print("\n>>> 完成：图已保存为 *.png，可直接插入报告。")


if __name__ == "__main__":
    main()