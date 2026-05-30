"""
Ames Housing — House Price Prediction with a Tree + Linear BLEND
================================================================

Predicts `SalePrice` for the Ames Housing dataset and reports the competition
metric, Root Mean Squared Logarithmic Error (RMSLE).
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# Configuration

TRAIN_PATH = "train.csv"
TARGET = "SalePrice"
RANDOM_STATE = 42
N_SPLITS = 5
BLEND_WEIGHT_TREE = 0.5

# Categorical columns where NaN means "this feature does not exist"
# Replaced "None" so  ordinal maps below can assign them the lowest rank (0).
NONE_FILL_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Numeric in storage, categorical in meaning -> treat as categories.
NUMERIC_AS_CATEGORICAL = ["MSSubClass", "YrSold", "MoSold"]

# Ordinal encodings
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

ORDINAL_MAPPINGS = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional":   {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Fence":        {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LotShape":     {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope":    {"Sev": 0, "Mod": 1, "Gtl": 2},
    "PavedDrive":   {"N": 0, "P": 1, "Y": 2},
    "Utilities":    {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "Street":       {"Grvl": 0, "Pave": 1},
    "CentralAir":   {"N": 0, "Y": 1},
}
for _c in QUALITY_COLS:
    ORDINAL_MAPPINGS[_c] = QUALITY_MAP


# Data loading
def load_data(path: str):
    # Separate features (X) from target (y)
    df = pd.read_csv(path)
    df = df.drop(columns=["Id"]) # Id carries no signal
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()
    return X, y


# Encoding + feature engineering
def encode(X: pd.DataFrame) -> pd.DataFrame:
    # Cast numeric-categoricals, engineer features, fill 'None', ordinal-map
    X = X.copy()

    # Numeric-but-categorical -> string for one-hot encoding
    for col in NUMERIC_AS_CATEGORICAL:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Feature engineering
    X["TotalSF"] = (X["TotalBsmtSF"].fillna(0)
                    + X["1stFlrSF"].fillna(0) + X["2ndFlrSF"].fillna(0))
    X["TotalBath"] = (X["FullBath"].fillna(0) + 0.5 * X["HalfBath"].fillna(0)
                      + X["BsmtFullBath"].fillna(0) + 0.5 * X["BsmtHalfBath"].fillna(0))
    X["TotalPorchSF"] = (X["OpenPorchSF"].fillna(0) + X["EnclosedPorch"].fillna(0)
                         + X["3SsnPorch"].fillna(0) + X["ScreenPorch"].fillna(0)
                         + X["WoodDeckSF"].fillna(0))
    yr_sold = X["YrSold"].astype(int)
    X["HouseAge"] = yr_sold - X["YearBuilt"]
    X["SinceRemod"] = yr_sold - X["YearRemodAdd"]
    X["IsRemodeled"] = (X["YearBuilt"] != X["YearRemodAdd"]).astype(int)
    X["HasPool"] = (X["PoolArea"] > 0).astype(int)
    X["Has2ndFloor"] = (X["2ndFlrSF"] > 0).astype(int)
    X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
    X["HasBsmt"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)

    # Pseudo-missing categoricals: NaN means "feature absent" -> "None".
    for col in NONE_FILL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna("None")

    # Ordinal encoding. Values not in a map (a true NaN) stay NaN
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in X.columns:
            X[col] = X[col].map(mapping)

    return X


# Fold-specific preprocessors (fit on TRAIN only -> no leakage)

class TreePreprocessor:
    # Median/mode impute + one-hot

    def fit_transform(self, X):
        X = encode(X)
        is_num = X.dtypes.apply(pd.api.types.is_numeric_dtype)
        self.num_cols_ = X.columns[is_num]
        self.cat_cols_ = X.columns[~is_num]
        self.medians_ = {c: X[c].median() for c in self.num_cols_}
        self.modes_ = {c: X[c].mode()[0] for c in self.cat_cols_}
        X = self._impute(X)
        X = pd.get_dummies(X, columns=list(self.cat_cols_), drop_first=False)
        self.columns_ = X.columns
        return X

    def transform(self, X):
        X = self._impute(encode(X))
        X = pd.get_dummies(X, columns=list(self.cat_cols_), drop_first=False)
        return X.reindex(columns=self.columns_, fill_value=0)

    def _impute(self, X):
        for c in self.num_cols_:
            X[c] = X[c].fillna(self.medians_[c])
        for c in self.cat_cols_:
            X[c] = X[c].fillna(self.modes_[c])
        return X


class LinearPreprocessor:
    # One-hot + log1p of skewed numerics + standardisation, for Lasso.

    SKEW_THRESHOLD = 0.75

    def fit_transform(self, X):
        X = encode(X)
        is_num = X.dtypes.apply(pd.api.types.is_numeric_dtype)
        self.num_cols_ = X.columns[is_num]
        self.cat_cols_ = X.columns[~is_num]
        self.medians_ = {c: X[c].median() for c in self.num_cols_}
        self.modes_ = {c: X[c].mode()[0] for c in self.cat_cols_}
        X = self._impute(X)
        X = pd.get_dummies(X, columns=list(self.cat_cols_), drop_first=False)
        self.columns_ = X.columns

        # Identify continuous skewed columns
        cont = [c for c in X.columns if X[c].nunique() > 10]
        sk = X[cont].apply(lambda c: skew(c.dropna()))
        self.skewed_ = list(sk[sk.abs() > self.SKEW_THRESHOLD].index)
        self.shift_ = {c: min(0, X[c].min()) for c in self.skewed_}  # keep >= 0
        X = self._log_skewed(X)

        self.scaler_ = StandardScaler().fit(X)
        return pd.DataFrame(self.scaler_.transform(X), columns=X.columns, index=X.index)

    def transform(self, X):
        X = self._impute(encode(X))
        X = pd.get_dummies(X, columns=list(self.cat_cols_), drop_first=False)
        X = X.reindex(columns=self.columns_, fill_value=0)
        X = self._log_skewed(X)
        return pd.DataFrame(self.scaler_.transform(X), columns=X.columns, index=X.index)

    def _impute(self, X):
        for c in self.num_cols_:
            X[c] = X[c].fillna(self.medians_[c])
        for c in self.cat_cols_:
            X[c] = X[c].fillna(self.modes_[c])
        return X

    def _log_skewed(self, X):
        for c in self.skewed_:
            if c in X.columns:
                X[c] = np.log1p(X[c] - self.shift_[c])
        return X


# Models
def build_tree_model() -> XGBRegressor:
    # Gradient boosting regressor (unchanged baseline hyperparameters)
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_linear_model() -> LassoCV:
    # L1-regularised linear model; alpha chosen by internal CV on the fold.

    return LassoCV(alphas=np.logspace(-4, -1, 50), max_iter=5000,
                   random_state=RANDOM_STATE, n_jobs=-1)


# Metrics
def rmsle(y_true, y_pred) -> float:
    # RMSLE on the original price scale (== RMSE in log space here)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def rmsle_log(y_true_log, y_pred_log) -> float:
    # RMSLE when inputs are already on the log1p scale
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))


# Cross-validated evaluation of the blend (honest, leak-free)
def cross_validate_blend(X_raw: pd.DataFrame, y: pd.Series, n_splits=N_SPLITS):
    # K-fold CV reporting tree-only, linear-only, and blended RMSLE.

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_log = np.log1p(y.values)
    scores = {"tree": [], "linear": [], "blend": []}

    for tr_idx, va_idx in kf.split(X_raw):
        Xtr_raw, Xva_raw = X_raw.iloc[tr_idx], X_raw.iloc[va_idx]
        ytr, yva = y_log[tr_idx], y_log[va_idx]

        tp = TreePreprocessor()
        Xtr_t, Xva_t = tp.fit_transform(Xtr_raw), tp.transform(Xva_raw)
        tree = build_tree_model().fit(Xtr_t, ytr)
        tree_pred = tree.predict(Xva_t)

        lp = LinearPreprocessor()
        Xtr_l, Xva_l = lp.fit_transform(Xtr_raw), lp.transform(Xva_raw)
        lin = build_linear_model().fit(Xtr_l, ytr)
        lin_pred = lin.predict(Xva_l)

        w = BLEND_WEIGHT_TREE
        blend_pred = w * tree_pred + (1 - w) * lin_pred

        scores["tree"].append(rmsle_log(yva, tree_pred))
        scores["linear"].append(rmsle_log(yva, lin_pred))
        scores["blend"].append(rmsle_log(yva, blend_pred))

    return {k: np.array(v) for k, v in scores.items()}


# 7. Diagnostics
def report_multicollinearity(X_raw: pd.DataFrame):
    # Harmless for tree/linear-blend; reported, not removed.
    garage_num = [c for c in ["GarageCars", "GarageArea", "GarageYrBlt"] if c in X_raw.columns]
    corr = X_raw[garage_num].corr()
    print("\nCorrelation among numeric garage predictors (diagnostic only):")
    print(corr.round(2).to_string())


def plot_top_importances(model, feature_names, n=10, outfile="feature_importances.png"):
    # Horizontal bar chart of the n most important tree features.
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(n)

    plt.figure(figsize=(9, 5))
    plt.barh(top.index[::-1], top.values[::-1], color="#2b8cbe")
    plt.xlabel("Importance (gain)")
    plt.title(f"Top {n} Feature Importances — XGBoost branch")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"\nFeature-importance plot saved to: {outfile}")
    print(f"\nTop {n} features (tree branch):")
    for name, val in top.items():
        print(f"  {name:<25} {val:.4f}")


# Main
def main():

    X, y = load_data(TRAIN_PATH)
    print(f"Loaded data: {X.shape[0]} rows, {X.shape[1]} raw features.")

    # Correlated garage predictors (kept, not dropped)
    report_multicollinearity(X)

    # Cross-validated, leak-free evaluation of tree / linear / blend
    cv = cross_validate_blend(X, y)
    print(f"\n{'=' * 56}")
    print(f"  {N_SPLITS}-fold CV RMSLE (leak-free preprocessing)")
    print(f"{'-' * 56}")
    for name in ("tree", "linear", "blend"):
        print(f"    {name:<8} {cv[name].mean():.5f}  +/- {cv[name].std():.5f}")
    gain = cv["tree"].mean() - cv["blend"].mean()
    print(f"{'-' * 56}")
    print(f"    blend improves on tree-only by {gain:+.5f} RMSLE")
    print(f"{'=' * 56}")

    # Fit the tree branch on all data once, for the importance plot
    tp = TreePreprocessor()
    X_tree = tp.fit_transform(X)
    full_tree = build_tree_model().fit(X_tree, np.log1p(y.values))
    plot_top_importances(full_tree, X_tree.columns, n=10)


if __name__ == "__main__":
    main()
