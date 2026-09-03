import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    brier_score_loss
)
from sklearn.inspection import (
    permutation_importance,
    PartialDependenceDisplay
)
from sklearn.base import clone

try:
    import shap
except ImportError:
    shap = None


class AttributeModelAnalyzer:
    """
    Attribute-level model diagnostics for:
      - regression
      - binary classification

    Designed for comparing:
      1. actual target relationship
      2. predicted target relationship
      3. model feature usage
      4. model fidelity by attribute
      5. multiple trained models
    """

    def __init__(
        self,
        df,
        feature_cols,
        target_col,
        models,
        pred_cols=None,
        task="regression"
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame

        feature_cols : list[str]

        target_col : str

        models : dict
            Example:
            {
                "RRFE": trained_rrfe_model,
                "Boruta": trained_boruta_model
            }

        pred_cols : dict, optional
            Existing prediction columns:
            {
                "RRFE": "pred_rrfe",
                "Boruta": "pred_boruta"
            }

            If omitted, predictions will be generated from the models.

        task : {"regression", "binary"}
        """

        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.models = models
        self.pred_cols = pred_cols or {}
        self.task = task

        if task not in ["regression", "binary"]:
            raise ValueError(
                "task must be 'regression' or 'binary'"
            )

    # =========================================================
    # Prediction
    # =========================================================

    def get_prediction(self, model_name):

        if model_name in self.pred_cols:
            return self.df[self.pred_cols[model_name]]

        model = self.models[model_name]

        X = self.df[self.feature_cols]

        if self.task == "binary":
            return model.predict_proba(X)[:, 1]

        return model.predict(X)

    # =========================================================
    # Binning
    # =========================================================

    def create_bins(
        self,
        attribute,
        n_bins=10,
        method="quantile",
        special_values=None,
        categorical_threshold=10
    ):

        x = self.df[attribute]
        special_values = special_values or []

        out = pd.Series(
            index=x.index,
            dtype="object"
        )

        out.loc[x.isna()] = "Missing"

        for val in special_values:
            out.loc[x == val] = f"Special:{val}"

        valid = (
            x.notna()
            & ~x.isin(special_values)
        )

        xv = x.loc[valid]

        if (
            not pd.api.types.is_numeric_dtype(xv)
            or xv.nunique() <= categorical_threshold
        ):
            out.loc[valid] = xv.astype(str)
            return out

        if method == "quantile":

            binned = pd.qcut(
                xv,
                q=n_bins,
                duplicates="drop"
            )

        elif method == "equal_width":

            binned = pd.cut(
                xv,
                bins=n_bins,
                duplicates="drop"
            )

        else:
            raise ValueError(
                "method must be 'quantile' or 'equal_width'"
            )

        out.loc[valid] = binned.astype(str)

        return out

    # =========================================================
    # Bin summary
    # =========================================================

    def summarize_attribute(
        self,
        attribute,
        model_name,
        n_bins=10,
        method="quantile",
        special_values=None,
        min_bin_count=1
    ):

        pred = self.get_prediction(model_name)

        tmp = pd.DataFrame({
            "attribute": self.df[attribute],
            "actual": self.df[self.target_col],
            "pred": pred
        })

        tmp["bin"] = self.create_bins(
            attribute,
            n_bins=n_bins,
            method=method,
            special_values=special_values
        )

        tmp["error"] = (
            tmp["pred"] - tmp["actual"]
        )

        tmp["abs_error"] = np.abs(
            tmp["error"]
        )

        tmp["sq_error"] = (
            tmp["error"] ** 2
        )

        out = (
            tmp.groupby("bin", observed=True)
            .agg(
                count=("actual", "size"),
                actual_mean=("actual", "mean"),
                predicted_mean=("pred", "mean"),
                bias=("error", "mean"),
                mae=("abs_error", "mean"),
                mse=("sq_error", "mean")
            )
            .reset_index()
        )

        out["rmse"] = np.sqrt(out["mse"])

        out["volume_pct"] = (
            out["count"] / out["count"].sum()
        )

        overall_mae = tmp["abs_error"].mean()

        out["mae_ratio"] = (
            out["mae"] / overall_mae
        )

        return out[
            out["count"] >= min_bin_count
        ].copy()

    # =========================================================
    # Attribute fidelity metrics
    # =========================================================

    @staticmethod
    def _eta_squared(values, bins):

        d = pd.DataFrame({
            "value": values,
            "bin": bins
        }).dropna()

        mean_all = d["value"].mean()

        g = (
            d.groupby("bin", observed=True)["value"]
            .agg(["mean", "count"])
        )

        ss_between = np.sum(
            g["count"]
            * (g["mean"] - mean_all) ** 2
        )

        ss_total = np.sum(
            (d["value"] - mean_all) ** 2
        )

        if ss_total == 0:
            return np.nan

        return ss_between / ss_total

    def fidelity_metrics(
        self,
        attribute,
        model_name,
        n_bins=10,
        method="quantile",
        special_values=None,
        min_bin_count=1
    ):

        pred = self.get_prediction(model_name)

        bins = self.create_bins(
            attribute,
            n_bins=n_bins,
            method=method,
            special_values=special_values
        )

        s = self.summarize_attribute(
            attribute,
            model_name,
            n_bins=n_bins,
            method=method,
            special_values=special_values,
            min_bin_count=min_bin_count
        )

        w = s["count"] / s["count"].sum()

        curve_mae = np.sum(
            w * np.abs(
                s["actual_mean"]
                - s["predicted_mean"]
            )
        )

        curve_rmse = np.sqrt(
            np.sum(
                w * (
                    s["actual_mean"]
                    - s["predicted_mean"]
                ) ** 2
            )
        )

        actual_eta2 = self._eta_squared(
            self.df[self.target_col],
            bins
        )

        pred_eta2 = self._eta_squared(
            pred,
            bins
        )

        actual_range = (
            s["actual_mean"].max()
            - s["actual_mean"].min()
        )

        pred_range = (
            s["predicted_mean"].max()
            - s["predicted_mean"].min()
        )

        effect_ratio = (
            pred_range / actual_range
            if actual_range != 0
            else np.nan
        )

        shape_corr = spearmanr(
            s["actual_mean"],
            s["predicted_mean"]
        ).statistic

        worst = s.loc[s["mae"].idxmax()]

        return {
            "attribute": attribute,
            "model": model_name,

            "actual_eta2": actual_eta2,
            "predicted_eta2": pred_eta2,
            "eta2_gap": pred_eta2 - actual_eta2,

            "curve_mae": curve_mae,
            "curve_rmse": curve_rmse,

            "actual_effect_range": actual_range,
            "pred_effect_range": pred_range,
            "effect_strength_ratio": effect_ratio,

            "shape_spearman": shape_corr,

            "worst_bin": worst["bin"],
            "worst_bin_mae": worst["mae"],
            "worst_bin_bias": worst["bias"],
            "worst_bin_mae_ratio": worst["mae_ratio"]
        }

    # =========================================================
    # Permutation importance
    # =========================================================

    def permutation_importance_df(
        self,
        model_name,
        X=None,
        y=None,
        scoring=None,
        n_repeats=5,
        random_state=42,
        sample_size=None
    ):

        model = self.models[model_name]

        if X is None:
            X = self.df[self.feature_cols]

        if y is None:
            y = self.df[self.target_col]

        if sample_size is not None and len(X) > sample_size:

            idx = np.random.default_rng(
                random_state
            ).choice(
                len(X),
                size=sample_size,
                replace=False
            )

            X = X.iloc[idx]
            y = y.iloc[idx]

        if scoring is None:

            scoring = (
                "neg_mean_absolute_error"
                if self.task == "regression"
                else "roc_auc"
            )

        pi = permutation_importance(
            model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        out = pd.DataFrame({
            "attribute": X.columns,
            "permutation_mean": pi.importances_mean,
            "permutation_std": pi.importances_std
        })

        return out.sort_values(
            "permutation_mean",
            ascending=False
        )

    # =========================================================
    # SHAP importance
    # =========================================================

    def shap_importance(
        self,
        model_name,
        sample_size=100000,
        random_state=42
    ):

        if shap is None:
            raise ImportError(
                "Install shap first: pip install shap"
            )

        model = self.models[model_name]

        X = self.df[self.feature_cols]

        if (
            sample_size is not None
            and len(X) > sample_size
        ):
            X = X.sample(
                sample_size,
                random_state=random_state
            )

        explainer = shap.TreeExplainer(model)

        values = explainer.shap_values(X)

        # handle binary classifier versions
        if isinstance(values, list):
            values = values[-1]

        importance = np.mean(
            np.abs(values),
            axis=0
        )

        signed_mean = np.mean(
            values,
            axis=0
        )

        out = pd.DataFrame({
            "attribute": X.columns,
            "mean_abs_shap": importance,
            "mean_signed_shap": signed_mean
        })

        return out.sort_values(
            "mean_abs_shap",
            ascending=False
        )

    # =========================================================
    # SHAP dependence
    # =========================================================

    def plot_shap_dependence(
        self,
        model_name,
        attribute,
        interaction="auto",
        sample_size=50000,
        random_state=42
    ):

        if shap is None:
            raise ImportError(
                "Install shap first"
            )

        model = self.models[model_name]

        X = self.df[self.feature_cols]

        if len(X) > sample_size:
            X = X.sample(
                sample_size,
                random_state=random_state
            )

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap.dependence_plot(
            attribute,
            shap_values,
            X,
            interaction_index=interaction
        )

    # =========================================================
    # Partial dependence
    # =========================================================

    def plot_pdp(
        self,
        model_name,
        attribute,
        X=None,
        grid_resolution=20
    ):

        if X is None:
            X = self.df[self.feature_cols]

        model = self.models[model_name]

        PartialDependenceDisplay.from_estimator(
            model,
            X,
            [attribute],
            grid_resolution=grid_resolution
        )

        plt.tight_layout()
        plt.show()

    # =========================================================
    # Drop-column importance
    # =========================================================

    def drop_column_importance(
        self,
        model_name,
        attributes,
        X_train,
        y_train,
        X_eval,
        y_eval
    ):
        """
        Expensive. Recommended only for selected attributes.
        """

        baseline_model = self.models[model_name]

        if self.task == "regression":
            baseline_pred = baseline_model.predict(
                X_eval
            )
            baseline_metric = mean_absolute_error(
                y_eval,
                baseline_pred
            )

        else:
            baseline_pred = (
                baseline_model.predict_proba(
                    X_eval
                )[:, 1]
            )
            baseline_metric = roc_auc_score(
                y_eval,
                baseline_pred
            )

        results = []

        for attribute in attributes:

            features = [
                c for c in X_train.columns
                if c != attribute
            ]

            model_new = clone(
                baseline_model
            )

            model_new.fit(
                X_train[features],
                y_train
            )

            if self.task == "regression":

                pred = model_new.predict(
                    X_eval[features]
                )

                metric = mean_absolute_error(
                    y_eval,
                    pred
                )

                contribution = (
                    metric - baseline_metric
                )

            else:

                pred = (
                    model_new.predict_proba(
                        X_eval[features]
                    )[:, 1]
                )

                metric = roc_auc_score(
                    y_eval,
                    pred
                )

                contribution = (
                    baseline_metric - metric
                )

            results.append({
                "attribute": attribute,
                "baseline_metric": baseline_metric,
                "without_attribute_metric": metric,
                "drop_column_contribution": contribution
            })

        return pd.DataFrame(
            results
        ).sort_values(
            "drop_column_contribution",
            ascending=False
        )

    # =========================================================
    # Unified attribute report
    # =========================================================

    def build_attribute_report(
        self,
        model_name,
        attributes=None,
        n_bins=10,
        method="quantile",
        special_values=None,
        min_bin_count=1000,
        include_shap=True,
        include_permutation=True,
        permutation_sample_size=200000,
        shap_sample_size=100000
    ):

        if attributes is None:
            attributes = self.feature_cols

        # Fidelity
        rows = []

        for attr in attributes:

            try:
                rows.append(
                    self.fidelity_metrics(
                        attr,
                        model_name,
                        n_bins=n_bins,
                        method=method,
                        special_values=special_values,
                        min_bin_count=min_bin_count
                    )
                )

            except Exception as e:
                print(
                    f"Skipping {attr}: {e}"
                )

        report = pd.DataFrame(rows)

        # SHAP
        if include_shap:

            shap_df = self.shap_importance(
                model_name,
                sample_size=shap_sample_size
            )

            report = report.merge(
                shap_df,
                on="attribute",
                how="left"
            )

        # Permutation
        if include_permutation:

            pi_df = self.permutation_importance_df(
                model_name,
                sample_size=permutation_sample_size
            )

            report = report.merge(
                pi_df,
                on="attribute",
                how="left"
            )

        return report

  def performance_by_bins(
    self,
    attribute,
    model_name,
    n_bins=10,
    method="quantile",
    special_values=None,
    min_bin_count=100,
    tolerances=None
):
    """
    Calculate model performance metrics within each attribute bin.

    Parameters
    ----------
    attribute : str
        Attribute used for segmentation.

    model_name : str
        Key in self.models.

    n_bins : int
        Number of bins for continuous variables.

    method : {"quantile", "equal_width"}

    special_values : list, optional
        Values such as [-99999, -99998, -99997] that should
        be treated as separate categories.

    min_bin_count : int
        Minimum number of records required for a bin.

    tolerances : list, optional
        Regression only.
        Example: [5, 10, 20, 30]
        Calculates percentage of predictions within each tolerance.

    Returns
    -------
    pd.DataFrame
        One row per attribute bin.
    """

    pred = np.asarray(
        self.get_prediction(model_name)
    )

    y = np.asarray(
        self.df[self.target_col]
    )

    bins = self.create_bins(
        attribute=attribute,
        n_bins=n_bins,
        method=method,
        special_values=special_values
    )

    tmp = pd.DataFrame({
        "bin": bins,
        "actual": y,
        "predicted": pred
    })

    tmp = tmp.dropna(
        subset=["bin", "actual", "predicted"]
    )

    results = []

    for bin_name, g in tmp.groupby(
        "bin",
        observed=True,
        sort=False
    ):

        n = len(g)

        if n < min_bin_count:
            continue

        actual = g["actual"].to_numpy()
        predicted = g["predicted"].to_numpy()

        residual = predicted - actual
        abs_error = np.abs(residual)

        row = {
            "attribute": attribute,
            "model": model_name,
            "bin": bin_name,
            "count": n,
            "volume_pct": n / len(tmp),

            "actual_mean": np.mean(actual),
            "predicted_mean": np.mean(predicted),

            "bias": np.mean(residual),
            "abs_bias": np.abs(np.mean(residual)),
        }

        # =====================================================
        # REGRESSION
        # =====================================================

        if self.task == "regression":

            row.update({
                "mae":
                    np.mean(abs_error),

                "median_ae":
                    np.median(abs_error),

                "rmse":
                    np.sqrt(
                        np.mean(residual ** 2)
                    ),

                "p90_ae":
                    np.quantile(abs_error, 0.90),

                "p95_ae":
                    np.quantile(abs_error, 0.95),

                "p99_ae":
                    np.quantile(abs_error, 0.99),

                "actual_std":
                    np.std(actual, ddof=1),

                "predicted_std":
                    np.std(predicted, ddof=1),
            })

            # R2 only when target varies
            if np.var(actual) > 0:
                try:
                    row["r2"] = r2_score(
                        actual,
                        predicted
                    )
                except Exception:
                    row["r2"] = np.nan
            else:
                row["r2"] = np.nan

            # Spearman
            if (
                len(np.unique(actual)) > 1
                and
                len(np.unique(predicted)) > 1
            ):
                try:
                    row["spearman"] = (
                        spearmanr(
                            actual,
                            predicted
                        ).statistic
                    )
                except Exception:
                    row["spearman"] = np.nan
            else:
                row["spearman"] = np.nan

            # Accuracy within score tolerance
            if tolerances is not None:

                for tol in tolerances:

                    row[
                        f"pct_within_{tol}"
                    ] = np.mean(
                        abs_error <= tol
                    )

        # =====================================================
        # BINARY CLASSIFICATION
        # =====================================================

        else:

            actual_rate = np.mean(actual)
            predicted_rate = np.mean(predicted)

            row.update({
                "bad_rate":
                    actual_rate,

                "mean_pred_pd":
                    predicted_rate,

                "calibration_gap":
                    predicted_rate - actual_rate,

                "abs_calibration_gap":
                    abs(
                        predicted_rate - actual_rate
                    ),

                "brier":
                    np.mean(
                        (actual - predicted) ** 2
                    ),

                "logloss_component_mean":
                    -np.mean(
                        actual
                        * np.log(
                            np.clip(
                                predicted,
                                1e-15,
                                1 - 1e-15
                            )
                        )
                        +
                        (1 - actual)
                        * np.log(
                            np.clip(
                                1 - predicted,
                                1e-15,
                                1 - 1e-15
                            )
                        )
                    )
            })

            # AUC requires both classes
            if len(np.unique(actual)) == 2:
                try:
                    row["auc"] = roc_auc_score(
                        actual,
                        predicted
                    )
                except Exception:
                    row["auc"] = np.nan
            else:
                row["auc"] = np.nan

        results.append(row)

    out = pd.DataFrame(results)

    if len(out) == 0:
        return out

    # Overall metrics for comparison with each segment
    if self.task == "regression":

        overall_mae = np.mean(
            np.abs(pred - y)
        )

        overall_rmse = np.sqrt(
            np.mean(
                (pred - y) ** 2
            )
        )

        out["overall_mae"] = overall_mae
        out["overall_rmse"] = overall_rmse

        out["mae_vs_overall"] = (
            out["mae"] / overall_mae
        )

        out["rmse_vs_overall"] = (
            out["rmse"] / overall_rmse
        )

    else:

        overall_brier = np.mean(
            (y - pred) ** 2
        )

        out["overall_brier"] = overall_brier

        out["brier_vs_overall"] = (
            out["brier"] / overall_brier
        )

    return out

def worst_attribute_bins(
    self,
    attribute,
    model_name,
    n_bins=10,
    method="quantile",
    special_values=None,
    min_bin_count=1000,
    top_n=5
):

    perf = self.performance_by_bins(
        attribute=attribute,
        model_name=model_name,
        n_bins=n_bins,
        method=method,
        special_values=special_values,
        min_bin_count=min_bin_count
    )

    if self.task == "regression":

        return (
            perf
            .sort_values(
                "mae_vs_overall",
                ascending=False
            )
            .head(top_n)
        )

    else:

        return (
            perf
            .sort_values(
                "abs_calibration_gap",
                ascending=False
            )
            .head(top_n)
        )


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean(
        (y_true - mean_true) *
        (y_pred - mean_pred)
    )

    ccc = (
        2 * covariance /
        (
            var_true
            + var_pred
            + (mean_true - mean_pred) ** 2
        )
    )

    return ccc


actual_low = y_true <= 545
pred_low = y_pred <= 545

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

low_rv_recall = recall_score(
    actual_low,
    pred_low
)

low_rv_precision = precision_score(
    actual_low,
    pred_low
)

low_rv_f1 = f1_score(
    actual_low,
    pred_low
)

abs_error = np.abs(residual)

return {
    "N": len(y_true),

    # Overall fit
    "R2": r2,
    "MAE": mean_absolute_error(y_true, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    "Spearman": spearman,
    "Pearson": pearsonr(y_true, y_pred)[0],

    # Agreement
    "CCC": concordance_correlation_coefficient(
        y_true, y_pred
    ),

    # Bias
    "Mean Residual": float(np.mean(residual)),
    "Median Residual": float(np.median(residual)),
    "Residual Std": float(np.std(residual, ddof=1)),

    # Error distribution
    "Median AE": float(np.median(abs_error)),
    "P90 AE": float(np.percentile(abs_error, 90)),
    "P95 AE": float(np.percentile(abs_error, 95)),
    "P99 AE": float(np.percentile(abs_error, 99)),

    # Practical accuracy
    "Within 10": float(np.mean(abs_error <= 10)),
    "Within 20": float(np.mean(abs_error <= 20)),
    "Within 30": float(np.mean(abs_error <= 30)),

    # Distribution replication
    "Actual Std": float(np.std(y_true, ddof=1)),
    "Prediction Std": float(np.std(y_pred, ddof=1)),
    "Std Ratio": float(
        np.std(y_pred, ddof=1) /
        np.std(y_true, ddof=1)
    ),

    # Location
    "Mean Prediction": float(np.mean(y_pred)),
    "Mean Actual": float(np.mean(y_true)),
}


import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# 1. BINNING HELPER
# ============================================================

def make_bins(
    s,
    n_bins=10,
    method="quantile",
    custom_bins=None,
    special_values=None,
    categorical_threshold=10
):
    """
    Create bins for one attribute.

    Parameters
    ----------
    s : pd.Series
    n_bins : int
    method : {"quantile", "equal_width", "custom"}
    custom_bins : list, optional
    special_values : list, optional
        Example: [-99999, -99998, -99997]
    categorical_threshold : int
        Numeric variables with <= this many unique values
        are treated as categorical.

    Returns
    -------
    pd.Series of bin labels
    """

    special_values = special_values or []

    out = pd.Series(index=s.index, dtype="object")

    # Missing
    out.loc[s.isna()] = "Missing"

    # Special values
    for val in special_values:
        out.loc[s == val] = f"Special:{val}"

    valid = s.notna() & ~s.isin(special_values)
    sv = s.loc[valid]

    # Categorical / low-cardinality
    if (
        not pd.api.types.is_numeric_dtype(sv)
        or sv.nunique() <= categorical_threshold
    ):
        out.loc[valid] = sv.astype(str)
        return out

    if method == "quantile":
        binned = pd.qcut(
            sv,
            q=n_bins,
            duplicates="drop"
        )

    elif method == "equal_width":
        binned = pd.cut(
            sv,
            bins=n_bins,
            duplicates="drop"
        )

    elif method == "custom":
        if custom_bins is None:
            raise ValueError(
                "custom_bins must be provided "
                "when method='custom'"
            )

        binned = pd.cut(
            sv,
            bins=custom_bins,
            include_lowest=True
        )

    else:
        raise ValueError(
            "method must be 'quantile', "
            "'equal_width', or 'custom'"
        )

    # Keep Interval objects for correct ordering
    out.loc[valid] = binned

    return out

def bad_rate_by_bins(
    df,
    attribute,
    target,
    n_bins=10,
    method="quantile",
    custom_bins=None,
    special_values=None,
    min_bin_count=1
):
    """
    Calculate default/bad rate by attribute bin.

    Assumes:
        target = 1 => bad/default
        target = 0 => good
    """

    tmp = df[[attribute, target]].copy()

    tmp["bin"] = make_bins(
        tmp[attribute],
        n_bins=n_bins,
        method=method,
        custom_bins=custom_bins,
        special_values=special_values
    )

    tmp = tmp.dropna(
        subset=["bin", target]
    )

    out = (
        tmp.groupby("bin", observed=True, sort=False)
        .agg(
            count=(target, "size"),
            bad_count=(target, "sum"),
            bad_rate=(target, "mean")
        )
        .reset_index()
    )

    out["good_count"] = (
        out["count"] - out["bad_count"]
    )

    out["good_rate"] = 1 - out["bad_rate"]

    out["volume_pct"] = (
        out["count"] / out["count"].sum()
    )

    out = out[
        out["count"] >= min_bin_count
    ].copy()

    # Sort numeric interval bins
    def sort_key(x):
        if isinstance(x, pd.Interval):
            return x.left
        if str(x).startswith("Special:"):
            return np.inf - 1
        if x == "Missing":
            return np.inf
        return np.nan

    out["_sort"] = out["bin"].apply(sort_key)

    if out["_sort"].notna().any():
        out = (
            out
            .sort_values("_sort")
            .drop(columns="_sort")
            .reset_index(drop=True)
        )
    else:
        out = out.drop(columns="_sort")

    out["bin"] = out["bin"].astype(str)

    return out

br = bad_rate_by_bins(
    df=df,
    attribute="REV5627",
    target="default_flag",
    n_bins=10,
    method="quantile",
    special_values=[-99999, -99998, -99997]
)

display(br)

def calculate_iv(
    df,
    attribute,
    target,
    n_bins=10,
    method="quantile",
    custom_bins=None,
    special_values=None,
    smoothing=0.5
):
    """
    Calculate Weight of Evidence and Information Value.

    Assumes:
        target = 1 => bad
        target = 0 => good

    Returns
    -------
    iv_total : float
    iv_table : pd.DataFrame
    """

    tmp = df[[attribute, target]].copy()

    tmp["bin"] = make_bins(
        tmp[attribute],
        n_bins=n_bins,
        method=method,
        custom_bins=custom_bins,
        special_values=special_values
    )

    tmp = tmp.dropna(
        subset=["bin", target]
    )

    agg = (
        tmp.groupby("bin", observed=True, sort=False)
        .agg(
            count=(target, "size"),
            bad=(target, "sum")
        )
        .reset_index()
    )

    agg["good"] = (
        agg["count"] - agg["bad"]
    )

    # Smoothing avoids division by zero
    n_bins_actual = len(agg)

    total_bad = agg["bad"].sum()
    total_good = agg["good"].sum()

    agg["bad_dist"] = (
        agg["bad"] + smoothing
    ) / (
        total_bad
        + smoothing * n_bins_actual
    )

    agg["good_dist"] = (
        agg["good"] + smoothing
    ) / (
        total_good
        + smoothing * n_bins_actual
    )

    agg["woe"] = np.log(
        agg["good_dist"]
        / agg["bad_dist"]
    )

    agg["iv_component"] = (
        agg["good_dist"]
        - agg["bad_dist"]
    ) * agg["woe"]

    iv_total = agg["iv_component"].sum()

    agg["bad_rate"] = (
        agg["bad"] / agg["count"]
    )

    agg["volume_pct"] = (
        agg["count"] / agg["count"].sum()
    )

    agg["bin"] = agg["bin"].astype(str)

    return iv_total, agg

iv, iv_detail = calculate_iv(
    df=df,
    attribute="REV5627",
    target="default_flag",
    n_bins=10,
    method="quantile"
)

print("IV:", iv)
display(iv_detail)

def univariate_auc_gini(
    df,
    attribute,
    target,
    special_values=None
):
    """
    Calculate univariate AUC/Gini for a numeric attribute.

    Returns both raw and direction-adjusted values.
    """

    special_values = special_values or []

    tmp = df[[attribute, target]].copy()

    # Remove missing and special values for raw numeric AUC
    mask = (
        tmp[attribute].notna()
        & tmp[target].notna()
        & ~tmp[attribute].isin(special_values)
    )

    tmp = tmp.loc[mask]

    if len(tmp) == 0:
        return {
            "attribute": attribute,
            "auc_raw": np.nan,
            "auc_strength": np.nan,
            "gini_raw": np.nan,
            "gini_strength": np.nan,
            "direction": None
        }

    if tmp[target].nunique() < 2:
        return {
            "attribute": attribute,
            "auc_raw": np.nan,
            "auc_strength": np.nan,
            "gini_raw": np.nan,
            "gini_strength": np.nan,
            "direction": None
        }

    auc_raw = roc_auc_score(
        tmp[target],
        tmp[attribute]
    )

    # Direction-independent discrimination
    if auc_raw >= 0.5:
        auc_strength = auc_raw
        direction = "higher attribute -> higher default risk"
    else:
        auc_strength = 1 - auc_raw
        direction = "higher attribute -> lower default risk"

    gini_raw = 2 * auc_raw - 1

    gini_strength = (
        2 * auc_strength - 1
    )

    return {
        "attribute": attribute,
        "auc_raw": auc_raw,
        "auc_strength": auc_strength,
        "gini_raw": gini_raw,
        "gini_strength": gini_strength,
        "direction": direction
    }

auc_result = univariate_auc_gini(
    df,
    attribute="REV5627",
    target="default_flag"
)

auc_result

def calculate_mutual_information(
    df,
    attributes,
    target,
    sample_size=None,
    random_state=42
):
    """
    Calculate mutual information between attributes
    and binary default target.

    Parameters
    ----------
    attributes : list[str]
    """

    cols = attributes + [target]

    tmp = df[cols].copy()

    if sample_size is not None and len(tmp) > sample_size:
        tmp = tmp.sample(
            n=sample_size,
            random_state=random_state
        )

    y = tmp[target]

    results = []

    for attr in attributes:

        x = tmp[[attr]].copy()

        # MI cannot directly handle NaN
        if pd.api.types.is_numeric_dtype(x[attr]):

            # Simple missing treatment
            median = x[attr].median()

            x[attr] = x[attr].fillna(
                median
            )

            discrete = False

        else:

            x[attr] = (
                x[attr]
                .fillna("Missing")
                .astype("category")
                .cat.codes
            )

            discrete = True

        mi = mutual_info_classif(
            x,
            y,
            discrete_features=[discrete],
            random_state=random_state
        )[0]

        results.append({
            "attribute": attr,
            "mutual_information": mi
        })

    return (
        pd.DataFrame(results)
        .sort_values(
            "mutual_information",
            ascending=False
        )
        .reset_index(drop=True)
    )

mi_df = calculate_mutual_information(
    df=df,
    attributes=top_100_rv_features,
    target="default_flag",
    sample_size=300_000
)

def evaluate_features_against_default(
    df,
    attributes,
    target,
    n_bins=10,
    method="quantile",
    special_values=None,
    mi_sample_size=300_000,
    random_state=42
):

    results = []

    # MI once for all features
    mi_df = calculate_mutual_information(
        df=df,
        attributes=attributes,
        target=target,
        sample_size=mi_sample_size,
        random_state=random_state
    )

    mi_map = dict(
        zip(
            mi_df["attribute"],
            mi_df["mutual_information"]
        )
    )

    for attr in attributes:

        # IV
        try:
            iv, _ = calculate_iv(
                df=df,
                attribute=attr,
                target=target,
                n_bins=n_bins,
                method=method,
                special_values=special_values
            )
        except Exception:
            iv = np.nan

        # AUC / Gini
        try:
            auc_result = univariate_auc_gini(
                df=df,
                attribute=attr,
                target=target,
                special_values=special_values
            )
        except Exception:
            auc_result = {
                "auc_raw": np.nan,
                "auc_strength": np.nan,
                "gini_raw": np.nan,
                "gini_strength": np.nan,
                "direction": None
            }

        results.append({
            "attribute": attr,

            "iv": iv,

            "auc_raw":
                auc_result["auc_raw"],

            "auc_strength":
                auc_result["auc_strength"],

            "gini_raw":
                auc_result["gini_raw"],

            "gini_strength":
                auc_result["gini_strength"],

            "direction":
                auc_result["direction"],

            "mutual_information":
                mi_map.get(attr, np.nan)
        })

    return pd.DataFrame(results)

default_relationship = evaluate_features_against_default(
    df=df,
    attributes=top_100_rv_features,
    target="default_flag",
    n_bins=10,
    method="quantile",
    special_values=[
        -99999,
        -99998,
        -99997
    ],
    mi_sample_size=300_000
)
