"""Model training pipeline for the Pima Indians Diabetes dataset."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             auc, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:  # XGBoost is optional during import time for faster CI feedback
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - handled gracefully at runtime
    raise SystemExit(
        "xgboost is required for this project. Please install it via `pip install xgboost`."
    ) from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/diabetes.csv")
REPORTS_DIR = Path("reports")
ARTIFACTS_DIR = Path(".")
PROCESSED_TRAIN_PATH = Path("data/processed_train.csv")
PROCESSED_TEST_PATH = Path("data/processed_test.csv")
SCALER_PATH = Path("scaler.pkl")
MODEL_PATH = Path("best_model.pkl")
METRICS_PATH = REPORTS_DIR / "model_performance.csv"
SUMMARY_PATH = REPORTS_DIR / "summary_statistics.csv"
MISSING_PATH = REPORTS_DIR / "missing_values.csv"
ZERO_ANALYSIS_PATH = REPORTS_DIR / "zero_value_analysis.csv"
ROC_DIR = REPORTS_DIR / "roc_curves"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"
FEATURE_IMPORTANCE_DIR = REPORTS_DIR / "feature_importance"
SHAP_DIR = REPORTS_DIR / "shap"

ZERO_IMPUTED_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = -1


# ---------------------------------------------------------------------------
# Utility classes and functions
# ---------------------------------------------------------------------------

class ZeroMedianImputer(BaseEstimator, TransformerMixin):
    """Replace zero values in specified columns with the median of non-zero values."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        if isinstance(X, pd.DataFrame):
            data = X.copy()
            self.feature_names_in_ = list(data.columns)
        else:
            data = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", self.columns))
            self.feature_names_in_ = list(data.columns)

        self.medians_ = {}
        for column in self.columns:
            if column not in data:
                raise ValueError(f"Column '{column}' not found in input data during fitting.")
            series = data[column].replace(0, np.nan)
            median = series.median()
            if np.isnan(median):
                median = 0.0
            self.medians_[column] = median
        return self

    def transform(self, X: pd.DataFrame):  # noqa: D401 - sklearn API
        check_is_fitted(self, "medians_")
        if isinstance(X, pd.DataFrame):
            data = X.copy()
        else:
            data = pd.DataFrame(X, columns=self.feature_names_in_)

        for column, median in self.medians_.items():
            data[column] = data[column].replace(0, median)
        return data


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    best_params: Dict[str, float]
    estimator: Pipeline


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def ensure_directories() -> None:
    for directory in [
        REPORTS_DIR,
        ROC_DIR,
        CONFUSION_DIR,
        FEATURE_IMPORTANCE_DIR,
        SHAP_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    logging.info("Loading dataset from %s", path)
    data = pd.read_csv(path)
    logging.debug("Dataset head:\n%s", data.head())
    logging.info("Dataset shape: %s", data.shape)
    logging.info("Columns: %s", ", ".join(data.columns))
    return data


def perform_eda(data: pd.DataFrame) -> None:
    logging.info("Performing exploratory data analysis")
    summary = data.describe()
    summary.to_csv(SUMMARY_PATH)

    missing = data.isna().sum()
    missing.to_csv(MISSING_PATH, header=["missing_count"])

    zero_analysis = {
        column: int((data[column] == 0).sum())
        for column in data.columns if column != "Outcome"
    }
    zero_df = pd.DataFrame.from_dict(zero_analysis, orient="index", columns=["zero_count"])
    zero_df.to_csv(ZERO_ANALYSIS_PATH)

    ax_array = data.hist(bins=20, figsize=(12, 10))
    for ax in np.array(ax_array).flatten():
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_histograms.png")
    plt.close()

    pair_grid = sns.pairplot(data, hue="Outcome", diag_kind="hist")
    pair_grid.fig.suptitle("Feature Pairplot", y=1.02)
    pair_grid.fig.savefig(REPORTS_DIR / "pairplot.png")
    plt.close(pair_grid.fig)

    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png")
    plt.close()


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logging.info("Training samples: %s, Testing samples: %s", X_train.shape[0], X_test.shape[0])
    return X_train, X_test, y_train, y_test


def build_preprocessor(feature_names: Iterable[str]) -> Pipeline:
    zero_imputer = ZeroMedianImputer(columns=ZERO_IMPUTED_COLUMNS)
    numeric_pipeline = Pipeline(
        steps=[
            ("zero_imputer", zero_imputer),
            ("simple_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(feature_names)),
        ]
    )
    return preprocessor


def create_model_candidates() -> Dict[str, Tuple[BaseEstimator, Dict[str, Iterable[float]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_estimators=200,
            ),
            {
                "classifier__max_depth": [3, 4, 5],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__subsample": [0.8, 1.0],
            },
        ),
        "Support Vector Machine": (
            SVC(probability=True),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
            },
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [3, 5, 7, 11],
                "classifier__weights": ["uniform", "distance"],
            },
        ),
    }


def evaluate_model(
    name: str,
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    logging.info("Evaluating model %s", name)
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    logging.debug("Metrics for %s: %s", name, metrics)
    return metrics


def plot_confusion_matrix(name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    cm = confusion_matrix(y_test, estimator.predict(X_test))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(CONFUSION_DIR / f"{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()


def plot_roc_curve(name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_proba = estimator.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_DIR / f"{name.replace(' ', '_').lower()}_roc.png")
    plt.close()


def plot_feature_importance(name: str, estimator: Pipeline, feature_names: Iterable[str]) -> None:
    classifier = estimator.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        features = np.array(list(feature_names))[sorted_idx]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances[sorted_idx], y=features)
        plt.title(f"Feature Importance - {name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_DIR / f"{name.replace(' ', '_').lower()}_feature_importance.png")
        plt.close()


def generate_shap_summary(best_result: ModelResult, X_sample: pd.DataFrame) -> None:
    try:
        import shap
    except ImportError:  # pragma: no cover - optional dependency
        logging.warning("SHAP is not installed. Skipping explainability plot.")
        return

    classifier = best_result.estimator.named_steps["classifier"]
    preprocessor = best_result.estimator.named_steps["preprocessor"]

    X_processed = preprocessor.transform(X_sample)

    feature_names = None
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "predict_proba") and hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)
        plt.figure()
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_summary.png")
        plt.close()
    else:
        logging.info("Skipping SHAP summary: supported only for tree-based models in this pipeline.")


def train_models(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Dict[str, ModelResult]:
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: Dict[str, ModelResult] = {}
    candidates = create_model_candidates()

    for name, (estimator, param_grid) in candidates.items():
        logging.info("Training %s", name)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring="f1",
            n_jobs=N_JOBS,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_pipeline: Pipeline = grid_search.best_estimator_
        metrics = evaluate_model(name, best_pipeline, X_train, y_train)
        results[name] = ModelResult(
            name=name,
            metrics=metrics,
            best_params=grid_search.best_params_,
            estimator=best_pipeline,
        )
        logging.info("Best params for %s: %s", name, json.dumps(grid_search.best_params_))
    return results


def evaluate_on_test(results: Dict[str, ModelResult], X_test: pd.DataFrame, y_test: pd.Series, feature_names: Iterable[str]) -> ModelResult:
    records = []
    best_result: ModelResult | None = None

    for result in results.values():
        metrics = evaluate_model(result.name, result.estimator, X_test, y_test)
        records.append({"model": result.name, **metrics})
        plot_confusion_matrix(result.name, result.estimator, X_test, y_test)
        plot_roc_curve(result.name, result.estimator, X_test, y_test)
        plot_feature_importance(result.name, result.estimator, feature_names)

        if best_result is None or metrics["f1"] > best_result.metrics.get("f1", 0):
            best_result = ModelResult(
                name=result.name,
                metrics=metrics,
                best_params=result.best_params,
                estimator=result.estimator,
            )

    performance_df = pd.DataFrame.from_records(records).sort_values(by="f1", ascending=False)
    performance_df.to_csv(METRICS_PATH, index=False)

    assert best_result is not None, "At least one model should be trained."
    logging.info("Best model selected: %s with F1-score %.3f", best_result.name, best_result.metrics["f1"])
    return best_result


def save_artifacts(best_result: ModelResult) -> None:
    logging.info("Saving model artifacts")
    best_pipeline = best_result.estimator
    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    joblib.dump(preprocessor, SCALER_PATH)
    joblib.dump(classifier, MODEL_PATH)
    logging.info("Artifacts saved to %s and %s", SCALER_PATH, MODEL_PATH)


def save_processed_data(preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    logging.info("Saving processed datasets")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    train_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
    test_df = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)


def main(verbose: bool = False) -> None:
    configure_logging(verbose)
    ensure_directories()

    data = load_data()
    perform_eda(data)

    X_train, X_test, y_train, y_test = split_data(data)
    feature_names = X_train.columns

    preprocessor = build_preprocessor(feature_names)

    # Save processed datasets for reproducibility (use a temporary fitted copy)
    save_processed_data(preprocessor, X_train, X_test)

    # Rebuild an unfitted preprocessor for model training to avoid reusing fitted state
    preprocessor = build_preprocessor(feature_names)

    results = train_models(X_train, y_train, preprocessor)
    best_result = evaluate_on_test(results, X_test, y_test, feature_names)

    save_artifacts(best_result)

    # Generate SHAP summary using a small sample of the test set
    sample_size = min(100, len(X_test))
    sample_indices = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test.iloc[sample_indices]
    generate_shap_summary(best_result, X_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for diabetes prediction")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    main(verbose=args.verbose)
