import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import model_training as training


def test_evaluation_runs_preserve_history_and_isolate_plots(monkeypatch, tmp_path):
    reports = tmp_path / "reports"
    historical = {}
    for directory in ["confusion_matrices", "roc_curves", "feature_importance", "shap"]:
        for name in ["tree", "logistic"]:
            path = reports / directory / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"historical plot")
            historical[path] = path.read_bytes()
    for name in ["best_model.pkl", "scaler.pkl"]:
        path = tmp_path / name
        path.write_bytes(b"historical model")
        historical[path] = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "REPORTS_DIR", reports)

    X = pd.DataFrame({"feature": range(8)})
    y = pd.Series([0] * 4 + [1] * 4)
    first_run = training.ensure_directories()
    tree = Pipeline([
        ("preprocessor", StandardScaler()),
        ("classifier", DecisionTreeClassifier(max_depth=1, random_state=42)),
    ]).fit(X, y)
    first_result = training.ModelResult("tree", 1.0, {}, tree)
    training.evaluate_on_test(first_result, X, y, X.columns, first_run)
    training.save_artifacts(first_result, first_run)
    (first_run / "shap" / "shap_summary.png").write_bytes(b"previous run SHAP")
    first_snapshot = {path: path.read_bytes() for path in first_run.rglob("*") if path.is_file()}

    second_run = training.ensure_directories()
    logistic = Pipeline([
        ("preprocessor", StandardScaler()),
        ("classifier", LogisticRegression()),
    ]).fit(X, y)
    second_result = training.ModelResult("logistic", 0.8, {}, logistic)
    training.evaluate_on_test(second_result, X, y, X.columns, second_run)
    training.save_artifacts(second_result, second_run)

    assert first_run != second_run
    assert first_run.parent == second_run.parent == reports / "runs"
    assert {path.relative_to(second_run).as_posix() for path in second_run.rglob("*.png")} == {
        "confusion_matrices/logistic_confusion_matrix.png",
        "roc_curves/logistic_roc.png",
    }
    assert not list((second_run / "feature_importance").iterdir())
    assert not list((second_run / "shap").iterdir())
    assert (first_run / "feature_importance" / "tree_feature_importance.png").exists()
    for run, name in [(first_run, "tree"), (second_run, "logistic")]:
        assert pd.read_csv(run / "holdout_performance.csv")["model"].tolist() == [name]
        assert (run / "best_model.pkl").is_file()
        assert (run / "scaler.pkl").is_file()
        for path in (run / "confusion_matrices").glob("*.png"):
            assert plt.imread(path).size > 0
    for path, contents in {**historical, **first_snapshot}.items():
        assert path.read_bytes() == contents
