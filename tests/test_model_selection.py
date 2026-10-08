from unittest.mock import Mock

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import model_training as training


def test_train_models_records_grid_search_cv_scores(monkeypatch):
    X_train = pd.DataFrame({"Glucose": [100, 120]})
    y_train = pd.Series([0, 1])
    scores = iter([0.8, 0.6])

    class FakeGridSearch:
        def __init__(self, pipeline, **kwargs):
            assert kwargs["scoring"] == "f1"
            self.best_estimator_ = pipeline
            self.best_score_ = next(scores)
            self.best_params_ = {}

        def fit(self, X, y):
            assert X is X_train
            assert y is y_train

    monkeypatch.setattr(training, "GridSearchCV", FakeGridSearch)
    monkeypatch.setattr(training, "create_model_candidates", lambda: {
        "first": (LogisticRegression(), {}),
        "second": (LogisticRegression(), {}),
    })
    evaluation = Mock(side_effect=AssertionError("Selection must use CV, not fitted-data F1"))
    monkeypatch.setattr(training, "evaluate_model", evaluation)

    results = training.train_models(X_train, y_train, Mock())

    assert results["first"].cv_f1 == 0.8
    assert results["second"].cv_f1 == 0.6
    assert training.select_best_model(results) is results["first"]
    evaluation.assert_not_called()


def test_select_best_model_rejects_empty_candidates():
    with pytest.raises(ValueError, match="At least one model"):
        training.select_best_model({})


def test_real_cv_selection_ignores_better_holdout_challenger(monkeypatch, tmp_path):
    X_train = pd.DataFrame({"feature": range(40)})
    y_train = pd.Series([0] * 30 + [1] * 10)
    X_test = X_train.copy()
    y_test = 1 - y_train
    monkeypatch.setattr(training, "N_JOBS", 1)
    monkeypatch.setattr(training, "create_model_candidates", lambda: {
        "tree": (DecisionTreeClassifier(random_state=42), {"classifier__max_depth": [1, 2]}),
        "constant": (DummyClassifier(strategy="constant", constant=1), {}),
    })
    for name in ["plot_confusion_matrix", "plot_roc_curve", "plot_feature_importance"]:
        monkeypatch.setattr(training, name, Mock())

    # Fit actual GridSearchCV pipelines on a tiny synthetic training set.
    results = training.train_models(X_train, y_train, StandardScaler())
    winner = training.select_best_model(results)
    assert winner.name == "tree"
    assert winner.cv_f1 > results["constant"].cv_f1

    # A deliberately reversed holdout favors the CV loser. This comparison
    # exists only in the test; the production workflow evaluates the winner.
    challenger_metrics = training.evaluate_model("constant", results["constant"].estimator, X_test, y_test)
    winner_metrics = training.evaluate_on_test(winner, X_test, y_test, X_test.columns, tmp_path)
    assert challenger_metrics["f1"] > winner_metrics["f1"]
    assert training.select_best_model(results) is winner
    assert pd.read_csv(tmp_path / "holdout_performance.csv")["model"].tolist() == ["tree"]


@pytest.mark.parametrize("holdout_f1", [0.0, 1.0])
def test_main_keeps_cv_winner_regardless_of_holdout_score(monkeypatch, tmp_path, holdout_f1):
    X_train = pd.DataFrame({"Glucose": [100, 120]})
    X_test = pd.DataFrame({"Glucose": [150]})
    y_train = pd.Series([0, 1])
    y_test = pd.Series([1])
    loser = training.ModelResult("loser", 0.6, {}, Mock())
    winner = training.ModelResult("winner", 0.8, {}, Mock())
    results = {"loser": loser, "winner": winner}
    historical_report = tmp_path / "model_performance.csv"
    historical_report.write_text("historical results\n")

    monkeypatch.setattr(training, "ensure_directories", Mock(return_value=tmp_path))
    monkeypatch.setattr(training, "load_data", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "perform_eda", Mock())
    monkeypatch.setattr(training, "split_data", Mock(return_value=(X_train, X_test, y_train, y_test)))
    monkeypatch.setattr(training, "build_preprocessor", Mock())
    monkeypatch.setattr(training, "save_processed_data", Mock())
    train = Mock(return_value=results)
    monkeypatch.setattr(training, "train_models", train)
    selection = Mock(wraps=training.select_best_model)
    monkeypatch.setattr(training, "select_best_model", selection)

    def evaluate(name, estimator, X, y):
        selection.assert_called_once_with(results)
        assert name == "winner"
        assert estimator is winner.estimator
        assert X is X_test
        assert y is y_test
        return {"f1": holdout_f1}

    evaluation = Mock(side_effect=evaluate)
    monkeypatch.setattr(training, "evaluate_model", evaluation)
    plots = []
    for name in ["plot_confusion_matrix", "plot_roc_curve", "plot_feature_importance"]:
        plot = Mock()
        monkeypatch.setattr(training, name, plot)
        plots.append(plot)
    save = Mock()
    monkeypatch.setattr(training, "save_artifacts", save)
    monkeypatch.setattr(training, "generate_shap_summary", Mock())

    training.main()

    assert train.call_args.args[0] is X_train
    assert train.call_args.args[1] is y_train
    evaluation.assert_called_once()
    save.assert_called_once_with(winner, tmp_path)
    for plot in plots:
        plot.assert_called_once()
        assert plot.call_args.args[:2] == ("winner", winner.estimator)
        assert plot.call_args.args[-1] == tmp_path
    training.perform_eda.assert_called_once()
    assert training.perform_eda.call_args.args[-1] == tmp_path
    assert training.save_processed_data.call_args.args[-1] == tmp_path
    assert training.generate_shap_summary.call_args.args[-1] == tmp_path
    cv_report = pd.read_csv(tmp_path / "cv_selection.csv")
    assert cv_report.to_dict("records") == [
        {"model": "loser", "cv_f1": 0.6},
        {"model": "winner", "cv_f1": 0.8},
    ]
    holdout_report = pd.read_csv(tmp_path / "holdout_performance.csv")
    assert holdout_report.to_dict("records") == [{"model": "winner", "f1": holdout_f1}]
    assert historical_report.read_text() == "historical results\n"
