import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model_training import build_preprocessor, load_data, split_data


def test_prediction_pipeline_outputs_probability():
    data = load_data()
    X_train, X_test, y_train, _ = split_data(data)
    preprocessor = build_preprocessor(X_train.columns)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=500)),
    ])

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test.iloc[[0]])[0, 1]
    assert 0.0 <= proba <= 1.0
