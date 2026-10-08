import pandas as pd

from model_training import ZERO_IMPUTED_COLUMNS, ZeroMedianImputer


def test_zero_median_imputer_replaces_zero_values():
    data = pd.DataFrame({
        "Glucose": [0, 120, 150],
        "BloodPressure": [0, 70, 80],
        "SkinThickness": [0, 20, 30],
        "Insulin": [0, 85, 90],
        "BMI": [0, 25.0, 30.0],
        "Pregnancies": [2, 3, 4],
    })

    transformer = ZeroMedianImputer(columns=ZERO_IMPUTED_COLUMNS)
    transformed = transformer.fit_transform(data)
    expected_medians = pd.Series({
        "Glucose": 135.0,
        "BloodPressure": 75.0,
        "SkinThickness": 25.0,
        "Insulin": 87.5,
        "BMI": 27.5,
    })
    pd.testing.assert_series_equal(
        transformed.loc[0, ZERO_IMPUTED_COLUMNS], expected_medians,
        check_names=False, check_dtype=False,
    )
    pd.testing.assert_frame_equal(transformed.loc[1:], data.loc[1:], check_dtype=False)
    pd.testing.assert_series_equal(transformed["Pregnancies"], data["Pregnancies"])
    pd.testing.assert_frame_equal(transformer.transform(data), transformed)
    assert data.loc[0, ZERO_IMPUTED_COLUMNS].eq(0).all()
