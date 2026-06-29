import numpy as np
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
    expected = data[ZERO_IMPUTED_COLUMNS].replace(0, pd.NA).median()
    np.testing.assert_allclose(
        transformed.loc[0, ZERO_IMPUTED_COLUMNS].astype(float),
        expected.astype(float),
    )
