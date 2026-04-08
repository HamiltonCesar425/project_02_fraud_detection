import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src import data_loading, preprocessing, evaluate, api_predict


# ---------------------------------------------------------------------------
# Testes para data_loading
# ---------------------------------------------------------------------------
def test_load_data_success(tmp_path):
    csv_path = tmp_path / "fake.csv"
    df = pd.DataFrame({"feature1": [1, 2], "Class": [0, 1]})
    df.to_csv(csv_path, index=False)

    result = data_loading.load_data(str(csv_path))
    assert not result.empty
    assert "Class" in result.columns


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        data_loading.load_data("arquivo_inexistente.csv")


# ---------------------------------------------------------------------------
# Testes para preprocessing
# ---------------------------------------------------------------------------
def test_preprocess_data_success():
    df = pd.DataFrame({"feature1": [1, 2], "Class": [0, 1]})
    X_train, X_test, y_train, y_test = preprocessing.preprocess_data(df)
    assert X_train is not None
    assert y_train is not None


# ---------------------------------------------------------------------------
# Testes para evaluate
# ---------------------------------------------------------------------------
def test_evaluate_model_metrics():
    from sklearn.linear_model import LogisticRegression

    X = pd.DataFrame({"f1": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)

    report, matrix = evaluate.evaluate_model(model, X, y)
    assert "precision" in report
    assert matrix.shape == (2, 2)


# ---------------------------------------------------------------------------
# Testes para api_predict
# ---------------------------------------------------------------------------
def test_api_predict_returns_json():
    df = pd.DataFrame({"f1": [0.5, 0.9]})
    with patch("src.api_predict.model", MagicMock()) as mock_model:
        mock_model.predict_proba.return_value = [[0.2, 0.8], [0.7, 0.3]]
        result = api_predict.predict(df)
        assert isinstance(result, dict)
        assert "predictions" in result
